import copy
import gc
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from peft.tuners import lora
from torch.optim.lr_scheduler import CosineAnnealingLR

from algos import OWL_DELTAS, PURE_PRUNING_ALGORITHMS, SLR_ALGORITHMS
from compress import LLM_Compressor, sync_time

logger = logging.getLogger(__name__)

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

def find_layers(module, layer_type=None, name=''):
    """
    Recursively find layers of a specific type in a module.
    
    Args:
        module: The module to search
        layer_type: The layer type to search for (e.g., nn.Linear, lora.LoraLayer)
                   If None, defaults to nn.Linear
        name: Current name prefix (used in recursion)
    
    Returns:
        Dictionary mapping layer names to layer modules
    """
    if layer_type is None:
        layer_type = nn.Linear
        
    if isinstance(module, layer_type):
        return {name: module}
    
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layer_type=layer_type, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def transformer_matching(
    pruned_layer: nn.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    tm_args=None,
    arg=None,
    model_type="",
    disable_low_rank: bool = False,
    device=None,
):
    # Extract parameters from tm_args or use defaults
    if tm_args is not None:
        chunk_size = tm_args.chunk_size
        n_iters = tm_args.n_iters
        lr_init = tm_args.lr_init
        use_squared = tm_args.use_squared
    else:
        # Fallback defaults
        chunk_size = 8
        n_iters = 20
        lr_init = 2e-5
        use_squared = True
    
    pruned_layer.train()
    if not disable_low_rank:
        is_trainable_ls = []
    for param in pruned_layer.parameters():
        if disable_low_rank:
            param.requires_grad = not param.requires_grad
        else:
            is_trainable_ls.append(param.requires_grad)
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(pruned_layer.parameters(), lr=lr_init)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr_init/5)
    
    with torch.enable_grad():
        for it in range(n_iters):
            epoch_loss = 0.0
            for chunk, outs_chunk in zip(torch.split(inputs, chunk_size, dim=0), torch.split(outputs, chunk_size, dim=0)):
                chunk = chunk.to(device)
                outs_chunk = outs_chunk.to(device)
                
                # student forward
                if model_type.lower() == "llama":
                    out_new = pruned_layer(chunk, position_embeddings=arg)
                elif model_type.lower() == "opt":
                    out_new = pruned_layer(chunk, attention_mask=arg)
                
                # teacher forward
                if isinstance(out_new, tuple):
                    out_new = out_new[0]
                
                diff = out_new - outs_chunk
                chunk_loss = diff.pow(2).sum() if use_squared else diff.norm().sum()
                
                # skip updating the first epoch
                if it > 0:
                    optimizer.zero_grad()
                    chunk_loss.backward()
                    for module in pruned_layer.modules():
                        if isinstance(module, lora.LoraLayer):
                            if hasattr(module, 'weight'):
                                module.weight.grad = module.weight.grad * (module.weight != 0).float()
                    optimizer.step()
                
                epoch_loss += chunk_loss.item()
                
                # Clean up intermediate tensors
                del chunk, outs_chunk, out_new, diff, chunk_loss
                torch.cuda.empty_cache()  # Force GPU memory cleanup
                
            scheduler.step()
            logger.info("transformer_matching epoch_loss=%.6f", epoch_loss)
    
    # Restore original requires_grad states
    for id_param, param in enumerate(pruned_layer.parameters()):
        if disable_low_rank:
            param.requires_grad = not param.requires_grad
        else:
            param.requires_grad = is_trainable_ls[id_param]
    
    # Additional cleanup
    del optimizer, scheduler
    if not disable_low_rank:
        del is_trainable_ls
    torch.cuda.empty_cache()
    

@torch.no_grad()
def llm_compressor(model, dataloader, dev, nsamples=128, prunen=2, prunem=4, sparsity=None, percdamp=0.01, hess_percdamp=0.05, compression_alg="3basil", hess_diag=False, n_iters_oats_hassle_free=80, enable_transformer_matching=True, enable_owl_deltas=False, model_type="", model_name=None, tm_args=None):
    logger.info("Starting compressor")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    sparsity_init = sparsity
    
    # For S+LR algorithms, LoRA is initialized BEFORE compression, so we need to unwrap the PEFT model
    # For pure pruning algorithms, LoRA is NOT present yet, so no unwrapping needed
    if compression_alg in SLR_ALGORITHMS:
        # Check if model is a PEFT model and unwrap it using isinstance for robust detection
        if isinstance(model, PeftModel):
            logger.info(f"Unwrapping PEFT model for S+LR algorithm: {compression_alg}")
            model = model.base_model.model
        elif hasattr(model, 'base_model') and isinstance(model.base_model, PeftModel):
            logger.info(f"Unwrapping wrapped PEFT model for S+LR algorithm: {compression_alg}")
            model = model.base_model.model
        else:
            logger.warning(f"Expected PEFT model for S+LR algorithm {compression_alg}, but model is not a PEFT model (type: {type(model)})")

    # Architecture-specific setup
    if model_type.lower() == "llama":
        layers = model.model.layers
        # Move LLaMA-specific components to device
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif model_type.lower() == "opt":
        layers = model.model.decoder.layers
        # Move OPT-specific components to device
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'llama' or 'opt'.")

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    if model_type.lower() == "opt":
        cache["attention_mask"] = None
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(dev)
            cache["i"] += 1
            if model_type.lower() == "opt":
                cache["attention_mask"] = kwargs.get('attention_mask')
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    # Move architecture-specific components back to CPU
    if model_type.lower() == "llama":
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif model_type.lower() == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache.get("attention_mask") if model_type.lower() == "opt" else None
    
    if enable_transformer_matching:
        outs_old = torch.zeros_like(inps)

    if enable_owl_deltas:
        if model_name is None:
            logger.warning("enable_owl_deltas is True but model_name is not provided. OWL deltas will not be applied.")
            owl_deltas = None
        else:
            owl_deltas = OWL_DELTAS.get(model_name)
            if owl_deltas is None:
                logger.warning(f"No OWL deltas found for model: {model_name}. OWL deltas will not be applied.")
            else:
                logger.info(f"Loaded OWL deltas for model: {model_name} ({len(owl_deltas)} values)")
    else:
        owl_deltas = None

    logger.info("Ready for layerwise compression")

    for i in range(len(layers)):
        if enable_owl_deltas and owl_deltas is not None:
            if i < len(owl_deltas):
                sparsity = sparsity_init - owl_deltas[i]
            else:
                logger.warning(f"OWL deltas index {i} out of range (max: {len(owl_deltas)-1}). Using original sparsity.")
                sparsity = sparsity_init
            
        layer = layers[i].to(dev)
        old_layer = copy.deepcopy(layer)
        
        if compression_alg in PURE_PRUNING_ALGORITHMS:
            full = find_layers(layer, layer_type=nn.Linear)
        else:
            full = find_layers(layer, layer_type=lora.LoraLayer)

        sequential = [list(full.keys())]
        gpts = {}

        for names in sequential:
            subset = {n: full[n] for n in names}

            for name in subset:
                gpts[name] = LLM_Compressor(subset[name], name)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            start_XTX_time = sync_time()
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                if model_type.lower() == "llama":
                    hidden_states = inps[j]
                    position_ids = torch.arange(0, model.seqlen, device=dev).unsqueeze(0)
                    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
                    outs[j] = layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]
                elif model_type.lower() == "opt":
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
            logger.info("Layer %d XTX construction time: %.3f", i, sync_time() - start_XTX_time)

            for name in subset:
                logger.info("Layer %d name=%s", i, name)
                logger.info("Pruning ...")
                compressor = gpts[name]
                assert isinstance(compressor, LLM_Compressor)
                start_layer_time = sync_time()
                if compression_alg == "3basil":
                    e = compressor.threebasil(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                if compression_alg == "hassle-free-sparsegpt":
                    e = compressor.hassle_free_sparsegpt(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp, n_iters=n_iters_oats_hassle_free)
                if compression_alg == "eora_sparsegpt":
                    e = compressor.eora_sparsegpt(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                if compression_alg == "eora_alps":
                    e = compressor.eora_alps(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                if compression_alg == "hassle-free-alps":
                    e = compressor.hassle_free_alps(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp, n_iters=n_iters_oats_hassle_free)
                if compression_alg == "oats":
                    e = compressor.oats(prunen=prunen, prunem=prunem, sparsity=sparsity, n_iters=n_iters_oats_hassle_free)
                if compression_alg == "alps":
                    e = compressor.alps(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                if compression_alg == "sparsegpt":
                    e = compressor.sparsegpt(prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                if compression_alg == "wanda":
                    e = compressor.wanda(prunen=prunen, prunem=prunem, sparsity=sparsity)
                end_layer_time = sync_time()
                logger.info("Layer %d: %s time: %.3f", i, name, end_layer_time - start_layer_time)

                compressor.free()

        if enable_transformer_matching:
            for j in range(nsamples):
                if model_type.lower() == "llama":
                    hidden_states = inps[j]
                    position_ids = torch.arange(0, model.seqlen, device=dev).unsqueeze(0)
                    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
                    outs_old[j] = old_layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]
                elif model_type.lower() == "opt":
                    outs_old[j] = old_layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            old_layer.cpu()
            transformer_matching(layer, inps, outs_old, tm_args=tm_args, arg=position_embeddings if model_type=="llama" else attention_mask, model_type=model_type, device=dev)

        for j in range(nsamples):
            torch.cuda.empty_cache()
            gc.collect()
            if model_type.lower() == "llama":
                hidden_states = inps[j]
                position_ids = torch.arange(0, model.seqlen, device=dev).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
                outs[j] = layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]
                del position_embeddings
            elif model_type.lower() == "opt":
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            
        layers[i] = layer.cpu()
        del old_layer
        del layer
        del gpts
        torch.cuda.empty_cache()
        gc.collect()
        
        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def llm_eval(model, testenc, dev, model_type=""):
    testenc = testenc.input_ids
    # Save seqlen and config before unwrapping (these are set on the outer wrapper)
    seqlen = model.seqlen
    config = model.config
    
    nsamples = testenc.numel() // seqlen

    model.eval()
    use_cache = config.use_cache
    config.use_cache = False
    
    # Check if model is a PEFT model and unwrap it
    # Use isinstance check for robust PEFT model detection
    if isinstance(model, PeftModel):
        # Unwrap PEFT model: PeftModel -> model (LlamaForCausalLM/OPTForCausalLM)
        logger.info(f"Detected PEFT model, unwrapping for evaluation")
        model = model.base_model.model
    elif hasattr(model, 'base_model') and isinstance(model.base_model, PeftModel):
        # Handle case where model is PeftModelForCausalLM wrapper
        logger.info(f"Detected wrapped PEFT model, unwrapping for evaluation")
        model = model.base_model.model
    
    # Architecture-specific setup
    if model_type.lower() == "llama":
        layers = model.model.layers
        # Move LLaMA-specific components to device
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif model_type.lower() == "opt":
        layers = model.model.decoder.layers
        # Move OPT-specific components to device
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        # For OPT, we need to handle final layer norm differently
        if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'llama' or 'opt'.")

    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    if model_type.lower() == "opt":
        cache["attention_mask"] = None

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(dev)
            cache["i"] += 1
            if model_type.lower() == "opt":
                cache["attention_mask"] = kwargs.get('attention_mask')
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    # Move architecture-specific components back to CPU
    if model_type.lower() == "llama":
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif model_type.lower() == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()

    torch.cuda.empty_cache()
    gc.collect()

    outs = torch.zeros_like(inps)
    attention_mask = cache.get("attention_mask") if model_type.lower() == "opt" else None
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if model_type.lower() == "llama":
                hidden_states = inps[j]
                position_ids = torch.arange(0, seqlen, device=dev).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
                outs[j] = layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]
            elif model_type.lower() == "opt":
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()
        inps, outs = outs, inps

    # Final processing - architecture specific
    if model_type.lower() == "llama":
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)
    elif model_type.lower() == "opt":
        if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        
        # Apply final normalization based on architecture
        if model_type.lower() == "llama":
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        elif model_type.lower() == "opt":
            if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                hidden_states = model.model.decoder.project_out(hidden_states)
        
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * seqlen):((i + 1) * seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    config.use_cache = use_cache
    return ppl.item()
