import gc
import math
import os
from typing import Any, Dict, List, Optional, Union, cast

import click
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    get_peft_model,
)
from peft.tuners import lora
from torch.utils import _pytree as pytree
from transformers.trainer import Trainer


def recursive_getattr(obj, attr):
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj

def save_full_model(trainer: Trainer) -> None:
    if not isinstance(trainer.model, (PeftModelForCausalLM, PeftModelForSequenceClassification)):
        raise TypeError(
            f"Expected `PeftModelForCausalLM`, or "
            f"`PeftModelForSequenceClassification`, "
            f"but got {type(trainer.model)}")
    if not trainer.args.should_save:
        return

    state_dict = trainer.model.state_dict()
    file_name = os.path.join(
        trainer.args.output_dir,
        "full_model.pth")
    torch.save(state_dict, file_name)
    click.secho(f"Saved model state dict to {file_name}", fg="green")


def prepare_model_for_lora(
    model,
    num_ranks: int,
    compression_ratio: float = 0.0,
    rank_ratio: float = 0.2,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> PeftModelForCausalLM:
        
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    # Create rank pattern based on weight sizes
    # If rank_ratio == -1: Use uniform fixed rank (num_ranks) for all modules
    # If rank_ratio > 0: Compute rank for each module based on its size
    if rank_ratio != -1 and rank_ratio > 0.0:
        # S+LR algorithm: compute rank pattern based on layer dimensions and compression
        rank_pattern = {}
        for name, module in model.named_modules():
            if any(target_module in name for target_module in target_modules):
                d_out, d_in = module.weight.size()
                rank = math.floor(rank_ratio * (1 - compression_ratio) * (d_out * d_in) / (d_out + d_in))
                rank_pattern[name] = rank
        print("-" * 100)
        print("rank_pattern:", rank_pattern)
        print()
        
        peft_config = LoraConfig(
            rank_pattern=rank_pattern,
            r=num_ranks,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM")
    elif rank_ratio == -1:
        # Post-pruning fine-tuning: use uniform fixed rank for all modules
        click.secho(f"Using uniform fixed rank r={num_ranks} for all LoRA modules", fg="yellow")
        peft_config = LoraConfig(
            r=num_ranks,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM")
    else:
        raise ValueError(f"Invalid rank_ratio={rank_ratio}. Must be -1 (fixed rank) or > 0 (computed rank)")

    new_model = get_peft_model(model, peft_config)    
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(new_model)}")
    return new_model