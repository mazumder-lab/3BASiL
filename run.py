#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code only supports decomposing Llama and OPT models into (S+LR) and pure pruning methods for now using the C4 calibration dataset for compression.

It has been heavily influenced by the following repositories:
1. LQLoRA https://github.com/HanGuo97/lq-lora
2. OATS https://github.com/stephenqz/OATS
3. ALPS https://github.com/mazumder-lab/ALPS
4. SparseGPT https://github.com/IST-DASLab/sparsegpt
5. HASSLE-free https://github.com/mazumder-lab/HASSLE-free/tree/main
"""
import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Optional

import datasets
import lm_eval
import torch
import transformers
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version

from algos import (
    COMPRESSION_ALGORITHMS,
    DIAGONAL_HESSIAN_ALGORITHMS,
    FULL_HESSIAN_ALGORITHMS,
    PPL_DATASETS,
    SLR_ALGORITHMS,
    ZERO_SHOT_TASKS,
)
from compress import sync_time
from config_utils import validate_and_log_compression_config
from experiments import mmlu_utils
from loaders import (
    get_c4_calibrationloader,
    get_c4_data,
    get_ppl_testloader,
    get_ptb_calibrationloader,
    get_wikitext2_calibrationloader,
)
from models import lora_utils
from pipeline import llm_compressor, llm_eval

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to save the model."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

@dataclass
class SparsityLRArguments:
    calibration_dataset: Optional[str] = field(
        default="c4", metadata={"help": "The name of the dataset to use for the layerwise reconstruction error."}
    )
    compression_ratio: float = field(
        default=0.0,
        metadata={"help": "Sparasity of pruning method"}
    )
    prunen: int = field(
        default=0,
        metadata={"help": "N for N:M sparsity"}
    )
    prunem: int = field(
        default=0,
        metadata={"help": "M for N:M sparsity"}
    )
    rank_ratio: float = field(
        default=-1,
        metadata={"help": "ratio of #params that go to rank"}
    )
    seqlen: int = field(
        default=2048,
        metadata={"help": "regularization term"}
    )
    nsamples: int = field(
        default=128,
        metadata={"help": "regularization term"}
    )
    percdamp: float = field(
        default=0.01,
        metadata={"help": "regularization term"}
    )
    hess_diag: bool = field(
        default=False,
        metadata={"help": "Incorporate more diagonal stuff in Hessian"}
    )
    hess_percdamp: float = field(
        default=0.015,
        metadata={"help": "regularization term"}
    )
    n_iters_oats_hassle_free: int = field(
        default=80,
        metadata={"help": "number of iterations for OATS and HASSLE-free algorithms"}
    )
    lora_num_ranks: int = field(default=8)
    lora_dropout: float = field(default=0.0)
    compression_alg: Optional[str] = field(default=None)
    lora_model_name: Optional[str] = field(default=None)
    load_model_checkpoint: Optional[str] = field(default=None)
    
    enable_owl_deltas: bool = field(
        default=True,
        metadata={"help": "enable OWL deltas: non-uniform sparsity pattern"}
    )
    
@dataclass
class TransformerMatchingArguments:
    enable_transformer_matching: bool = field(
        default=True,
        metadata={"help": "enable transformer matching"}
    )
    chunk_size: int = field(
        default=8,
        metadata={"help": "chunk size for transformer matching"}
    )
    n_iters: int = field(
        default=20,
        metadata={"help": "number of iterations for transformer matching"}
    )
    lr_init: float = field(
        default=2e-5,
        metadata={"help": "learning rate for transformer matching"}
    )
    use_squared: bool = field(
        default=True,
        metadata={"help": "use squared loss for transformer matching"}
    )
    disable_low_rank: bool = field(
        default=False,
        metadata={"help": "disable low-rank training for transformer matching"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, SparsityLRArguments, TransformerMatchingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, slr_args, tm_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, slr_args, tm_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("You are instantiating a new model from scratch. This is not supported by this script.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("You are instantiating a non-standard tokenizer from scratch. This is not supported by this script.")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )        
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Validate and compute compression configuration early to fail fast
    compression_config = validate_and_log_compression_config(
        compression_alg=slr_args.compression_alg,
        prunen=slr_args.prunen,
        prunem=slr_args.prunem,
        compression_ratio=slr_args.compression_ratio,
        rank_ratio=slr_args.rank_ratio,
        lora_num_ranks=slr_args.lora_num_ranks,
        do_train=training_args.do_train,
    )

    # Load and preprocess datasets only if training is required
    if slr_args.calibration_dataset is not None:
        # Load calibration data
        if slr_args.calibration_dataset == "c4":
            calibration_loader = get_c4_calibrationloader(nsamples=slr_args.nsamples, tokenizer=tokenizer, seqlen=slr_args.seqlen)
        elif slr_args.calibration_dataset == "wikitext2":
            calibration_loader = get_wikitext2_calibrationloader(nsamples=slr_args.nsamples, tokenizer=tokenizer, seqlen=slr_args.seqlen)
        elif slr_args.calibration_dataset == "ptb":
            calibration_loader = get_ptb_calibrationloader(nsamples=slr_args.nsamples, tokenizer=tokenizer, seqlen=slr_args.seqlen)
        else:
            raise ValueError(f"Not supported calibration dataset: {slr_args.calibration_dataset}")
    if training_args.do_train:
        if data_args.dataset_name == "c4":
            logger.warning(f"Using C4 dataset for training (`dataset_name` = {data_args.dataset_name})")
            raw_datasets = get_c4_data(split='train')
            # Wrap in DatasetDict for consistency with other dataset branches
            raw_datasets = datasets.DatasetDict({"train": raw_datasets})
        elif data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub for training.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming
            )
        else:
            data_files = {}
            dataset_args = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = list(raw_datasets["train"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output
        
        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                # Use multiprocessing with error handling
                try:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )
                except (RuntimeError, TimeoutError, ConnectionResetError, Exception) as e:
                    # Multiprocessing can fail for various reasons (memory, connection issues, etc.)
                    logger.warning(f"Multiprocessing failed with {data_args.preprocessing_num_workers} workers: {type(e).__name__}: {e}")
                    logger.warning("Retrying with single process (num_proc=1)...")
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=1,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            # Use multiprocessing with error handling
            try:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            except (RuntimeError, TimeoutError, ConnectionResetError, Exception) as e:
                # Multiprocessing can fail for various reasons (memory, connection issues, etc.)
                logger.warning(f"Multiprocessing failed with {data_args.preprocessing_num_workers} workers: {type(e).__name__}: {e}")
                logger.warning("Retrying with single process (num_proc=1)...")
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=1,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # Starting the Sparse + LR pruning procedure
    if "llama" in model_args.model_name_or_path.lower():
        model_type = "llama"
    elif "opt" in model_args.model_name_or_path.lower():
        model_type = "opt"
    else:
        raise ValueError(f"Model type not supported: {model_args.model_name_or_path}. Only Llama and Opt models are supported.")
    
    # Use computed values from early validation
    sparsity = compression_config['sparsity']
    rank_ratio = compression_config['rank_ratio']
    num_ranks = compression_config['lora_num_ranks']
    
    # Set compression parameters
    compression_ratio = slr_args.compression_ratio
    prunen = slr_args.prunen
    prunem = slr_args.prunem
    model.seqlen = slr_args.seqlen
    nsamples = slr_args.nsamples
    percdamp = slr_args.percdamp
    hess_diag = slr_args.hess_diag
    hess_percdamp = slr_args.hess_percdamp
    enable_transformer_matching = tm_args.enable_transformer_matching
    
    logger.info("="*80)
    logger.info("Runtime Configuration")
    logger.info("="*80)
    logger.info("SparsityLRArguments:")
    logger.info(f"  compression_alg: {slr_args.compression_alg}")
    logger.info(f"  seqlen: {model.seqlen}")
    logger.info(f"  nsamples: {nsamples}")
    logger.info(f"  percdamp: {percdamp}")
    logger.info(f"  hess_diag: {hess_diag}")
    logger.info(f"  hess_percdamp: {hess_percdamp}")
    logger.info(f"  enable_owl_deltas: {slr_args.enable_owl_deltas}")
    logger.info(f"  load_model_checkpoint: {slr_args.load_model_checkpoint}")
    logger.info(f"\nTransformerMatchingArguments:")
    logger.info(f"  enable_transformer_matching: {enable_transformer_matching}")
    logger.info(f"  chunk_size: {tm_args.chunk_size}")
    logger.info(f"  n_iters: {tm_args.n_iters}")
    logger.info(f"  lr_init: {tm_args.lr_init}")
    logger.info(f"  use_squared: {tm_args.use_squared}")
    logger.info(f"  disable_low_rank: {tm_args.disable_low_rank}")
    logger.info(f"\nModelArguments:")
    logger.info(f"  model_name_or_path: {model_args.model_name_or_path}")
    logger.info(f"  cache_dir: {model_args.cache_dir}")
    logger.info(f"  torch_dtype: {model_args.torch_dtype}")
    logger.info(f"  low_cpu_mem_usage: {model_args.low_cpu_mem_usage}")
    logger.info("="*80)
    model.eval()

    # Initialize LoRA BEFORE compression if needed (S+LR algorithms)
    if compression_config['needs_lora_init_before_compression']:
        logger.info("Initializing LoRA adapters before compression (S+LR requirement)...")
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=num_ranks,
            compression_ratio=compression_ratio,
            rank_ratio=rank_ratio,
            lora_dropout=slr_args.lora_dropout,
        )

    # Run compression
    if slr_args.compression_alg in COMPRESSION_ALGORITHMS:
        start_compression_time = sync_time()
        llm_compressor(model, calibration_loader, dev=training_args.device, nsamples=nsamples, prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, compression_alg=slr_args.compression_alg, hess_diag=hess_diag, hess_percdamp=hess_percdamp, n_iters_oats_hassle_free=slr_args.n_iters_oats_hassle_free, enable_transformer_matching=enable_transformer_matching, tm_args=tm_args, enable_owl_deltas=slr_args.enable_owl_deltas, model_type=model_type, model_name=model_args.model_name_or_path)
        # Save compressed model if output directory is configured
        if model_args.save_dir is not None:
            save_path = os.path.join(slr_args.save_dir, f"compressed_model_{slr_args.compression_alg}_sp{sparsity}_nm_{prunen}_{prunem}.pth")
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved compressed model to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save compressed model: {e}")
        end_compression_time = sync_time()
        logger.info("="*80)
        logger.info(f"Compression completed in {end_compression_time - start_compression_time:.2f}s")
        logger.info("="*80)
        del calibration_loader
        torch.cuda.empty_cache()
        gc.collect()
    # elif model_args.load_model_checkpoint is not None:            
    elif slr_args.compression_alg == "lora_checkpoint":
        if slr_args.load_model_checkpoint is None:
            raise ValueError("load_model_checkpoint must be specified when compression_alg='lora_checkpoint'")
        logger.info(f"Loading model checkpoint from {slr_args.load_model_checkpoint}")
        with torch.no_grad():
            state_dict = torch.load(slr_args.load_model_checkpoint)
            model.load_state_dict(state_dict)
            del state_dict
        logger.info("Model checkpoint loaded successfully")
    elif slr_args.compression_alg == "dense":
        logger.info("Skipping compression - proceeding with evaluation of dense baseline model.")
    else:
        raise ValueError(f"Compression algorithm {slr_args.compression_alg} not supported.")
    
    # Initialize LoRA AFTER compression if needed (Pure Pruning + Fine-tuning)
    if compression_config['needs_lora_init_after_compression']:
        logger.info("="*80)
        logger.info("Initializing LoRA adapters for fine-tuning (after pruning)...")
        logger.info("="*80)
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=num_ranks,
            compression_ratio=0,  # No compression constraint for post-pruning LoRA
            rank_ratio=-1,
            lora_dropout=slr_args.lora_dropout,
        )
        logger.info(f"LoRA adapters initialized with rank={num_ranks}")


    # NOTE: For opt-30b, evaluation is done only on PPL datasets, otherwise it results in Cuda OOM, if only using one GPU.
    if "opt-30b" in model_args.model_name_or_path.lower():
        if training_args.do_eval:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("*** Evaluate ***")
            model.eval()
            for dataset_name in PPL_DATASETS:
                testloader = get_ppl_testloader(dataset_name=dataset_name, tokenizer=tokenizer, seqlen=model.seqlen)
                logger.info(f"Evaluating on dataset: {dataset_name}")
                dataset_ppl = llm_eval(model, testloader, dev=training_args.device, model_type=model_type)
                logger.info(f"pretrain_{dataset_name}_ppl: {dataset_ppl:.4f}")
        return

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        callbacks=None,
    )

    # Evaluation
    if training_args.do_eval:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("*** Evaluate ***")
        model.eval()
        model.to(training_args.device)
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=training_args.per_device_eval_batch_size) 
        with torch.no_grad():
            ### LM Eval Harness ###
            zs_results = lm_eval.simple_evaluate(hflm, tasks=ZERO_SHOT_TASKS, num_fewshot=0, batch_size=training_args.per_device_eval_batch_size)['results']
            logger.info("Zero-shot evaluation results:")
            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}
            acc_avg = mmlu_utils.calculate_avg_accuracy(ZERO_SHOT_TASKS, zs_results)
            metric_vals['average_zero_shot'] = round(acc_avg, 4)
            logger.info(f"Zero-shot results: {metric_vals}")
        trainer.log_metrics("pretrain_zs_results", metric_vals)
        trainer.save_metrics("pretrain_zs_results", metric_vals)
        for dataset_name in PPL_DATASETS:
            testloader = get_ppl_testloader(dataset_name=dataset_name, tokenizer=tokenizer, seqlen=model.seqlen)
            logger.info(f"Evaluating on dataset: {dataset_name}")
            dataset_ppl = llm_eval(model, testloader, dev="cuda", model_type=model_type)
            trainer.log_metrics(f"pretrain_{dataset_name}_ppl", {f"{dataset_name}_ppl": dataset_ppl})
            trainer.save_metrics(f"pretrain_{dataset_name}_ppl", {f"{dataset_name}_ppl": dataset_ppl})


    # Training
    if training_args.do_train:
        torch.cuda.empty_cache()
        gc.collect()
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Evaluation Post Fine-Tuning
        if training_args.do_eval:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("*** Evaluate ***")
            model.eval()
            model.to(training_args.device)
            hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=training_args.per_device_eval_batch_size) 
            with torch.no_grad():
                ### LM Eval Harness ###
                zs_results = lm_eval.simple_evaluate(hflm, tasks=ZERO_SHOT_TASKS, num_fewshot=0, batch_size=training_args.per_device_eval_batch_size)['results']
                logger.info("Zero-shot evaluation results:")
                metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}
                acc_avg = mmlu_utils.calculate_avg_accuracy(ZERO_SHOT_TASKS, zs_results)
                metric_vals['average_zero_shot'] = round(acc_avg, 4)
                logger.info(f"Zero-shot results: {metric_vals}")
            trainer.log_metrics("post_zs_results", metric_vals)
            trainer.save_metrics("post_zs_results", metric_vals)
            
            for dataset_name in PPL_DATASETS:
                testloader = get_ppl_testloader(dataset_name=dataset_name, tokenizer=tokenizer, seqlen=model.seqlen)
                logger.info(f"Evaluating on dataset: {dataset_name}")
                dataset_ppl = llm_eval(model, testloader, dev=training_args.device, model_type=model_type)
                trainer.log_metrics(f"post_{dataset_name}_ppl", {f"{dataset_name}_ppl": dataset_ppl})
                trainer.save_metrics(f"post_{dataset_name}_ppl", {f"{dataset_name}_ppl": dataset_ppl})

if __name__ == "__main__":
    main()