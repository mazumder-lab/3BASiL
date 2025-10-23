# 3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs

Official repository for the NeurIPS 2025 paper:  
**3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs**

**3BASiL** is a novel and highly efficient Sparse + Low-Rank (S+LR) compression framework for Large Language Models.

**Key innovations:**
- **Highly efficient S+LR optimization**: (S+LR) decomposition of LLMs through a 3Block ADMM framework
- **Transformer Matching**: A universal post-compression technique that improves performance across different compression algorithms for both (S+LR) and pure pruning methods

This repository provides implementations of 3BASiL along with comprehensive baselines including OATS, HASSLE-free, EoRA, and pure pruning methods (SparseGPT, ALPS, Wanda) for Llama and OPT model families.

---

## Repository Structure

```
.
├── run.py                              # Main entry point for compression and evaluation
├── algos.py                            # Algorithm definitions and constants
├── compress.py                         # Core compression implementations
├── pipeline.py                         # High-level execution pipeline
├── config_utils.py                     # Configuration validation and utilities
├── loaders.py                          # Dataset loaders (C4, WikiText2, PTB)
├── calibration_testppl_loaders.py      # Additional data loading utilities
├── models/
│   ├── lora_utils.py                   # LoRA adapter initialization and management
│   └── __init__.py
├── experiments/
│   ├── mmlu_utils.py                   # MMLU evaluation utilities
│   ├── callback_utils.py               # Training callbacks
│   └── __init__.py
├── requirements.txt                    # Python package dependencies
└── LICENSE                             # Apache 2.0 License
```
---

## Requirements

The code requires Python 3.11 and the following packages (see `requirements.txt`):

```
torch>=2.0.1
transformers>=4.29.2
datasets>=2.12.0
peft>=0.12.0
lm-eval>=0.4.5
numpy>=1.24.3
click>=8.0.4
evaluate>=0.4.3
tqdm>=4.65.0
```
Optional: `wandb>=0.19.7` for experiment tracking (used in `pipeline.py` with fallback)


### Installation via Conda

A conda environment file is also provided:
```bash
conda env create -f environment.yml
```

Or install via pip:
```bash
pip install -r requirements.txt
```
---

## Usage

### 3BASiL: Sparse + Low-Rank (S+LR) Compression

The main contribution of this work is the **3BASiL algorithm** for Sparse + Low-Rank compression. Below are examples demonstrating the algorithm's capabilities.

#### 3BASiL with Transformer Matching (TM)

**Example 1: Unstructured 50% sparsity + rank 128 + TM**

```bash
python run.py \
    --model_name_or_path /path/to/Meta-Llama-3-8B \
    --output_dir ./results/Meta-Llama-3-8B_3BASiL_unstr-0.5_rank-128_TM \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --compression_alg 3basil \
    --compression_ratio 0.5 \
    --prunen 0 \
    --prunem 0 \
    --rank_ratio -1 \
    --lora_num_ranks 128 \
    --enable_transformer_matching True \
    --seqlen 2048 \
    --nsamples 128 \
    --percdamp 0.005 \
    --hess_diag True \
    --hess_percdamp 0.005 \
    --seed 42 \
    --n_iters_oats_hassle_free 40
```

**Example 2: Unstructured 70% sparsity + rank 128 without TM**

```bash
python run.py \
    --model_name_or_path /path/to/Meta-Llama-3-8B \
    --output_dir ./results/Meta-Llama-3-8B_3BASiL_unstr-0.7_rank-128_noTM \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --compression_alg 3basil \
    --compression_ratio 0.7 \
    --prunen 0 \
    --prunem 0 \
    --rank_ratio -1 \
    --lora_num_ranks 128 \
    --enable_transformer_matching False \
    --seqlen 2048 \
    --nsamples 128 \
    --percdamp 0.005 \
    --hess_diag True \
    --hess_percdamp 0.005 \
    --seed 42 \
    --n_iters_oats_hassle_free 40
```

#### 3BASiL with OWL (Optimal Weight Learning) Deltas

**Example 3: Unstructured 70% sparsity + rank 64 + OWL + TM**

```bash
python run.py \
    --model_name_or_path /path/to/Meta-Llama-3-8B \
    --output_dir ./results/Meta-Llama-3-8B_3BASiL_unstr-0.7_rk64_OWL_TM \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --compression_alg 3basil \
    --compression_ratio 0.7 \
    --prunen 0 \
    --prunem 0 \
    --rank_ratio -1 \
    --lora_num_ranks 64 \
    --enable_owl_deltas True \
    --enable_transformer_matching True \
    --seqlen 2048 \
    --nsamples 128 \
    --percdamp 0.005 \
    --hess_diag True \
    --hess_percdamp 0.005 \
    --seed 42 \
    --n_iters_oats_hassle_free 40
```


### Universality of Transformer Matching

A key finding of this work is that **Transformer Matching (TM) improves performance across different compression algorithms**, not just 3BASiL. Below we demonstrate TM applied to pure pruning methods.

#### SparseGPT 2:4 + Transformer Matching

```bash
python run.py \
    --model_name_or_path /path/to/Meta-Llama-3-8B \
    --output_dir ./results/Meta-Llama-3-8B_SparseGPT_nm2-4_TM \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --compression_alg sparsegpt \
    --prunen 2 \
    --prunem 4 \
    --compression_ratio -1 \
    --rank_ratio -1 \
    --enable_transformer_matching True \
    --seqlen 2048 \
    --nsamples 128 \
    --percdamp 0.005 \
    --hess_diag True \
    --hess_percdamp 0.005 \
    --seed 42
```

## Important Arguments

### Compression Parameters

- **`--compression_alg`**  
  Compression algorithm to use. Options: `sparsegpt`, `alps`, `wanda`, `3basil`, `oats`, `hassle-free-sparsegpt`, `hassle-free-alps`, `eora_sparsegpt`, `eora_alps`, `dense`

- **`--compression_ratio`**  
  Target compression ratio (fraction of parameters to remove). For unstructured sparsity or S+LR algorithms. Range: [0, 1]

- **`--prunen` / `--prunem`**  
  N:M structured sparsity pattern. Both must be 0 (unstructured) or both non-zero (e.g., 2:4, 4:8)

- **`--rank_ratio`**  
  Ratio of parameters allocated to low-rank component (for S+LR algorithms). Set to -1 for fixed rank mode.

- **`--lora_num_ranks`**  
  Fixed LoRA rank. Used when rank_ratio=-1 or for post-pruning fine-tuning.

### Calibration Parameters

- **`--nsamples`**  
  Number of calibration samples. More samples improve quality but increase runtime. Default: 128

- **`--seqlen`**  
  Sequence length for calibration samples. Default: 2048

- **`--calibration_dataset`**  
  Calibration dataset. Options: `c4`, `wikitext2`, `ptb`. Default: `c4`

- **`--percdamp`**  
  Damping parameter for Hessian regularization. Default: 0.01

- **`--hess_diag`**  
  Use diagonal Hessian approximation (faster, less accurate). Default: False

### Transformer Matching Parameters

- **`--enable_transformer_matching`**  
  Enable transformer matching post-compression optimization. Default: True

- **`--chunk_size`**  
  Mini-batch size for transformer matching. Default: 8

- **`--n_iters`**  
  Number of optimization iterations for transformer matching. Default: 20

- **`--lr_init`**  
  Initial learning rate for transformer matching. Default: 2e-5

### OWL (Optimal Weight Learning) Parameters

- **`--enable_owl_deltas`**  
  Enable non-uniform sparsity patterns using OWL deltas. Default: True

### Training Parameters

- **`--do_train`**  
  Enable fine-tuning after compression

- **`--do_eval`**  
  Enable evaluation (perplexity + zero-shot tasks)

- **`--dataset_name`**  
  LoRA fine-tuning dataset. Options: `c4`, or any HuggingFace dataset name

---

## Acknowledgements

This codebase has been heavily influenced by and builds upon the following repositories:

- **LQLoRA** [[GitHub](https://github.com/HanGuo97/lq-lora)] [[Paper](https://arxiv.org/abs/2311.12023)] - Low-rank plus quantized matrix decomposition techniques
- **OATS** [[GitHub](https://github.com/stephenqz/OATS)] - Optimal adaptive threshold selection methods
- **ALPS** [[GitHub](https://github.com/mazumder-lab/ALPS)] - Adaptive layerwise pruning strategies
- **SparseGPT** [[GitHub](https://github.com/IST-DASLab/sparsegpt)] - One-shot pruning baseline
- **HASSLE-free** [[GitHub](https://github.com/mazumder-lab/HASSLE-free)] - Hessian-free optimization framework

We thank the authors of these works for their contributions to the field and for making their code publicly available.

---

## License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or feedback, please open an issue on GitHub or contact me on *mmakni@mit.edu*.
