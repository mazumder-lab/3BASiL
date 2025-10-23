"""
Configuration validation and computation utilities for 3BASiL compression framework.

This module provides unified validation, computation, and logging for compression configurations
across pure pruning and sparse+low-rank (S+LR) algorithms.
"""

import logging
from typing import Dict

from algos import PURE_PRUNING_ALGORITHMS, SLR_ALGORITHMS

logger = logging.getLogger(__name__)


def validate_and_log_compression_config(
    compression_alg: str,
    prunen: int,
    prunem: int,
    compression_ratio: float,
    rank_ratio: float,
    lora_num_ranks: int,
    do_train: bool,
) -> Dict[str, any]:
    """
    Validate compression configuration and return computed values.
    
    This function:
    1. Validates parameter combinations for consistency
    2. Computes sparsity and rank_ratio based on algorithm type
    3. Logs comprehensive configuration information
    4. Determines when to initialize LoRA components
    
    Args:
        compression_alg: Algorithm name (e.g., 'sparsegpt', '3basil', 'oats')
        prunen: N for N:M sparsity (0 for unstructured)
        prunem: M for N:M sparsity (0 for unstructured)
        compression_ratio: Target compression ratio (-1 to disable)
        rank_ratio: Ratio of parameters for low-rank (-1 for fixed rank)
        lora_num_ranks: Fixed LoRA rank (0 to use rank_ratio)
        do_train: Whether fine-tuning will be performed
    
    Returns:
        Dictionary with keys:
            - sparsity: Computed sparsity ratio (-1 for N:M pattern)
            - rank_ratio: Computed or validated rank_ratio
            - lora_num_ranks: Final LoRA rank to use
            - needs_lora_init_before_compression: Whether to init LoRA before compression
            - needs_lora_init_after_compression: Whether to init LoRA after compression
    
    Raises:
        ValueError: If configuration is invalid
    """
    
    # Determine algorithm type
    if compression_alg in SLR_ALGORITHMS:
        is_slr = True
    elif compression_alg in PURE_PRUNING_ALGORITHMS:
        is_slr = False
    elif compression_alg == "dense":
        is_slr = None
        return {
            'sparsity': -1,
            'rank_ratio': -1,
            'lora_num_ranks': 0,
            'needs_lora_init_before_compression': False,
            'needs_lora_init_after_compression': False,
        }
    else:
        raise ValueError(
            f"Unknown compression algorithm: '{compression_alg}'. "
            f"Supported algorithms: {PURE_PRUNING_ALGORITHMS + SLR_ALGORITHMS}"
        )
    
    # Validation: prunen/prunem consistency
    if (prunen == 0) != (prunem == 0):
        raise ValueError(
            f"prunen and prunem must be either both 0 (unstructured) or both non-zero (N:M). "
            f"Got prunen={prunen}, prunem={prunem}"
        )
    
    # Validation: Pure pruning doesn't use rank_ratio during compression
    if not is_slr and rank_ratio != -1:
        logger.warning(
            f"Pure pruning algorithm '{compression_alg}' doesn't use rank_ratio during compression. "
            f"Ignoring rank_ratio={rank_ratio} (will only be used if do_train=True for fine-tuning)."
        )
        rank_ratio = -1
    
    # ============================================================================
    # PURE PRUNING ALGORITHMS
    # ============================================================================
    if not is_slr:
        logger.info("=" * 80)
        logger.info(f"PURE PRUNING Configuration: {compression_alg}")
        logger.info("=" * 80)
        
        # Determine sparsity pattern
        if prunen != 0:
            sparsity = -1  # N:M pattern (sparsity handled by prunen/prunem)
            logger.info(f"  Sparsity Pattern: {prunen}:{prunem} structured")
            logger.info(f"  Sparsity Level: {prunen}/{prunem} = {prunen/prunem:.1%} dense, {1-prunen/prunem:.1%} sparse")
        else:
            sparsity = compression_ratio
            if sparsity < 0 or sparsity > 1:
                raise ValueError(
                    f"For unstructured sparsity, compression_ratio must be in [0, 1]. "
                    f"Got compression_ratio={compression_ratio}"
                )
            logger.info(f"  Sparsity Pattern: Unstructured")
            logger.info(f"  Sparsity Level: {sparsity:.1%} sparse, {1-sparsity:.1%} dense")
        
        # Check if we need LoRA for fine-tuning
        if do_train:
            if lora_num_ranks == 0:
                lora_num_ranks = 64  # Default
                logger.warning(
                    f"Pure pruning with fine-tuning (do_train=True) but lora_num_ranks=0. "
                    f"Auto-setting lora_num_ranks=64 for fine-tuning. "
                    f"Set --lora_num_ranks explicitly to change this."
                )
            logger.info(f"  Fine-tuning: Enabled (do_train=True)")
            logger.info(f"  LoRA: Will be initialized AFTER compression with rank={lora_num_ranks}")
            needs_lora_init_after = True
        else:
            if lora_num_ranks > 0:
                logger.warning(
                    f"lora_num_ranks={lora_num_ranks} set but do_train=False. "
                    f"LoRA will NOT be initialized (no fine-tuning planned)."
                )
                lora_num_ranks = 0
            logger.info(f"  Fine-tuning: Disabled (do_train=False)")
            needs_lora_init_after = False
        
        logger.info("=" * 80)
        
        return {
            'sparsity': sparsity,
            'rank_ratio': -1,
            'lora_num_ranks': lora_num_ranks,
            'needs_lora_init_before_compression': False,
            'needs_lora_init_after_compression': needs_lora_init_after,
        }
    
    # ============================================================================
    # SPARSE + LOW-RANK (S+LR) ALGORITHMS
    # ============================================================================
    else:
        logger.info("=" * 80)
        logger.info(f"SPARSE + LOW-RANK Configuration: {compression_alg}")
        logger.info("=" * 80)
        
        # Validation: S+LR requires low-rank component
        if lora_num_ranks == 0 and rank_ratio == -1:
            raise ValueError(
                f"S+LR algorithm '{compression_alg}' requires a low-rank component. "
                f"Either set --lora_num_ranks > 0 (for fixed rank) or "
                f"--rank_ratio > 0 (for compression-ratio-driven rank)."
            )
        
        if lora_num_ranks < 0:
            raise ValueError(f"lora_num_ranks must be >= 0, got {lora_num_ranks}")
        
        # Process N:M + LR configuration
        if prunen != 0:
            logger.info(f"  Sparsity Pattern: {prunen}:{prunem} structured + Low-Rank")
            logger.info(f"  Sparsity Level: {prunen}/{prunem} = {prunen/prunem:.1%} dense, {1-prunen/prunem:.1%} sparse")
            sparsity = -1  # N:M pattern
            
            if compression_ratio == -1:
                # Mode: N:M + Fixed Rank
                if lora_num_ranks == 0:
                    raise ValueError(
                        f"For N:M + Fixed Rank mode, lora_num_ranks must be > 0. "
                        f"Got lora_num_ranks={lora_num_ranks}"
                    )
                logger.info(f"  Mode: N:M + Fixed Rank")
                logger.info(f"  LoRA Rank: {lora_num_ranks} (fixed)")
                rank_ratio = -1
                
                # Calculate achieved compression
                dense_ratio = prunen / prunem
                param_reduction = 1 - dense_ratio  # Fraction removed by pruning
                logger.info(f"  Note: Overall compression depends on model architecture and rank={lora_num_ranks}")
                
            else:
                # Mode: N:M + Compression-Ratio-Driven
                if compression_ratio < 0 or compression_ratio > 1:
                    raise ValueError(
                        f"compression_ratio must be in [0, 1], got {compression_ratio}"
                    )
                
                nm_density = prunen / prunem
                if (1 - compression_ratio) <= nm_density:
                    raise ValueError(
                        f"Invalid configuration: compression_ratio={compression_ratio:.3f} is not achievable. "
                        f"For {prunen}:{prunem} sparsity (density={nm_density:.3f}), "
                        f"compression_ratio must be > {1 - nm_density:.3f} "
                        f"(i.e., more aggressive compression needed)."
                    )
                
                rank_ratio = (1 - nm_density - compression_ratio) / (1 - compression_ratio)
                logger.info(f"  Mode: N:M + Compression-Ratio-Driven")
                logger.info(f"  Target Compression: {compression_ratio:.3f} ({(1-compression_ratio)*100:.1f}% params retained)")
                logger.info(f"  Computed rank_ratio: {rank_ratio:.4f}")
                logger.info(f"  LoRA rank will be computed based on model architecture and rank_ratio")
        
        # Process Unstructured + LR configuration
        else:
            logger.info(f"  Sparsity Pattern: Unstructured + Low-Rank")
            
            if rank_ratio == -1:
                # Mode: Unstructured + Fixed Rank
                if lora_num_ranks == 0:
                    raise ValueError(
                        f"For Unstructured + Fixed Rank mode, lora_num_ranks must be > 0. "
                        f"Got lora_num_ranks={lora_num_ranks}"
                    )
                if compression_ratio < 0 or compression_ratio > 1:
                    raise ValueError(
                        f"For unstructured sparsity, compression_ratio must be in [0, 1]. "
                        f"Got compression_ratio={compression_ratio}"
                    )
                
                sparsity = compression_ratio
                logger.info(f"  Mode: Unstructured (ratio={sparsity:.3f}) + Fixed Rank")
                logger.info(f"  Unstructured Sparsity: {sparsity:.1%} sparse, {1-sparsity:.1%} dense")
                logger.info(f"  LoRA Rank: {lora_num_ranks} (fixed)")
                
            else:
                # Mode: Unstructured + Compression-Ratio-Driven
                if compression_ratio < 0 or compression_ratio > 1:
                    raise ValueError(
                        f"compression_ratio must be in [0, 1], got {compression_ratio}"
                    )
                if rank_ratio <= 0 or rank_ratio > 1:
                    raise ValueError(
                        f"rank_ratio must be in (0, 1], got {rank_ratio}"
                    )
                
                sparsity = compression_ratio + rank_ratio - compression_ratio * rank_ratio
                logger.info(f"  Mode: Unstructured + Compression-Ratio-Driven")
                logger.info(f"  Target Compression: {compression_ratio:.3f} ({(1-compression_ratio)*100:.1f}% params retained)")
                logger.info(f"  Rank Ratio: {rank_ratio:.3f}")
                logger.info(f"  Computed unstructured sparsity: {sparsity:.4f} ({sparsity*100:.1f}% sparse)")
                logger.info(f"  LoRA rank will be computed based on model architecture and rank_ratio")
        
        logger.info(f"  LoRA Initialization: BEFORE compression (S+LR requirement)")
        if do_train:
            logger.info(f"  Fine-tuning: Enabled (do_train=True) - will fine-tune compressed LoRA components")
        else:
            logger.info(f"  Fine-tuning: Disabled (do_train=False)")
        logger.info("=" * 80)
        
        return {
            'sparsity': sparsity,
            'rank_ratio': rank_ratio,
            'lora_num_ranks': lora_num_ranks,
            'needs_lora_init_before_compression': True,
            'needs_lora_init_after_compression': False,
        }

