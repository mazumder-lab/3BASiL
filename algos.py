"""
Compression algorithm constants and configurations.

This module defines the available compression algorithms and their categorizations
for the 3BASiL compression framework.


"""

from typing import Final, List

# Pure pruning algorithms that only apply sparsity without low-rank approximation
PURE_PRUNING_ALGORITHMS: Final[List[str]] = [
    "sparsegpt",
    "alps",
    "wanda",
]

# Sparse + Low-Rank (SLR) algorithms that combine pruning with low-rank decomposition
SLR_ALGORITHMS: Final[List[str]] = [
    "3basil",
    "oats",
    "hassle-free-sparsegpt",
    "hassle-free-alps",
    "eora_sparsegpt",
    "eora_alps",
]

# All supported compression algorithms
COMPRESSION_ALGORITHMS: Final[List[str]] = PURE_PRUNING_ALGORITHMS + SLR_ALGORITHMS

# Algorithms that use diagonal Hessian approximation (computationally efficient)
DIAGONAL_HESSIAN_ALGORITHMS: Final[List[str]] = [
    "wanda",
    "oats",
]

# Algorithms that use full Hessian computation (more accurate but slower)
FULL_HESSIAN_ALGORITHMS: Final[List[str]] = [
    "3basil",
    "hassle-free-sparsegpt",
    "hassle-free-alps",
    "eora_sparsegpt",
    "eora_alps",
    "sparsegpt",
    "alps",
]

ZERO_SHOT_TASKS: Final[List[str]] = [   
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "rte",
    "openbookqa",
    "boolq",
]

PPL_DATASETS: Final[List[str]] = [
    "wikitext2",
    "ptb",
    "c4",
]

# OWL deltas values extracted from the paper OATS (Optimal Adaptive Threshold Selection), 
# which is computed from the paper OWL (Optimal Weight Learning)
OWL_DELTAS = {
    "/home/mmakni/orcd/pool/Meta-Llama-3-8B": [
        0.030332429999999966, -0.02647441000000006, 0.06844161999999998,
        0.08346081999999999, 0.03263278999999997, 0.037657790000000024,
        0.03329596000000001, 0.03576663999999996, 0.036910620000000005,
        0.035973359999999954, 0.040336619999999934, 0.03760768000000003,
        0.025321199999999933, 0.04359360999999995, 0.029471359999999946,
        0.02946841999999994, 0.005488669999999973, 0.007622550000000006,
        0.0034176900000000288, -0.004022390000000042, -0.013647900000000046,
        -0.028375209999999984, -0.03668506000000005, -0.03929044000000004,
        -0.04648211000000002, -0.049364460000000054, -0.05529379000000001, -0.05152182000000005,
        -0.06396902999999998, -0.06046841999999997, -0.06466561000000004, -0.07653918000000004
    ]
}


def get_algorithm_type(alg: str) -> str:
    """
    Return the algorithm type for the given compression algorithm.
    
    Args:
        alg: Algorithm name
    
    Returns:
        'pure_pruning' or 'slr'
    
    Raises:
        ValueError: If algorithm is unknown
    """
    if alg in PURE_PRUNING_ALGORITHMS:
        return "pure_pruning"
    elif alg in SLR_ALGORITHMS:
        return "slr"
    else:
        raise ValueError(
            f"Unknown algorithm: '{alg}'. "
            f"Supported: {PURE_PRUNING_ALGORITHMS + SLR_ALGORITHMS}"
        )