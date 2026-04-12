"""Variant registry — adding a new GA variant is one entry here.

Each variant provides:
- config_cls: the Config subclass (handles from_dict + validation)
- algorithm: the entry-point function (accepts raw params dict)
- crossover_ops / mutation_ops: for the /api/methods endpoint
"""

from p1_binary import BinaryConfig, algorithm as binary_algorithm, crossover_ops as binary_cross, mutation_ops as binary_mut
from p2_real import RealConfig, algorithm as real_algorithm, crossover_ops as real_cross, mutation_ops as real_mut

VARIANTS: dict[str, dict] = {
    'binary': {
        'config_cls': BinaryConfig,
        'algorithm': binary_algorithm,
        'crossover_ops': binary_cross,
        'mutation_ops': binary_mut,
    },
    'real': {
        'config_cls': RealConfig,
        'algorithm': real_algorithm,
        'crossover_ops': real_cross,
        'mutation_ops': real_mut,
    },
}
