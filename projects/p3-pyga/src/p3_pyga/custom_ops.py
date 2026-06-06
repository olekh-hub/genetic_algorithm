from __future__ import annotations

import numpy as np


def custom_crossover(parents, offspring_size, ga_instance):
    """Single-point style crossover from Projekt 3 example_03."""
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        split = np.random.randint(0, offspring_size[1])
        parent1[split:] = parent2[split:]
        offspring.append(parent1)
        idx += 1
    return np.array(offspring)


def custom_mutation(offspring, ga_instance):
    """Perturb one random gene (example_03 style)."""
    low = ga_instance.init_range_low
    high = ga_instance.init_range_high
    for row in range(offspring.shape[0]):
        gene_idx = np.random.randint(0, offspring.shape[1])
        offspring[row, gene_idx] = np.clip(
            offspring[row, gene_idx] + np.random.uniform(-0.1, 0.1) * (high - low),
            low,
            high,
        )
    return offspring
