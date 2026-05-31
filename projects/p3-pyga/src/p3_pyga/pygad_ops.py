from __future__ import annotations

import numpy as np


def arithmetic_crossover(parents, offspring_size, ga_instance):
    low = ga_instance.init_range_low
    high = ga_instance.init_range_high
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :].copy()
        p2 = parents[(idx + 1) % parents.shape[0], :].copy()
        k = np.random.random()
        child = np.clip(k * p1 + (1 - k) * p2, low, high)
        offspring.append(child)
        idx += 1
    return np.array(offspring)


def averaging_crossover(parents, offspring_size, ga_instance):
    low = ga_instance.init_range_low
    high = ga_instance.init_range_high
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :]
        p2 = parents[(idx + 1) % parents.shape[0], :]
        offspring.append(np.clip((p1 + p2) / 2.0, low, high))
        idx += 1
    return np.array(offspring)


def blend_alpha_crossover(parents, offspring_size, ga_instance):
    alpha = 0.5
    low = ga_instance.init_range_low
    high = ga_instance.init_range_high
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :]
        p2 = parents[(idx + 1) % parents.shape[0], :]
        child = []
        for x, y in zip(p1, p2):
            lo, hi = min(x, y), max(x, y)
            d = hi - lo
            child.append(np.clip(np.random.uniform(lo - alpha * d, hi + alpha * d), low, high))
        offspring.append(np.array(child))
        idx += 1
    return np.array(offspring)


def blend_alpha_beta_crossover(parents, offspring_size, ga_instance):
    alpha, beta = 0.5, 0.5
    low = ga_instance.init_range_low
    high = ga_instance.init_range_high
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :]
        p2 = parents[(idx + 1) % parents.shape[0], :]
        child = []
        for x, y in zip(p1, p2):
            lo, hi = min(x, y), max(x, y)
            d = hi - lo
            child.append(np.clip(np.random.uniform(lo - alpha * d, hi + beta * d), low, high))
        offspring.append(np.array(child))
        idx += 1
    return np.array(offspring)
