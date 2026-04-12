import numpy as np


def inversion(chromosome: list) -> list:
    c = chromosome.copy()
    a, b = sorted(np.random.choice(len(c), size=2, replace=False))
    c[a:b + 1] = c[a:b + 1][::-1]
    return c
