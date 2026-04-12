import numpy as np

from ga_core.registry import OperatorRegistry

mutation_ops = OperatorRegistry()


@mutation_ops.add('uniform')
def uniform_mutation(chromosome: list, a_bound: float, b_bound: float, **kwargs) -> list:
    c = chromosome.copy()
    ix = np.random.randint(0, len(c))
    c[ix] = np.random.uniform(a_bound, b_bound)
    return c


@mutation_ops.add('gaussian')
def gaussian_mutation(chromosome: list, a_bound: float, b_bound: float, sigma: float = 0.1) -> list:
    c = chromosome.copy()
    ix = np.random.randint(0, len(c))
    c[ix] = np.clip(c[ix] + np.random.normal(0, sigma), a_bound, b_bound)
    return c
