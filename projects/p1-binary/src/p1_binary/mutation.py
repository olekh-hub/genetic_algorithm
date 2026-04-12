import numpy as np

from ga_core.registry import OperatorRegistry

mutation_ops = OperatorRegistry()


@mutation_ops.add('single_point')
def single_mutation(chromosome: list) -> list:
    c = chromosome.copy()
    ix = np.random.randint(0, len(c))
    c[ix] ^= 1
    return c


@mutation_ops.add('two_point')
def double_mutation(chromosome: list) -> list:
    c = chromosome.copy()
    ixs = np.random.choice(len(c), size=2, replace=False)
    for i in ixs:
        c[i] ^= 1
    return c


@mutation_ops.add('edge')
def edge_mutation(chromosome: list) -> list:
    c = chromosome.copy()
    ix = np.random.choice([0, len(c) - 1])
    c[ix] ^= 1
    return c
