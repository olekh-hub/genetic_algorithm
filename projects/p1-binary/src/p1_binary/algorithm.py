from __future__ import annotations

import numpy as np
import pandas as pd

from ga_core.algorithm import GeneticAlgorithm
from ga_core.objective import function_registry
from p1_binary.config import BinaryConfig
from p1_binary.encoding import decode_chromosome, encode_chromosome
from p1_binary.population import initiate_population, evaluate_population
from p1_binary.crossover import crossover_ops
from p1_binary.mutation import mutation_ops
from p1_binary.inversion import inversion


class BinaryGA(GeneticAlgorithm[BinaryConfig]):

    def create_population(self, config: BinaryConfig) -> pd.DataFrame:
        return initiate_population(config.n_dims, config.n_bits, config.pop_size)

    def evaluate(self, population: pd.DataFrame, config: BinaryConfig) -> pd.DataFrame:
        return evaluate_population(population, config.a, config.b, config.n_dims, config.n_bits, self.objective)

    def apply_crossover(self, fn, parent1: list, parent2: list, config: BinaryConfig) -> list:
        return fn(parent1, parent2)

    def apply_mutation(self, fn, chromosome: list, config: BinaryConfig) -> list:
        return fn(chromosome)

    def post_mutate(self, chromosome: list, config: BinaryConfig) -> list:
        if np.random.random() < config.inversion_prob:
            return inversion(chromosome)
        return chromosome

    def decode(self, chromosome: list, config: BinaryConfig) -> list:
        return decode_chromosome(chromosome, config.a, config.b, config.n_dims, config.n_bits)

    def encode(self, variables: list[float], config: BinaryConfig) -> list:
        return encode_chromosome(variables, config.a, config.b, config.n_bits)

    def perturb(self, chromosome: list, config: BinaryConfig) -> list:
        c = chromosome.copy()
        n_flips = max(1, len(c) // 20)
        for idx in np.random.choice(len(c), size=n_flips, replace=False):
            c[idx] ^= 1
        return c


def algorithm(params: dict) -> dict:
    """Entry point for routes (backwards-compatible dict API)."""
    config = BinaryConfig.from_dict(params)
    objective = function_registry[config.function]()
    ga = BinaryGA(crossover_ops, mutation_ops, objective)
    return ga.run(config)
