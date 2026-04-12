from __future__ import annotations

from ga_core.algorithm import GeneticAlgorithm
from ga_core.objective import function_registry
from p2_real.config import RealConfig
from p2_real.crossover import crossover_ops
from p2_real.mutation import mutation_ops


class RealGA(GeneticAlgorithm[RealConfig]):
    """Real-valued GA — uses base class defaults for population, evaluate, decode."""

    def apply_crossover(self, fn, parent1: list, parent2: list, config: RealConfig) -> list:
        return fn(parent1, parent2, config.a, config.b, key=self.objective)

    def apply_mutation(self, fn, chromosome: list, config: RealConfig) -> list:
        return fn(chromosome, config.a, config.b, sigma=config.sigma)


def algorithm(params: dict) -> dict:
    config = RealConfig.from_dict(params)
    objective = function_registry[config.function]()
    return RealGA(crossover_ops, mutation_ops, objective).run(config)
