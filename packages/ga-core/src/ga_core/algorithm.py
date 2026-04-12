from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TypedDict

import numpy as np
import pandas as pd

from ga_core.config import GAConfig
from ga_core.objective import ObjectiveFunction
from ga_core.registry import OperatorRegistry
from ga_core.selection import selection_ops

C = TypeVar('C', bound=GAConfig)


class RunResult(TypedDict):
    best_fitness: float
    best_variables: list[float]
    best_history: list[float]
    best_positions_history: list[list[float]]
    population_history: list[dict]
    mean_history: list[float]
    std_history: list[float]
    elapsed: float
    total_epochs: int
    stopped_early: bool


class GeneticAlgorithm(ABC, Generic[C]):
    """Abstract base class for genetic algorithm variants.

    Operator registries and the objective function are **injected**
    through the constructor.  Subclasses only override the hooks
    that differ from the defaults.
    """

    def __init__(
        self,
        crossover_ops: OperatorRegistry,
        mutation_ops: OperatorRegistry,
        objective: ObjectiveFunction,
    ) -> None:
        self.crossover_ops = crossover_ops
        self.mutation_ops = mutation_ops
        self.objective = objective

    # -- hooks (override what differs) -------------------------------------

    def create_population(self, config: C) -> pd.DataFrame:
        """Default: random real-valued chromosomes in [a, b]."""
        data = [
            {'chromosome': np.random.uniform(config.a, config.b, size=config.n_dims).tolist()}
            for _ in range(config.pop_size)
        ]
        return pd.DataFrame(data)

    def evaluate(self, population: pd.DataFrame, config: C) -> pd.DataFrame:
        """Default: apply self.objective to each chromosome."""
        population = population.copy()
        population['fitness'] = population['chromosome'].apply(self.objective)
        return population

    @abstractmethod
    def apply_crossover(self, fn, parent1: list, parent2: list, config: C) -> list: ...

    @abstractmethod
    def apply_mutation(self, fn, chromosome: list, config: C) -> list: ...

    def decode(self, chromosome: list, config: C) -> list:
        """Default: chromosome IS the real-valued representation."""
        return list(chromosome)

    def encode(self, variables: list[float], config: C) -> list:
        """Default: variables ARE the chromosome."""
        return list(variables)

    def post_mutate(self, chromosome: list, config: C) -> list:
        """Optional extra operator after mutation. No-op by default."""
        return chromosome

    def perturb(self, chromosome: list, config: C) -> list:
        """Small noise for population seeding. Default: 5% uniform."""
        spread = (config.b - config.a) * 0.05
        return [
            float(np.clip(g + np.random.uniform(-spread, spread), config.a, config.b))
            for g in chromosome
        ]

    # -- shared evolutionary loop ------------------------------------------

    def run(self, config: C) -> RunResult:
        cross_fn = self.crossover_ops[config.crossover]
        mut_fn = self.mutation_ops[config.mutation]
        select_fn = selection_ops[config.selection]

        if config.start_point is not None:
            base = self.encode(config.start_point, config)
            rows = [{'chromosome': base}]
            for _ in range(config.pop_size - 1):
                rows.append({'chromosome': self.perturb(base, config)})
            current_pop = pd.DataFrame(rows)
        else:
            current_pop = self.create_population(config)

        best_history: list[float] = []
        best_positions_history: list[list[float]] = []
        population_history: list[dict] = []
        mean_history: list[float] = []
        std_history: list[float] = []

        best_so_far = float('inf') if not config.maximize else float('-inf')
        no_improvement = 0
        stopped_early = False

        t_start = time.time()

        for _ in range(config.epochs):
            evaluated = self.evaluate(current_pop, config)

            # -- selection (registry lookup, not if/elif) ------------------
            selected = select_fn(evaluated, config.n_best, config.k, config.maximize)

            # -- crossover -------------------------------------------------
            children: list[list] = []
            parents = selected['chromosome'].tolist()
            random.shuffle(parents)

            for i in range(0, len(parents) - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                if np.random.random() < config.crossover_prob:
                    children.append(self.apply_crossover(cross_fn, p1, p2, config))
                    children.append(self.apply_crossover(cross_fn, p2, p1, config))
                else:
                    children.append(p1.copy())
                    children.append(p2.copy())

            # -- mutation + post-mutation ----------------------------------
            for i in range(len(children)):
                if np.random.random() < config.mutation_prob:
                    children[i] = self.apply_mutation(mut_fn, children[i], config)
                children[i] = self.post_mutate(children[i], config)

            # -- next generation -------------------------------------------
            new_pop = [{'chromosome': ch} for ch in children]
            while len(new_pop) < config.pop_size - config.elite_size:
                new_pop.append({'chromosome': random.choice(children).copy()})

            elite = selection_ops['best'](evaluated, config.elite_size, config.maximize)
            for _, row in elite.iterrows():
                new_pop.append({'chromosome': row['chromosome'].copy()})

            current_pop = pd.DataFrame(new_pop)

            # -- tracking --------------------------------------------------
            if config.maximize:
                best_val = evaluated['fitness'].max()
                best_idx = evaluated['fitness'].idxmax()
            else:
                best_val = evaluated['fitness'].min()
                best_idx = evaluated['fitness'].idxmin()

            best_history.append(best_val)
            best_positions_history.append(
                self.decode(evaluated.loc[best_idx, 'chromosome'], config),
            )
            population_history.append({
                'pos': [self.decode(row['chromosome'], config) for _, row in evaluated.iterrows()],
                'fit': evaluated['fitness'].tolist(),
            })
            mean_history.append(evaluated['fitness'].mean())
            std_history.append(evaluated['fitness'].std())

            # -- early stopping --------------------------------------------
            if config.early_stop:
                improved = (
                    (best_val < best_so_far - config.min_delta)
                    if not config.maximize
                    else (best_val > best_so_far + config.min_delta)
                )
                if improved:
                    best_so_far = best_val
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= config.patience:
                        stopped_early = True
                        break
            else:
                best_so_far = best_val

        elapsed = time.time() - t_start

        final_eval = self.evaluate(current_pop, config)
        if config.maximize:
            best_row = final_eval.loc[final_eval['fitness'].idxmax()]
        else:
            best_row = final_eval.loc[final_eval['fitness'].idxmin()]

        return RunResult(
            best_fitness=float(best_row['fitness']),
            best_variables=self.decode(best_row['chromosome'], config),
            best_history=best_history,
            best_positions_history=best_positions_history,
            population_history=population_history,
            mean_history=mean_history,
            std_history=std_history,
            elapsed=elapsed,
            total_epochs=len(best_history),
            stopped_early=stopped_early,
        )
