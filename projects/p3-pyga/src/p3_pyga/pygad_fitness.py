from __future__ import annotations

import numpy as np

from ga_core.objective import function_registry
from p3_pyga.encoding import decode_binary_solution

EPS = 1e-10


def objective_fitness(raw: float, maximize: bool) -> float:
    if maximize:
        return raw
    return 1.0 / (raw + EPS)


def invert_fitness(fitness: float, maximize: bool) -> float:
    if maximize:
        return fitness
    return 1.0 / (fitness + EPS) - EPS


class PyGadFitness:
    """Picklable fitness callable (required for PyGAD process parallel mode)."""

    def __init__(
        self,
        representation: str,
        function: str,
        a: float,
        b: float,
        n_dims: int,
        n_bits: int,
        maximize: bool,
    ) -> None:
        self.representation = representation
        self.a = a
        self.b = b
        self.n_dims = n_dims
        self.n_bits = n_bits
        self.maximize = maximize
        self._objective = function_registry[function]()

    def __call__(self, ga_instance, solution, solution_idx):
        if self.representation == "binary":
            raw = self._objective(
                decode_binary_solution(solution, self.a, self.b, self.n_dims, self.n_bits),
            )
        else:
            raw = self._objective(solution.tolist())
        return objective_fitness(raw, self.maximize)


class PyGadGenerationCallback:
    """Picklable early-stopping callback for on_generation."""

    def __init__(
        self,
        maximize: bool,
        early_stop: bool,
        patience: int,
        min_delta: float,
    ) -> None:
        self.maximize = maximize
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.best_obj = float("inf") if not maximize else float("-inf")
        self.stall = 0

    def __call__(self, ga_instance):
        best_fit = float(np.max(ga_instance.last_generation_fitness))
        best_obj = invert_fitness(best_fit, self.maximize)
        if self.early_stop:
            improved = (
                (best_obj < self.best_obj - self.min_delta)
                if not self.maximize
                else (best_obj > self.best_obj + self.min_delta)
            )
            if improved:
                self.best_obj = best_obj
                self.stall = 0
            else:
                self.stall += 1
                if self.stall >= self.patience:
                    return "stop"
        else:
            self.best_obj = best_obj
        return None
