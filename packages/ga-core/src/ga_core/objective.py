from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ga_core.registry import OperatorRegistry

function_registry = OperatorRegistry()


class ObjectiveFunction(ABC):
    """Abstract base for objective/test functions.

    Subclasses implement ``__call__`` to evaluate a point.
    The ``surface`` method is provided for free and works with any 2-D function.
    """

    name: str

    @abstractmethod
    def __call__(self, dims: list[float]) -> float: ...

    def surface(
        self,
        x1_min: float, x1_max: float,
        x2_min: float, x2_max: float,
        resolution: int = 80,
    ) -> tuple[list, list, list]:
        x = np.linspace(x1_min, x1_max, resolution)
        y = np.linspace(x2_min, x2_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self([X[i, j], Y[i, j]])
        return X.tolist(), Y.tolist(), Z.tolist()


@function_registry.add('rosenbrock')
class Rosenbrock(ObjectiveFunction):
    name = 'rosenbrock'

    def __call__(self, dims: list[float]) -> float:
        f_x = 0.0
        for i in range(len(dims) - 1):
            f_x += 100 * (dims[i + 1] - dims[i] ** 2) ** 2 + (1 - dims[i]) ** 2
        return f_x
