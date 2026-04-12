from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from p1_binary.encoding import get_fitness

if TYPE_CHECKING:
    from ga_core.objective import ObjectiveFunction


def initiate_population(n_dims: int, n_bits: int, size: int) -> pd.DataFrame:
    length = n_dims * n_bits
    data = [{'chromosome': np.random.randint(2, size=length).tolist()} for _ in range(size)]
    return pd.DataFrame(data)


def evaluate_population(population: pd.DataFrame, a: float, b: float,
                        n_dims: int, n_bits: int,
                        objective: ObjectiveFunction) -> pd.DataFrame:
    population = population.copy()
    population['fitness'] = population['chromosome'].apply(
        lambda x: get_fitness(x, a, b, n_dims, n_bits, objective)
    )
    return population
