from __future__ import annotations

import numpy as np
import pandas as pd

from ga_core.registry import OperatorRegistry

selection_ops = OperatorRegistry()


@selection_ops.add('best')
def best_selection(population: pd.DataFrame, n_best: int, _k: int = 3, maximize: bool = False) -> pd.DataFrame:
    return population.sort_values(by='fitness', ascending=not maximize).head(n_best).reset_index(drop=True)


@selection_ops.add('roulette')
def roulette_selection(population: pd.DataFrame, n_best: int, _k: int = 3, maximize: bool = False) -> pd.DataFrame:
    pop = population.copy()
    if maximize:
        min_f = pop['fitness'].min()
        pop['prob'] = pop['fitness'] - min_f + 1e-10
    else:
        max_f = pop['fitness'].max()
        pop['prob'] = max_f - pop['fitness'] + 1e-10
    total = pop['prob'].sum()
    if total == 0:
        pop['prob'] = 1.0
        total = pop['prob'].sum()
    selected = np.random.choice(pop.index, size=n_best, p=pop['prob'] / total)
    return pop.loc[selected].drop(columns=['prob']).reset_index(drop=True)


@selection_ops.add('tournament')
def tournament_selection(population: pd.DataFrame, n_best: int, k: int, maximize: bool = False) -> pd.DataFrame:
    selected = []
    for _ in range(n_best):
        group = population.sample(n=min(k, len(population)), replace=False)
        winner = group.sort_values(by='fitness', ascending=not maximize).head(1)
        selected.append(winner)
    return pd.concat(selected, ignore_index=True)
