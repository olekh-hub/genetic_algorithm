from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ga_core.registry import OperatorRegistry

if TYPE_CHECKING:
    pass

crossover_ops = OperatorRegistry()


@crossover_ops.add('arithmetic')
def arithmetic_crossover(p1: list, p2: list, a_bound: float, b_bound: float, **_kw) -> list:
    k = np.random.random()
    return [np.clip(k * x + (1 - k) * y, a_bound, b_bound) for x, y in zip(p1, p2)]


@crossover_ops.add('linear')
def linear_crossover(p1: list, p2: list, a_bound: float, b_bound: float,
                     key: Callable[[list], float] | None = None, **_kw) -> list:
    """Linear crossover — produces 3 candidates, returns the best.

    *key* is the objective function used to compare candidates.
    If not provided, falls back to the first candidate (no ranking).
    """
    c1 = [0.5 * x + 0.5 * y for x, y in zip(p1, p2)]
    c2 = [1.5 * x - 0.5 * y for x, y in zip(p1, p2)]
    c3 = [-0.5 * x + 1.5 * y for x, y in zip(p1, p2)]
    candidates = [[np.clip(v, a_bound, b_bound) for v in c] for c in [c1, c2, c3]]
    if key is not None:
        return min(candidates, key=key)
    return candidates[0]


@crossover_ops.add('blend_alpha')
def blend_alpha_crossover(p1: list, p2: list, a_bound: float, b_bound: float,
                          alpha: float = 0.5, **_kw) -> list:
    child = []
    for x, y in zip(p1, p2):
        lo, hi = min(x, y), max(x, y)
        d = hi - lo
        child.append(np.clip(np.random.uniform(lo - alpha * d, hi + alpha * d), a_bound, b_bound))
    return child


@crossover_ops.add('blend_alpha_beta')
def blend_alpha_beta_crossover(p1: list, p2: list, a_bound: float, b_bound: float,
                                alpha: float = 0.5, beta: float = 0.5, **_kw) -> list:
    child = []
    for x, y in zip(p1, p2):
        lo, hi = min(x, y), max(x, y)
        d = hi - lo
        child.append(np.clip(np.random.uniform(lo - alpha * d, hi + beta * d), a_bound, b_bound))
    return child


@crossover_ops.add('averaging')
def averaging_crossover(p1: list, p2: list, a_bound: float, b_bound: float, **_kw) -> list:
    return [np.clip((x + y) / 2.0, a_bound, b_bound) for x, y in zip(p1, p2)]
