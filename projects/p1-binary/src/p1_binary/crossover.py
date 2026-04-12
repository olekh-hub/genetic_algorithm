import numpy as np

from ga_core.registry import OperatorRegistry

crossover_ops = OperatorRegistry()


@crossover_ops.add('single_point')
def single_point_crossover(first: list, second: list) -> list:
    ix = np.random.randint(1, len(first))
    return first[:ix] + second[ix:]


@crossover_ops.add('two_point')
def two_point_crossover(first: list, second: list) -> list:
    pts = sorted(np.random.choice(range(1, len(first)), size=2, replace=False))
    return first[:pts[0]] + second[pts[0]:pts[1]] + first[pts[1]:]


@crossover_ops.add('uniform')
def uniform_crossover(first: list, second: list) -> list:
    return [f if np.random.random() < 0.5 else s for f, s in zip(first, second)]


@crossover_ops.add('granular')
def granular_crossover(first: list, second: list, grain: int = 4) -> list:
    child = []
    use_first = True
    for i in range(0, len(first), grain):
        chunk = first[i:i + grain] if use_first else second[i:i + grain]
        child.extend(chunk)
        use_first = not use_first
    return child
