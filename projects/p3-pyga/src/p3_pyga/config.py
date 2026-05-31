from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


Representation = Literal["binary", "real"]
Engine = Literal["custom", "pygad"]
ParallelMode = Literal["none", "thread", "process"]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    representation: Representation
    engine: Engine
    parallel: ParallelMode = "none"
    use_custom_ops: bool = False
    a: float = -2.048
    b: float = 2.048
    n_dims: int = 2
    n_bits: int = 50
    epochs: int = 100
    pop_size: int = 100
    n_best: int = 20
    elite_size: int = 2
    k: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    inversion_prob: float = 0.1
    sigma: float = 0.1
    selection: str = "tournament"
    crossover: str = "uniform"
    mutation: str = "single_point"
    function: str = "rosenbrock"
    maximize: bool = False
    early_stop: bool = False
    patience: int = 20
    min_delta: float = 1e-6
    parallel_workers: int = 4

    def to_params(self) -> dict:
        d = asdict(self)
        d.pop("experiment_id", None)
        d.pop("engine", None)
        d.pop("parallel", None)
        d.pop("use_custom_ops", None)
        d.pop("parallel_workers", None)
        d["variant"] = self.representation
        if self.engine == "pygad":
            d["implementation"] = f"pygad_{self.parallel}"
            if self.use_custom_ops:
                d["implementation"] += "_custom_ops"
        else:
            d["implementation"] = f"custom_{self.representation}"
        return d
