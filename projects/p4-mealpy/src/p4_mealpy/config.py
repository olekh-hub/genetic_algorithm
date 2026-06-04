from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class PSOConfig:
    w: float = 0.729
    c1: float = 2.05
    c2: float = 2.05

@dataclass
class ExperimentSpec:
    name: str
    n_dims: int = 2
    a: float = -5.0
    b: float = 5.0
    epochs: int = 100
    pop_size: int = 50
    pso_params: PSOConfig = field(default_factory=PSOConfig)
    run_p1: bool = True
    run_p2: bool = True
    extra_meta: Dict[str, Any] = field(default_factory=dict)