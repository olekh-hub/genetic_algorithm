from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GAConfig:
    """Base configuration shared by all GA variants."""

    a: float
    b: float
    n_dims: int
    epochs: int
    pop_size: int
    n_best: int
    elite_size: int
    k: int
    crossover_prob: float
    mutation_prob: float
    selection: str
    crossover: str
    mutation: str
    function: str = 'rosenbrock'
    maximize: bool = False
    early_stop: bool = False
    patience: int = 20
    min_delta: float = 1e-6
    start_point: list[float] | None = None

    def __post_init__(self) -> None:
        if self.a >= self.b:
            raise ValueError(f"a ({self.a}) must be less than b ({self.b})")
        if not 0 <= self.crossover_prob <= 1:
            raise ValueError(f"crossover_prob must be in [0, 1], got {self.crossover_prob}")
        if not 0 <= self.mutation_prob <= 1:
            raise ValueError(f"mutation_prob must be in [0, 1], got {self.mutation_prob}")
        if self.n_best > self.pop_size:
            raise ValueError(f"n_best ({self.n_best}) must be <= pop_size ({self.pop_size})")
        if self.elite_size > self.pop_size:
            raise ValueError(f"elite_size ({self.elite_size}) must be <= pop_size ({self.pop_size})")
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

    @classmethod
    def _base_kwargs(cls, d: dict) -> dict:
        """Extract and coerce the common fields from a raw param dict."""
        return dict(
            a=float(d['a']),
            b=float(d['b']),
            n_dims=int(d['n_dims']),
            epochs=int(d['epochs']),
            pop_size=int(d['pop_size']),
            n_best=int(d['n_best']),
            elite_size=int(d['elite_size']),
            k=int(d['k']),
            crossover_prob=float(d['crossover_prob']),
            mutation_prob=float(d['mutation_prob']),
            selection=d['selection'],
            crossover=d['crossover'],
            mutation=d['mutation'],
            function=d.get('function', 'rosenbrock'),
            maximize=bool(d.get('maximize', False)),
            early_stop=bool(d.get('early_stop', False)),
            patience=int(d.get('patience', 20)),
            min_delta=float(d.get('min_delta', 1e-6)),
            start_point=[float(x) for x in d['start_point']] if d.get('start_point') else None,
        )

    @classmethod
    def from_dict(cls, d: dict) -> GAConfig:
        return cls(**cls._base_kwargs(d))
