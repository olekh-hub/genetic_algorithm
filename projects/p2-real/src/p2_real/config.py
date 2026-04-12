from __future__ import annotations

from dataclasses import dataclass

from ga_core.config import GAConfig


@dataclass
class RealConfig(GAConfig):
    """Configuration for the real-valued chromosome GA."""

    sigma: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

    @classmethod
    def from_dict(cls, d: dict) -> RealConfig:
        return cls(
            **cls._base_kwargs(d),
            sigma=float(d.get('sigma', 0.1)),
        )
