from __future__ import annotations

from dataclasses import dataclass

from ga_core.config import GAConfig


@dataclass
class BinaryConfig(GAConfig):
    """Configuration for the binary-encoded GA."""

    n_bits: int = 20
    inversion_prob: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {self.n_bits}")
        if not 0 <= self.inversion_prob <= 1:
            raise ValueError(f"inversion_prob must be in [0, 1], got {self.inversion_prob}")

    @classmethod
    def from_dict(cls, d: dict) -> BinaryConfig:
        return cls(
            **cls._base_kwargs(d),
            n_bits=int(d.get('n_bits', 20)),
            inversion_prob=float(d.get('inversion_prob', 0)),
        )
