from __future__ import annotations

from p1_binary.encoding import decode_chromosome as decode_binary


def decode_binary_solution(solution, a: float, b: float, n_dims: int, n_bits: int) -> list[float]:
    bits = [int(round(g)) for g in solution]
    return decode_binary(bits, a, b, n_dims, n_bits)
