from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ga_core.objective import ObjectiveFunction


def binary_to_decimal(b_list: list, a: float, b: float) -> float:
    d, p = 0, 0
    for bit in reversed(b_list):
        if bit == 1:
            d += 2 ** p
        p += 1
    return a + d * ((b - a) / (2 ** len(b_list) - 1))


def decode_chromosome(chromosome: list, a: float, b: float, n_dims: int, n_bits: int) -> list:
    values = []
    for i in range(n_dims):
        segment = chromosome[i * n_bits:(i + 1) * n_bits]
        values.append(binary_to_decimal(segment, a, b))
    return values


def get_fitness(chromosome: list, a: float, b: float, n_dims: int, n_bits: int,
                objective: ObjectiveFunction) -> float:
    return objective(decode_chromosome(chromosome, a, b, n_dims, n_bits))


def decimal_to_binary(value: float, a: float, b: float, n_bits: int) -> list:
    max_int = 2 ** n_bits - 1
    scaled = round((value - a) / (b - a) * max_int)
    scaled = max(0, min(max_int, scaled))
    bits = []
    for _ in range(n_bits):
        bits.append(scaled & 1)
        scaled >>= 1
    return list(reversed(bits))


def encode_chromosome(values: list, a: float, b: float, n_bits: int) -> list:
    chromosome: list[int] = []
    for v in values:
        chromosome.extend(decimal_to_binary(v, a, b, n_bits))
    return chromosome
