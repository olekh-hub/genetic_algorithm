from __future__ import annotations

import time

import pygad

from p3_pyga.config import ExperimentSpec
from p3_pyga.custom_ops import custom_crossover, custom_mutation
from p3_pyga.encoding import decode_binary_solution
from p3_pyga.pygad_fitness import PyGadFitness, PyGadGenerationCallback, invert_fitness
from p3_pyga.pygad_ops import (
    arithmetic_crossover,
    averaging_crossover,
    blend_alpha_beta_crossover,
    blend_alpha_crossover,
)

SELECTION_MAP = {
    "best": "rank",
    "roulette": "rws",
    "tournament": "tournament",
}

BINARY_CROSSOVER_MAP = {
    "single_point": "single_point",
    "two_point": "two_points",
    "uniform": "uniform",
    "granular": "uniform",
}

REAL_CROSSOVER_BUILTIN = {
    "uniform": "uniform",
    "single_point": "single_point",
    "two_point": "two_points",
}

REAL_CROSSOVER_CUSTOM = {
    "arithmetic": arithmetic_crossover,
    "averaging": averaging_crossover,
    "blend_alpha": blend_alpha_crossover,
    "blend_alpha_beta": blend_alpha_beta_crossover,
    "linear": arithmetic_crossover,
}


def _binary_mutation_num_genes(mutation: str) -> int:
    if mutation == "two_point":
        return 2
    return 1


def run_pygad(spec: ExperimentSpec) -> dict:
    fitness_func = PyGadFitness(
        spec.representation,
        spec.function,
        spec.a,
        spec.b,
        spec.n_dims,
        spec.n_bits,
        spec.maximize,
    )
    gen_callback = PyGadGenerationCallback(
        spec.maximize, spec.early_stop, spec.patience, spec.min_delta,
    )

    if spec.representation == "binary":
        num_genes = spec.n_dims * spec.n_bits
        gene_space = [0, 1]
        gene_type = int
        crossover_type = BINARY_CROSSOVER_MAP.get(spec.crossover, "uniform")
        mutation_type = "random"
        mutation_num_genes = _binary_mutation_num_genes(spec.mutation)
        init_low, init_high = 0, 1
        mut_low, mut_high = 0, 1
    else:
        num_genes = spec.n_dims
        gene_space = {"low": spec.a, "high": spec.b}
        gene_type = float
        if spec.crossover in REAL_CROSSOVER_CUSTOM:
            crossover_type = REAL_CROSSOVER_CUSTOM[spec.crossover]
        else:
            crossover_type = REAL_CROSSOVER_BUILTIN.get(spec.crossover, "uniform")
        mutation_type = "random"
        mutation_num_genes = 1
        init_low, init_high = spec.a, spec.b
        mut_low, mut_high = spec.a, spec.b

    ga_kwargs: dict = {
        "num_generations": spec.epochs,
        "sol_per_pop": spec.pop_size,
        "num_parents_mating": min(spec.n_best, spec.pop_size),
        "num_genes": num_genes,
        "fitness_func": fitness_func,
        "gene_space": gene_space,
        "gene_type": gene_type,
        "parent_selection_type": SELECTION_MAP.get(spec.selection, "tournament"),
        "K_tournament": spec.k,
        "crossover_type": crossover_type,
        "crossover_probability": spec.crossover_prob,
        "mutation_type": mutation_type,
        "mutation_probability": spec.mutation_prob,
        "mutation_num_genes": mutation_num_genes,
        "keep_elitism": spec.elite_size,
        "init_range_low": init_low,
        "init_range_high": init_high,
        "random_mutation_min_val": mut_low,
        "random_mutation_max_val": mut_high,
        "on_generation": gen_callback,
    }

    if spec.use_custom_ops:
        ga_kwargs["crossover_type"] = custom_crossover
        ga_kwargs["mutation_type"] = custom_mutation
        ga_kwargs.pop("crossover_probability", None)

    if spec.parallel == "thread":
        ga_kwargs["parallel_processing"] = ["thread", spec.parallel_workers]
    elif spec.parallel == "process":
        ga_kwargs["parallel_processing"] = ["process", spec.parallel_workers]

    t0 = time.time()
    ga = pygad.GA(**ga_kwargs)
    ga.run()
    elapsed = time.time() - t0

    best_solution, best_fitness, _ = ga.best_solution()
    best_obj = invert_fitness(float(best_fitness), spec.maximize)

    if spec.representation == "binary":
        best_vars = decode_binary_solution(best_solution, spec.a, spec.b, spec.n_dims, spec.n_bits)
    else:
        best_vars = best_solution.tolist()

    best_history = [invert_fitness(float(f), spec.maximize) for f in ga.best_solutions_fitness]

    return {
        "best_fitness": best_obj,
        "best_variables": best_vars,
        "best_history": best_history,
        "mean_history": [],
        "std_history": [],
        "elapsed": elapsed,
        "total_epochs": len(best_history),
        "stopped_early": spec.early_stop and gen_callback.stall >= spec.patience,
    }
