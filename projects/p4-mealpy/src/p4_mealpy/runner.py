import time
import numpy as np
from mealpy import FloatVar, PSO
from ga_core.objective import Rosenbrock
from p4_mealpy.config import ExperimentSpec

try:
    from p1_binary.algorithm import algorithm as run_p1_ga
except ImportError:
    run_p1_ga = None

try:
    from p2_real.algorithm import algorithm as run_p2_ga
except ImportError:
    run_p2_ga = None

BASELINE_COMMON_PARAMS = {
    "n_best": 5,
    "elite_size": 2,
    "k": 2,
    "crossover_prob": 0.9,
    "mutation_prob": 0.1,
    "selection": "tournament",
    "crossover": "single_point",
    "mutation": "single_point",
    "function": "rosenbrock",
    "maximize": False,
    "early_stop": False,
    "patience": 20,
    "min_delta": 1e-6,
    "start_point": None,
}


def _make_common_params(spec: ExperimentSpec) -> dict:
    return {
        "a": spec.a,
        "b": spec.b,
        "n_dims": spec.n_dims,
        "epochs": spec.epochs,
        "pop_size": spec.pop_size,
        **BASELINE_COMMON_PARAMS,
    }


def _make_p1_params(spec: ExperimentSpec) -> dict:
    params = _make_common_params(spec)
    params.update({
        "n_bits": 20,
        "inversion_prob": 0.0,
    })
    return params


def _make_p2_params(spec: ExperimentSpec) -> dict:
    params = _make_common_params(spec)
    params.update({
        "crossover": "arithmetic",
        "mutation": "gaussian",
        "sigma": 0.1,
    })
    return params


def run_mealpy_pso(spec: ExperimentSpec) -> dict:
    objective = Rosenbrock()

    def fitness_function(solution):
        return objective(solution.tolist())

    problem_dict = {
        "bounds": FloatVar(lb=(spec.a,) * spec.n_dims, ub=(spec.b,) * spec.n_dims),
        "obj_func": fitness_function,
        "minmax": "min"
    }

    model = PSO.OriginalPSO(
        epoch=spec.epochs, 
        pop_size=spec.pop_size, 
        c1=spec.pso_params.c1, 
        c2=spec.pso_params.c2, 
        w=spec.pso_params.w
    )

    t0 = time.time()
    g_best = model.solve(problem_dict)
    elapsed = time.time() - t0

    history = [agent.target.fitness for agent in model.history.list_global_best]

    return {
        "best_fitness": g_best.target.fitness,
        "best_variables": g_best.solution.tolist(),
        "best_history": history,
        "elapsed": elapsed,
    }

def run_baseline_p1(spec: ExperimentSpec) -> dict:
    if not run_p1_ga:
        return {"best_fitness": None, "best_history": [], "elapsed": 0.0}

    params = _make_p1_params(spec)
    t0 = time.time()
    res = run_p1_ga(params)
    elapsed = time.time() - t0
    res["elapsed"] = elapsed
    return res

def run_baseline_p2(spec: ExperimentSpec) -> dict:
    if not run_p2_ga:
        return {"best_fitness": None, "best_history": [], "elapsed": 0.0}

    params = _make_p2_params(spec)
    t0 = time.time()
    res = run_p2_ga(params)
    elapsed = time.time() - t0
    res["elapsed"] = elapsed
    return res