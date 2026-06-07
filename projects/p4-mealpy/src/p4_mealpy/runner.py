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


def run_mealpy_pso(spec: ExperimentSpec, n_runs: int = 5) -> dict:
    objective = Rosenbrock()

    def fitness_function(solution):
        return objective(solution.tolist())

    problem_dict = {
        "bounds": FloatVar(lb=(spec.a,) * spec.n_dims, ub=(spec.b,) * spec.n_dims),
        "obj_func": fitness_function,
        "minmax": "min"
    }

    best_fitnesses = []
    elapsed_times = []
    histories = []

    for run in range(n_runs):
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
        best_fitnesses.append(g_best.target.fitness)
        elapsed_times.append(elapsed)
        histories.append(history)

    best_fitness_mean = np.mean(best_fitnesses)
    best_fitness_std = np.std(best_fitnesses)
    elapsed_mean = np.mean(elapsed_times)
    elapsed_std = np.std(elapsed_times)
    
    best_run_idx = np.argmin(best_fitnesses)
    best_run_history = histories[best_run_idx]
    
    max_len = max(len(h) for h in histories)
    history_mean = []
    history_std = []
    for i in range(max_len):
        values = [h[i] if i < len(h) else h[-1] for h in histories]
        history_mean.append(np.mean(values))
        history_std.append(np.std(values))

    return {
        "best_fitness": best_fitness_mean,
        "best_fitness_std": best_fitness_std,
        "best_fitness_min": np.min(best_fitnesses),
        "best_fitness_max": np.max(best_fitnesses),
        "best_variables": None,
        "best_history": history_mean,
        "best_history_std": history_std,
        "best_run_history": best_run_history,
        "elapsed": elapsed_mean,
        "elapsed_std": elapsed_std,
        "n_runs": n_runs,
    }

def run_baseline_p1(n_dims: int, epochs: int, pop_size: int = 50, n_runs: int = 5) -> dict:
    if not run_p1_ga:
        return {
            "best_fitness": None, "best_fitness_std": None, "best_fitness_min": None, "best_fitness_max": None,
            "best_history": [], "best_history_std": [], "elapsed": 0.0, "elapsed_std": 0.0, "n_runs": n_runs
        }

    params = {
        "a": -5.0,
        "b": 5.0,
        "n_dims": n_dims,
        "epochs": epochs,
        "pop_size": pop_size,
        "n_bits": 20,
        "inversion_prob": 0.0,
        **BASELINE_COMMON_PARAMS,
    }
    best_fitnesses = []
    elapsed_times = []
    histories = []

    for run in range(n_runs):
        t0 = time.time()
        res = run_p1_ga(params)
        elapsed = time.time() - t0
        
        best_fitnesses.append(res.get("best_fitness", np.nan))
        elapsed_times.append(elapsed)
        histories.append(res.get("best_history", []))

    best_fitness_mean = np.nanmean(best_fitnesses)
    best_fitness_std = np.nanstd(best_fitnesses)
    elapsed_mean = np.mean(elapsed_times)
    elapsed_std = np.std(elapsed_times)

    best_run_idx = np.nanargmin(best_fitnesses)
    best_run_history = histories[best_run_idx] if not np.isnan(best_fitnesses[best_run_idx]) else []

    max_len = max(len(h) for h in histories) if histories else 0
    history_mean = []
    history_std = []
    if max_len > 0:
        for i in range(max_len):
            values = [h[i] if i < len(h) else h[-1] for h in histories if h]
            if values:
                history_mean.append(np.mean(values))
                history_std.append(np.std(values))

    return {
        "best_fitness": best_fitness_mean,
        "best_fitness_std": best_fitness_std,
        "best_fitness_min": np.min(best_fitnesses),
        "best_fitness_max": np.max(best_fitnesses),
        "best_history": history_mean,
        "best_history_std": history_std,
        "best_run_history": best_run_history,
        "elapsed": elapsed_mean,
        "elapsed_std": elapsed_std,
        "n_runs": n_runs,
    }

def run_baseline_p2(n_dims: int, epochs: int, pop_size: int = 50, n_runs: int = 5) -> dict:
    if not run_p2_ga:
        return {
            "best_fitness": None, "best_fitness_std": None, "best_fitness_min": None, "best_fitness_max": None,
            "best_history": [], "best_history_std": [], "elapsed": 0.0, "elapsed_std": 0.0, "n_runs": n_runs
        }

    params = {
        "a": -5.0,
        "b": 5.0,
        "n_dims": n_dims,
        "epochs": epochs,
        "pop_size": pop_size,
        "n_best": 5,
        "elite_size": 2,
        "k": 2,
        "crossover_prob": 0.9,
        "mutation_prob": 0.1,
        "selection": "tournament",
        "crossover": "arithmetic",
        "mutation": "gaussian",
        "sigma": 0.1,
        "function": "rosenbrock",
        "maximize": False,
        "early_stop": False,
        "patience": 20,
        "min_delta": 1e-6,
        "start_point": None,
    }
    best_fitnesses = []
    elapsed_times = []
    histories = []

    for run in range(n_runs):
        t0 = time.time()
        res = run_p2_ga(params)
        elapsed = time.time() - t0
        
        best_fitnesses.append(res.get("best_fitness", np.nan))
        elapsed_times.append(elapsed)
        histories.append(res.get("best_history", []))

    best_fitness_mean = np.nanmean(best_fitnesses)
    best_fitness_std = np.nanstd(best_fitnesses)
    elapsed_mean = np.mean(elapsed_times)
    elapsed_std = np.std(elapsed_times)

    max_len = max(len(h) for h in histories) if histories else 0
    history_mean = []
    history_std = []
    if max_len > 0:
        for i in range(max_len):
            values = [h[i] if i < len(h) else h[-1] for h in histories if h]
            if values:
                history_mean.append(np.mean(values))
                history_std.append(np.std(values))

    best_run_idx = np.nanargmin(best_fitnesses)
    best_run_history = histories[best_run_idx]

    return {
        "best_fitness": best_fitness_mean,
        "best_fitness_std": best_fitness_std,
        "best_fitness_min": np.min(best_fitnesses),
        "best_fitness_max": np.max(best_fitnesses),
        "best_history": history_mean,
        "best_history_std": history_std,
        "best_run_history": best_run_history,
        "elapsed": elapsed_mean,
        "elapsed_std": elapsed_std,
        "n_runs": n_runs,
    }