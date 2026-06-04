import time
import numpy as np
from mealpy import FloatVar, PSO
from ga_core.objective import Rosenbrock

try:
    from p1_binary.algorithm import run_ga as run_p1_ga
    from p1_binary.config import BinaryGAConfig
except ImportError:
    run_p1_ga, BinaryGAConfig = None, None

try:
    from p2_real.algorithm import run_ga as run_p2_ga
    from p2_real.config import RealGAConfig
except ImportError:
    run_p2_ga, RealGAConfig = None, None

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
    if not run_p1_ga or not BinaryGAConfig:
        return {"best_fitness": None, "best_history": [], "elapsed": 0.0}
    
    cfg = BinaryGAConfig(
        objective="rosenbrock", 
        a=spec.a, 
        b=spec.b, 
        n_dims=spec.n_dims, 
        pop_size=spec.pop_size, 
        epochs=spec.epochs, 
        maximize=False
    )
    t0 = time.time()
    res = run_p1_ga(cfg)
    elapsed = time.time() - t0
    res["elapsed"] = elapsed
    return res

def run_baseline_p2(spec: ExperimentSpec) -> dict:
    if not run_p2_ga or not RealGAConfig:
        return {"best_fitness": None, "best_history": [], "elapsed": 0.0}
    
    cfg = RealGAConfig(
        objective="rosenbrock", 
        a=spec.a, 
        b=spec.b, 
        n_dims=spec.n_dims, 
        pop_size=spec.pop_size, 
        epochs=spec.epochs, 
        maximize=False
    )
    t0 = time.time()
    res = run_p2_ga(cfg)
    elapsed = time.time() - t0
    res["elapsed"] = elapsed
    return res