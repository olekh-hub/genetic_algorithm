import numpy as np
import pandas as pd
import random
import time
import csv


# -- test function -------------------------------------------------------------

def rosenbrock(dims: list) -> float:
    f_x = 0
    for i in range(len(dims) - 1):
        f_x += 100 * (dims[i + 1] - dims[i] ** 2) ** 2 + (1 - dims[i]) ** 2
    return f_x


# -- population (real-valued) --------------------------------------------------

def initiate_population(n_dims: int, a: float, b: float, size: int) -> pd.DataFrame:
    data = [{'chromosome': np.random.uniform(a, b, size=n_dims).tolist()} for _ in range(size)]
    return pd.DataFrame(data)


def evaluate_population(population: pd.DataFrame) -> pd.DataFrame:
    population = population.copy()
    population['fitness'] = population['chromosome'].apply(rosenbrock)
    return population


# -- selection (same as P1) ----------------------------------------------------

def best_selection(population: pd.DataFrame, n_best: int, maximize: bool = False) -> pd.DataFrame:
    return population.sort_values(by='fitness', ascending=not maximize).head(n_best).reset_index(drop=True)


def roulette_selection(population: pd.DataFrame, n_best: int, maximize: bool = False) -> pd.DataFrame:
    pop = population.copy()
    if maximize:
        min_f = pop['fitness'].min()
        pop['prob'] = pop['fitness'] - min_f + 1e-10
    else:
        max_f = pop['fitness'].max()
        pop['prob'] = max_f - pop['fitness'] + 1e-10
    total = pop['prob'].sum()
    if total == 0:
        pop['prob'] = 1.0
        total = pop['prob'].sum()
    selected = np.random.choice(pop.index, size=n_best, p=pop['prob'] / total)
    return pop.loc[selected].drop(columns=['prob']).reset_index(drop=True)


def tournament_selection(population: pd.DataFrame, n_best: int, k: int, maximize: bool = False) -> pd.DataFrame:
    selected = []
    for _ in range(n_best):
        group = population.sample(n=min(k, len(population)), replace=False)
        winner = group.sort_values(by='fitness', ascending=not maximize).head(1)
        selected.append(winner)
    return pd.concat(selected, ignore_index=True)


# -- crossover (real-valued) ---------------------------------------------------

def arithmetic_crossover(p1: list, p2: list, a_bound: float, b_bound: float) -> list:
    k = np.random.random()
    return [np.clip(k * x + (1 - k) * y, a_bound, b_bound) for x, y in zip(p1, p2)]


def linear_crossover(p1: list, p2: list, a_bound: float, b_bound: float) -> list:
    # produces three children, returns the best (lowest rosenbrock)
    c1 = [0.5 * x + 0.5 * y for x, y in zip(p1, p2)]
    c2 = [1.5 * x - 0.5 * y for x, y in zip(p1, p2)]
    c3 = [-0.5 * x + 1.5 * y for x, y in zip(p1, p2)]
    candidates = []
    for c in [c1, c2, c3]:
        clipped = [np.clip(v, a_bound, b_bound) for v in c]
        candidates.append(clipped)
    return min(candidates, key=rosenbrock)


def blend_alpha_crossover(p1: list, p2: list, a_bound: float, b_bound: float, alpha: float = 0.5) -> list:
    child = []
    for x, y in zip(p1, p2):
        lo, hi = min(x, y), max(x, y)
        d = hi - lo
        child.append(np.clip(np.random.uniform(lo - alpha * d, hi + alpha * d), a_bound, b_bound))
    return child


def blend_alpha_beta_crossover(p1: list, p2: list, a_bound: float, b_bound: float,
                                alpha: float = 0.5, beta: float = 0.5) -> list:
    child = []
    for x, y in zip(p1, p2):
        lo, hi = min(x, y), max(x, y)
        d = hi - lo
        child.append(np.clip(np.random.uniform(lo - alpha * d, hi + beta * d), a_bound, b_bound))
    return child


def averaging_crossover(p1: list, p2: list, a_bound: float, b_bound: float) -> list:
    return [np.clip((x + y) / 2.0, a_bound, b_bound) for x, y in zip(p1, p2)]


CROSSOVER_FN = {
    'arithmetic': arithmetic_crossover,
    'linear': linear_crossover,
    'blend_alpha': blend_alpha_crossover,
    'blend_alpha_beta': blend_alpha_beta_crossover,
    'averaging': averaging_crossover,
}


# -- mutation (real-valued) ----------------------------------------------------

def uniform_mutation(chromosome: list, a_bound: float, b_bound: float, **kwargs) -> list:
    c = chromosome.copy()
    ix = np.random.randint(0, len(c))
    c[ix] = np.random.uniform(a_bound, b_bound)
    return c


def gaussian_mutation(chromosome: list, a_bound: float, b_bound: float, sigma: float = 0.1) -> list:
    c = chromosome.copy()
    ix = np.random.randint(0, len(c))
    c[ix] = np.clip(c[ix] + np.random.normal(0, sigma), a_bound, b_bound)
    return c


MUTATION_FN = {
    'uniform': uniform_mutation,
    'gaussian': gaussian_mutation,
}


# -- surface data for visualization --------------------------------------------

def rosenbrock_surface(a: float, b: float, resolution: int = 60):
    x = np.linspace(a, b, resolution)
    y = np.linspace(a, b, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = rosenbrock([X[i, j], Y[i, j]])
    return X.tolist(), Y.tolist(), Z.tolist()


# -- main loop -----------------------------------------------------------------

def algorithm(params: dict):
    a = params['a']
    b = params['b']
    n_dims = params['n_dims']
    epochs = params['epochs']
    pop_size = params['pop_size']
    n_best = params['n_best']
    elite_size = params['elite_size']
    crossover_prob = params['crossover_prob']
    mutation_prob = params['mutation_prob']
    k = params['k']
    selection = params['selection']
    crossover = params['crossover']
    mutation = params['mutation']
    maximize = params['maximize']
    sigma = params.get('sigma', 0.1)

    early_stop = params.get('early_stop', False)
    patience = params.get('patience', 20)
    min_delta = params.get('min_delta', 1e-6)

    cross_fn = CROSSOVER_FN[crossover]
    mut_fn = MUTATION_FN[mutation]

    current_pop = initiate_population(n_dims, a, b, pop_size)
    best_history = []
    mean_history = []
    std_history = []

    best_so_far = float('inf') if not maximize else float('-inf')
    no_improvement = 0
    stopped_early = False

    t_start = time.time()

    for ep in range(epochs):
        evaluated = evaluate_population(current_pop)

        if selection == 'best':
            selected = best_selection(evaluated, n_best, maximize)
        elif selection == 'roulette':
            selected = roulette_selection(evaluated, n_best, maximize)
        else:
            selected = tournament_selection(evaluated, n_best, k, maximize)

        children = []
        parents = selected['chromosome'].tolist()
        random.shuffle(parents)

        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            if np.random.random() < crossover_prob:
                children.append(cross_fn(p1, p2, a, b))
                children.append(cross_fn(p2, p1, a, b))
            else:
                children.append(p1.copy())
                children.append(p2.copy())

        for i in range(len(children)):
            if np.random.random() < mutation_prob:
                children[i] = mut_fn(children[i], a, b, sigma=sigma)

        new_pop = [{'chromosome': ch} for ch in children]
        while len(new_pop) < pop_size - elite_size:
            new_pop.append({'chromosome': random.choice(children).copy()})

        elite = best_selection(evaluated, elite_size, maximize)
        for _, row in elite.iterrows():
            new_pop.append({'chromosome': row['chromosome'].copy()})

        current_pop = pd.DataFrame(new_pop)

        best_val = evaluated['fitness'].min() if not maximize else evaluated['fitness'].max()
        best_history.append(best_val)
        mean_history.append(evaluated['fitness'].mean())
        std_history.append(evaluated['fitness'].std())

        if early_stop:
            improved = (best_val < best_so_far - min_delta) if not maximize \
                else (best_val > best_so_far + min_delta)
            if improved:
                best_so_far = best_val
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    stopped_early = True
                    break
        else:
            best_so_far = best_val

    elapsed = time.time() - t_start

    final_eval = evaluate_population(current_pop)
    if maximize:
        best_row = final_eval.loc[final_eval['fitness'].idxmax()]
    else:
        best_row = final_eval.loc[final_eval['fitness'].idxmin()]

    return {
        'best_fitness': float(best_row['fitness']),
        'best_variables': best_row['chromosome'],
        'best_history': best_history,
        'mean_history': mean_history,
        'std_history': std_history,
        'elapsed': elapsed,
        'total_epochs': len(best_history),
        'stopped_early': stopped_early,
    }


# -- save to csv ---------------------------------------------------------------

def save_results_csv(results: dict, params: dict, filepath: str):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(['parameter', 'value'])
        for key, val in params.items():
            w.writerow([key, val])
        w.writerow([])
        w.writerow(['best_fitness', results['best_fitness']])
        w.writerow(['best_variables', results['best_variables']])
        w.writerow(['elapsed_s', f"{results['elapsed']:.3f}"])
        w.writerow([])
        w.writerow(['epoch', 'best', 'mean', 'std'])
        for i, (b, m, s) in enumerate(zip(results['best_history'],
                                           results['mean_history'],
                                           results['std_history'])):
            w.writerow([i, b, m, s])
