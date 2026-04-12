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


# -- encoding / decoding ------------------------------------------------------

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


def get_fitness(chromosome: list, a: float, b: float, n_dims: int, n_bits: int) -> float:
    return rosenbrock(decode_chromosome(chromosome, a, b, n_dims, n_bits))


# -- population ----------------------------------------------------------------

def initiate_population(n_dims: int, n_bits: int, size: int) -> pd.DataFrame:
    length = n_dims * n_bits
    data = [{'chromosome': np.random.randint(2, size=length).tolist()} for _ in range(size)]
    return pd.DataFrame(data)


def evaluate_population(population: pd.DataFrame, a, b, n_dims, n_bits) -> pd.DataFrame:
    population = population.copy()
    population['fitness'] = population['chromosome'].apply(
        lambda x: get_fitness(x, a, b, n_dims, n_bits)
    )
    return population


# -- selection -----------------------------------------------------------------

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


# -- crossover ----------------------------------------------------------------

def single_point_crossover(first: list, second: list) -> list:
    ix = np.random.randint(1, len(first))
    return first[:ix] + second[ix:]


def two_point_crossover(first: list, second: list) -> list:
    pts = sorted(np.random.choice(range(1, len(first)), size=2, replace=False))
    return first[:pts[0]] + second[pts[0]:pts[1]] + first[pts[1]:]


def uniform_crossover(first: list, second: list) -> list:
    return [f if np.random.random() < 0.5 else s for f, s in zip(first, second)]


def granular_crossover(first: list, second: list, grain: int = 4) -> list:
    child = []
    use_first = True
    for i in range(0, len(first), grain):
        chunk = first[i:i + grain] if use_first else second[i:i + grain]
        child.extend(chunk)
        use_first = not use_first
    return child


CROSSOVER_FN = {
    'single_point': single_point_crossover,
    'two_point': two_point_crossover,
    'uniform': uniform_crossover,
    'granular': granular_crossover,
}


# -- mutation ------------------------------------------------------------------

def single_mutation(chromosome: list) -> list:
    c = chromosome.copy()
    ix = np.random.randint(0, len(c))
    c[ix] ^= 1
    return c


def double_mutation(chromosome: list) -> list:
    c = chromosome.copy()
    ixs = np.random.choice(len(c), size=2, replace=False)
    for i in ixs:
        c[i] ^= 1
    return c


def edge_mutation(chromosome: list) -> list:
    c = chromosome.copy()
    ix = np.random.choice([0, len(c) - 1])
    c[ix] ^= 1
    return c


MUTATION_FN = {
    'single_point': single_mutation,
    'two_point': double_mutation,
    'edge': edge_mutation,
}


# -- inversion -----------------------------------------------------------------

def inversion(chromosome: list) -> list:
    c = chromosome.copy()
    a, b = sorted(np.random.choice(len(c), size=2, replace=False))
    c[a:b + 1] = c[a:b + 1][::-1]
    return c


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
    n_bits = params['n_bits']
    epochs = params['epochs']
    pop_size = params['pop_size']
    n_best = params['n_best']
    elite_size = params['elite_size']
    crossover_prob = params['crossover_prob']
    mutation_prob = params['mutation_prob']
    inversion_prob = params['inversion_prob']
    k = params['k']
    selection = params['selection']
    crossover = params['crossover']
    mutation = params['mutation']
    maximize = params['maximize']

    early_stop = params.get('early_stop', False)
    patience = params.get('patience', 20)
    min_delta = params.get('min_delta', 1e-6)

    cross_fn = CROSSOVER_FN[crossover]
    mut_fn = MUTATION_FN[mutation]

    current_pop = initiate_population(n_dims, n_bits, pop_size)
    best_history = []
    mean_history = []
    std_history = []

    best_so_far = float('inf') if not maximize else float('-inf')
    no_improvement = 0
    stopped_early = False

    t_start = time.time()

    for ep in range(epochs):
        evaluated = evaluate_population(current_pop, a, b, n_dims, n_bits)

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
                children.append(cross_fn(p1, p2))
                children.append(cross_fn(p2, p1))
            else:
                children.append(p1.copy())
                children.append(p2.copy())

        for i in range(len(children)):
            if np.random.random() < mutation_prob:
                children[i] = mut_fn(children[i])
            if np.random.random() < inversion_prob:
                children[i] = inversion(children[i])

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

        # early stopping check
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

    final_eval = evaluate_population(current_pop, a, b, n_dims, n_bits)
    if maximize:
        best_row = final_eval.loc[final_eval['fitness'].idxmax()]
    else:
        best_row = final_eval.loc[final_eval['fitness'].idxmin()]

    best_decoded = decode_chromosome(best_row['chromosome'], a, b, n_dims, n_bits)

    return {
        'best_fitness': float(best_row['fitness']),
        'best_variables': best_decoded,
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
