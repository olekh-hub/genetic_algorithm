import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt



A = -2.048
B = 2.048
N_DIMS = 2
N_BITS = 50
EPOCHS = 200
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.1
INVERSION_PROB = 0.1
N_BEST = 20
POPULATION_SIZE = 100
ELITE_SIZE = 2
K = 3

def rosenbrock(n_dims: list) -> float:
    f_x = 0
    for i in range(0, len(n_dims) - 1):
        x_i = n_dims[i]
        x_i_next = n_dims[i+1]
        f_x += 100*(x_i_next - x_i**2)**2 + (1-x_i)**2
    return f_x

    
def binary_to_decimal(b_list: list, a: float, b: float) -> float:
    d, p = 0, 0 
    for bit in reversed(b_list):
        if bit == 1:
            d += 2 ** p
        p += 1

    L = len(b_list)

    x = a + d * ((b - a) / (2**L - 1))
    return x


def get_rosenbrock(chromosome: list, a: float, b: float, n_dims: int, n_bits: int) -> float:
    
    decoded_values = []

    for i in range(n_dims):
        start = i * n_bits
        end = start + n_bits
        segment = chromosome[start: end]

        real_value = binary_to_decimal(segment, a, b)
        decoded_values.append(real_value) 

    fitness = rosenbrock(decoded_values)

    return fitness


def initiate_population(n_dims: int, n_bits: int, size: int) -> pd.DataFrame:
    length = n_dims * n_bits
    data = []
    
    for _ in range(size):
        chromosome = np.random.randint(2, size=length).tolist()
        data.append({'chromosome': chromosome})
    
    return pd.DataFrame(data)
    
def evaluate_population(population: pd.DataFrame) -> pd.DataFrame:
    population['fitness'] = population['chromosome'].apply(lambda x: get_rosenbrock(x, A, B, N_DIMS, N_BITS))

    return population

def best_method(population: pd.DataFrame, n_best: int) -> pd.DataFrame:

    return population.sort_values(by='fitness').head(n_best)

def roulette_method(population: pd.DataFrame, n_best: int) -> pd.DataFrame:
    worst_fitness = population['fitness'].max()

    population['prob'] = (worst_fitness - population['fitness'])
    total_prob = population['prob'].sum()

    selected = np.random.choice(population.index, size=n_best, p=population['prob'] / total_prob)

    return population.loc[selected].reset_index(drop=True)

def tournament_method(population: pd.DataFrame, n_best: int, k: int) -> pd.DataFrame:

    selected = []

    for _ in range(n_best):
        tournament_chosen = population.sample(n=k, replace=False)
        best = tournament_chosen.sort_values(by='fitness').head(1)
        selected.append(best)

    selected_df = pd.concat(selected, ignore_index=True)

    return selected_df

def single_crossover(first: list, second: list) -> list:
    ix = np.random.randint(0, len(first))

    child = first[0: ix] + second[ix:]

    return child

def two_point_crossover(first: list, second: list) -> list:
    # TBF
    pass
    


def single_mutation(chromosome: list) -> list:
    ix = np.random.randint(0, len(chromosome))

    chromosome[ix] = abs(chromosome[ix] - 1)

    return chromosome

def double_mutation(chromosome: list) -> list:
    ix = np.random.choice(len(chromosome), size=2, replace=False)

    for i in ix:
        chromosome[i] = abs(chromosome[i] - 1)

    return chromosome

def edge_mutation(chromosome: list) -> list:
    ix = np.random.choice([0, len(chromosome) -1])

    chromosome[ix] = abs(chromosome[ix] - 1)

    return chromosome

def inversion(chromosome: list) -> list:
    first_ix = np.random.randint(0, len(chromosome) - 1)
    second_ix = np.random.randint(first_ix, len(chromosome) - 1)

    if first_ix == second_ix:
        return chromosome

    chromosome[first_ix: second_ix] = chromosome[first_ix: second_ix][::-1]

    return chromosome


def plot_results(fitness_history: pd.DataFrame):
    plt.figure(figsize=(14,7))
    plt.plot(fitness_history)
    plt.xlabel('EPOCH')
    plt.ylabel('fitness value')
    plt.title('fitness value by epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def algorithm(selection_method: str, mutation_method: str):
    current_population_df = initiate_population(N_DIMS, N_BITS, size=POPULATION_SIZE)
    fitness_history = []
    for ep in range(EPOCHS):
        evaluated_population_df = evaluate_population(current_population_df)
        if selection_method == 'best':
            selected_chromosomes = best_method(evaluated_population_df, n_best=N_BEST)
        elif selection_method == 'roulette': 
            selected_chromosomes = roulette_method(evaluated_population_df, n_best=N_BEST)
        elif selection_method == 'tournament':
            selected_chromosomes = tournament_method(evaluated_population_df, n_best=N_BEST, k=K)
        else:
            return f'wtf man, pass a valid method name'
        children = []
        for i in range(0, N_BEST -1 , 2):
            parent1 = selected_chromosomes.iloc[i]['chromosome']
            parent2 = selected_chromosomes.iloc[i+1]['chromosome']

            if np.random.random() < CROSSOVER_PROB:
                children.append(single_crossover(parent1, parent2))
                children.append(single_crossover(parent2, parent1))
            else:
                children.append(parent1.copy())
                children.append(parent2.copy())

        for i in range(len(children)):
            if np.random.random() < MUTATION_PROB:
                if mutation_method == 'single':
                    children[i] = single_mutation(children[i])
                elif mutation_method == 'double':
                    children[i] = double_mutation(children[i])
                elif mutation_method == 'edge':
                    children[i] = edge_mutation(children[i])
            
            if np.random.random() < INVERSION_PROB:
                children[i] = inversion(children[i])

        new_population = []
        for ch in children:
            new_population.append({'chromosome': ch})

        while len(new_population) < POPULATION_SIZE - ELITE_SIZE:
            new_population.append({'chromosome': random.choice(children)})

        elite_df = best_method(evaluated_population_df, n_best=ELITE_SIZE)
        for _, row in elite_df.iterrows():
            new_population.append({'chromosome': row['chromosome']})

        current_population_df = pd.DataFrame(new_population)

        best_in_epoch = evaluated_population_df['fitness'].min()
        fitness_history.append(best_in_epoch)
        
        print(f'Epoch: {ep}: best fitness: {best_in_epoch}')

    plot_results(fitness_history)
    return evaluate_population(current_population_df)

def main():
    algorithm('best', 'double')

if __name__ == '__main__':
    main()