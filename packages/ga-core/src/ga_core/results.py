import csv


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
