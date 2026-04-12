from flask import Flask, render_template, request, jsonify, send_file
from test_function import (algorithm, save_results_csv, rosenbrock_surface,
                           CROSSOVER_FN, MUTATION_FN)
from datetime import datetime
import tempfile
import os

app = Flask(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

last_run = {'results': None, 'params': None, 'filename': None}


@app.route('/')
def index():
    return render_template('index.html',
                           crossover_methods=list(CROSSOVER_FN.keys()),
                           mutation_methods=list(MUTATION_FN.keys()))


@app.route('/surface', methods=['POST'])
def surface():
    data = request.json
    a = float(data.get('a', -2.048))
    b = float(data.get('b', 2.048))
    X, Y, Z = rosenbrock_surface(a, b)
    return jsonify({'x': X, 'y': Y, 'z': Z})


@app.route('/run', methods=['POST'])
def run():
    data = request.json
    params = {
        'a': float(data['a']),
        'b': float(data['b']),
        'n_dims': int(data['n_dims']),
        'n_bits': int(data['n_bits']),
        'epochs': int(data['epochs']),
        'pop_size': int(data['pop_size']),
        'n_best': int(data['n_best']),
        'elite_size': int(data['elite_size']),
        'k': int(data['k']),
        'crossover_prob': float(data['crossover_prob']),
        'mutation_prob': float(data['mutation_prob']),
        'inversion_prob': float(data['inversion_prob']),
        'selection': data['selection'],
        'crossover': data['crossover'],
        'mutation': data['mutation'],
        'maximize': data.get('maximize', False),
        'early_stop': data.get('early_stop', False),
        'patience': int(data.get('patience', 20)),
        'min_delta': float(data.get('min_delta', 1e-6)),
    }

    try:
        results = algorithm(params)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'run_{timestamp}.csv'
    filepath = os.path.join(RESULTS_DIR, filename)
    save_results_csv(results, params, filepath)

    last_run['results'] = results
    last_run['params'] = params
    last_run['filename'] = filename

    return jsonify({
        'best_fitness': results['best_fitness'],
        'best_variables': results['best_variables'],
        'best_history': results['best_history'],
        'mean_history': results['mean_history'],
        'std_history': results['std_history'],
        'elapsed': results['elapsed'],
        'total_epochs': results['total_epochs'],
        'stopped_early': results['stopped_early'],
        'saved_as': filename,
    })


@app.route('/download', methods=['GET'])
def download():
    if last_run['results'] is None:
        return jsonify({'error': 'no results yet'}), 400

    filepath = os.path.join(RESULTS_DIR, last_run['filename'])
    return send_file(filepath, as_attachment=True,
                     download_name=last_run['filename'], mimetype='text/csv')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
