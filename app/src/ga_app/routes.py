from datetime import datetime
import os

from flask import Blueprint, request, jsonify, send_file, current_app

from ga_core.objective import function_registry
from ga_core.results import save_results_csv
from ga_app.variants import VARIANTS

api = Blueprint('api', __name__)

last_run = {'results': None, 'params': None, 'filename': None}


@api.route('/functions')
def functions():
    return jsonify({'functions': function_registry.names})


@api.route('/methods')
def methods():
    result = {}
    for name, v in VARIANTS.items():
        result[f'{name}_crossover'] = v['crossover_ops'].names
        result[f'{name}_mutation'] = v['mutation_ops'].names
    return jsonify(result)


@api.route('/surface', methods=['POST'])
def surface():
    data = request.json
    fn_name = data.get('function', 'rosenbrock')
    objective = function_registry[fn_name]()
    X, Y, Z = objective.surface(
        float(data.get('x1_min', -2.048)),
        float(data.get('x1_max', 2.048)),
        float(data.get('x2_min', -2.048)),
        float(data.get('x2_max', 2.048)),
    )
    return jsonify({'x': X, 'y': Y, 'z': Z})


@api.route('/run', methods=['POST'])
def run():
    data = request.json
    variant_name = data.get('variant', 'binary')

    if variant_name not in VARIANTS:
        return jsonify({'error': f"Unknown variant '{variant_name}'. Available: {list(VARIANTS.keys())}"}), 400

    variant = VARIANTS[variant_name]

    try:
        results = variant['algorithm'](data)
    except (ValueError, KeyError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Algorithm failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'run_{variant_name}_{timestamp}.csv'
    filepath = os.path.join(current_app.config['RESULTS_DIR'], filename)
    save_results_csv(results, data, filepath)

    last_run['results'] = results
    last_run['params'] = data
    last_run['filename'] = filename

    return jsonify({
        **results,
        'saved_as': filename,
    })


@api.route('/download')
def download():
    if last_run['results'] is None:
        return jsonify({'error': 'no results yet'}), 400

    filepath = os.path.join(current_app.config['RESULTS_DIR'], last_run['filename'])
    return send_file(filepath, as_attachment=True,
                     download_name=last_run['filename'], mimetype='text/csv')
