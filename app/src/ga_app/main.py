import os

from flask import Flask, send_from_directory
from flask_cors import CORS

from ga_app.routes import api


def create_app():
    frontend_dist = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dist')
    frontend_dist = os.path.abspath(frontend_dist)

    app = Flask(__name__)
    CORS(app)

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    app.config['RESULTS_DIR'] = results_dir

    app.register_blueprint(api, url_prefix='/api')

    if os.path.isdir(frontend_dist):
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_frontend(path):
            file_path = os.path.join(frontend_dist, path)
            if path and os.path.isfile(file_path):
                return send_from_directory(frontend_dist, path)
            return send_from_directory(frontend_dist, 'index.html')

    return app


def main():
    app = create_app()
    app.run(debug=True, port=5050)


if __name__ == '__main__':
    main()
