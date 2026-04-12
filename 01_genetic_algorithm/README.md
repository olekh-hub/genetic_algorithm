# Genetic algorithm (Rosenbrock)

Web app that runs a classic genetic algorithm on the **Rosenbrock** function: binary chromosomes, configurable bounds and dimensions, several selection / crossover / mutation operators, inversion, elitism, optional early stopping, and optional maximize mode.

The UI shows a **3D Rosenbrock surface** over your chosen bounds, **fitness charts** over epochs, elapsed time, and **auto-saves** each run as a timestamped CSV under `results/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser. Use **Download Last CSV** to re-download the file from the most recent run.
