# Real Chromosome GA (Rosenbrock)

Extension of Project 1 that replaces binary chromosomes with **real-valued** ones. Each gene is a float within `[a, b]` — no binary encoding needed.

New crossover operators: **arithmetic**, **linear**, **blend-alpha**, **blend-alpha-beta**, **averaging**.
New mutation operators: **uniform**, **gaussian** (configurable sigma).

Everything else (selection, elitism, early stopping, 3D surface plot, auto-save) works the same as P1.

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

Open **http://127.0.0.1:5001** in a browser (port 5001 to avoid conflict with P1).
