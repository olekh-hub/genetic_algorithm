# Project 3 — PyGAD

Reimplements the genetic algorithm using [PyGAD](https://pygad.readthedocs.io/) and benchmarks it against the custom **binary** (P1) and **real** (P2) engines from this monorepo.

## Features

- **Binary** chromosomes via `gene_space=[0, 1]` and P1-compatible decoding
- **Real** chromosomes via bounded `gene_space`
- PyGAD built-in operators mapped to P1/P2 names (selection, crossover, mutation)
- **Custom crossover / mutation** (Projekt 3 `example_03` style)
- **Parallel fitness evaluation**: `thread` and `process` modes
- **Early stopping** via generation callback
- Auto-saved per-run CSVs + benchmark summary
- Comparison plots: encoding, selection, crossover, early stop, parallel speedup, convergence, heatmaps

## Setup

From the monorepo root:

```bash
uv sync --all-packages
```

Or only this package:

```bash
cd projects/p3-pyga
uv sync
```

## Run benchmark

```bash
cd projects/p3-pyga
uv run python main.py benchmark          # full grid (~minutes)
uv run python main.py benchmark --quick  # smoke test
```

Results land in `results/`; per-epoch histories in `output/histories/`.

## Generate plots

```bash
uv run python main.py plot
```

Charts are written to `output/` (PNG):

| File | Description |
|------|-------------|
| `01_encoding_overview.png` | Custom vs PyGAD — fitness & time |
| `02_selection_effect.png` | Selection method comparison |
| `03_crossover_effect.png` | Crossover by encoding |
| `04_early_stop.png` | Early stopping (full benchmark) |
| `05_parallel_speedup.png` | Thread vs process (PyGAD) |
| `06_population_scaling.png` | Population size sweep |
| `07_convergence_panel.png` | Best objective over epochs |
| `08_heatmap_crossover_mutation.png` | Crossover × mutation grid |

## Compare with P1 / P2

| | P1 (`p1-binary`) | P2 (`p2-real`) | P3 (`p3-pyga`) |
|---|---|---|---|
| Encoding | Binary bits | Float genes | PyGAD binary or real |
| Engine | Custom loop | Custom loop | `pygad.GA` |
| Parallel | — | — | thread / process |

Use the same Rosenbrock bounds, population size, and operator names where possible; see `src/p3_pyga/suite.py` for the experiment grid.
