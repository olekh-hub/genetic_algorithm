from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path

import pandas as pd

from ga_core.results import save_results_csv
from p1_binary import algorithm as binary_algorithm
from p2_real import algorithm as real_algorithm
from p3_pyga.config import ExperimentSpec
from p3_pyga.pygad_runner import run_pygad


def _slug(*parts: str) -> str:
    return "_".join(p.replace(" ", "-") for p in parts if p)


def iter_experiments(quick: bool = False):
    epochs = 40 if quick else 120
    pop_sizes = [100] if quick else [100, 200]
    selections = ["best", "tournament"] if quick else ["best", "roulette", "tournament"]
    early_flags = [False] if quick else [False, True]

    if quick:
        binary_ops = [
            ("single_point", "single_point"),
            ("uniform", "single_point"),
        ]
        real_ops = [
            ("arithmetic", "gaussian"),
            ("blend_alpha", "uniform"),
        ]
    else:
        binary_ops = [
            ("single_point", "single_point"),
            ("uniform", "single_point"),
            ("two_point", "two_point"),
        ]
        real_ops = [
            ("arithmetic", "gaussian"),
            ("blend_alpha", "uniform"),
            ("averaging", "gaussian"),
            ("blend_alpha_beta", "uniform"),
        ]

    counter = 0

    for rep, ops in [("binary", binary_ops), ("real", real_ops)]:
        for selection in selections:
            for crossover, mutation in ops:
                for pop_size in pop_sizes:
                    for early_stop in early_flags:
                        base = dict(
                            representation=rep,
                            epochs=epochs,
                            pop_size=pop_size,
                            selection=selection,
                            crossover=crossover,
                            mutation=mutation,
                            early_stop=early_stop,
                            patience=15 if quick else 20,
                        )

                        cid = _slug(rep, selection, crossover, mutation, f"p{pop_size}", f"es{int(early_stop)}")
                        counter += 1

                        yield ExperimentSpec(
                            experiment_id=f"{cid}_custom",
                            engine="custom",
                            parallel="none",
                            **base,
                        )
                        yield ExperimentSpec(
                            experiment_id=f"{cid}_pygad",
                            engine="pygad",
                            parallel="none",
                            **base,
                        )

    if not quick:
        for parallel in ("thread", "process"):
            yield ExperimentSpec(
                experiment_id=_slug("real", "tournament", "uniform", f"pygad_{parallel}"),
                representation="real",
                engine="pygad",
                parallel=parallel,
                selection="tournament",
                crossover="blend_alpha",
                mutation="uniform",
                epochs=epochs,
                pop_size=100,
            )
            yield ExperimentSpec(
                experiment_id=_slug("binary", "tournament", "uniform", f"pygad_{parallel}"),
                representation="binary",
                engine="pygad",
                parallel=parallel,
                selection="tournament",
                crossover="uniform",
                mutation="single_point",
                epochs=epochs,
                pop_size=100,
            )

        yield ExperimentSpec(
            experiment_id="real_pygad_custom_ops",
            representation="real",
            engine="pygad",
            parallel="thread",
            use_custom_ops=True,
            selection="tournament",
            crossover="blend_alpha",
            mutation="uniform",
            epochs=epochs,
            pop_size=100,
        )
        yield ExperimentSpec(
            experiment_id="binary_pygad_custom_ops",
            representation="binary",
            engine="pygad",
            parallel="thread",
            use_custom_ops=True,
            selection="tournament",
            crossover="single_point",
            mutation="single_point",
            epochs=epochs,
            pop_size=100,
        )


def run_single(spec: ExperimentSpec) -> dict:
    params = spec.to_params()

    if spec.engine == "custom":
        if spec.representation == "binary":
            results = binary_algorithm(params)
        else:
            results = real_algorithm(params)
    else:
        results = run_pygad(spec)

    row = {**params, **results}
    row["experiment_id"] = spec.experiment_id
    row["engine"] = spec.engine
    row["parallel"] = spec.parallel
    row["use_custom_ops"] = spec.use_custom_ops
    for heavy in ("best_positions_history", "population_history"):
        row.pop(heavy, None)
    return row


def run_benchmark(output_dir: Path | None = None, quick: bool = False) -> pd.DataFrame:
    root = output_dir or Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    histories: dict[str, list[float]] = {}

    specs = list(iter_experiments(quick=quick))
    print(f"Running {len(specs)} experiments ({'quick' if quick else 'full'} mode)...")

    for i, spec in enumerate(specs, 1):
        print(f"  [{i}/{len(specs)}] {spec.experiment_id}")
        try:
            row = run_single(spec)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            row = {
                "experiment_id": spec.experiment_id,
                "error": str(exc),
                **spec.to_params(),
                "engine": spec.engine,
                "parallel": spec.parallel,
                "use_custom_ops": spec.use_custom_ops,
            }
            rows.append(row)
            continue

        histories[spec.experiment_id] = row.get("best_history", [])
        run_path = results_dir / f"{spec.experiment_id}.csv"
        save_results_csv(
            {k: row[k] for k in ("best_fitness", "best_variables", "best_history", "mean_history", "std_history", "elapsed", "total_epochs", "stopped_early") if k in row},
            spec.to_params(),
            str(run_path),
        )
        row["results_path"] = str(run_path)
        row.pop("best_history", None)
        row.pop("mean_history", None)
        row.pop("std_history", None)
        row.pop("best_variables", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"benchmark_{stamp}.csv"
    df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    hist_dir = root / "output" / "histories"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for eid, hist in histories.items():
        pd.DataFrame({"epoch": range(len(hist)), "best": hist}).to_csv(
            hist_dir / f"{eid}.csv", index=False,
        )

    return df
