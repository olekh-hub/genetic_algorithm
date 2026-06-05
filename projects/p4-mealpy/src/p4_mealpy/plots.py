from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _style():
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rc("font", size=11)
    plt.rc("figure", autolayout=True)


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _scenario_label(name: str) -> str:
    return name.replace("_", " ").title()


def _plot_algorithm_line(ax, x, y, label, color, marker, linestyle):
    ax.plot(x, y, marker=marker, color=color, linestyle=linestyle, linewidth=2, markersize=7, label=label)


def _epoch_to_threshold(history: list[float], threshold: float) -> float:
    if not history:
        return np.nan
    for index, value in enumerate(history, start=1):
        if value <= threshold:
            return float(index)
    return np.nan


def _start_to_end_improvement(history: list[float]) -> float:
    if not history:
        return np.nan
    start = history[0]
    end = history[-1]
    if start == 0:
        return np.nan
    return 100.0 * (start - end) / abs(start)


def plot_convergence_panel(suite_results: dict, out_path: Path) -> None:
    names = list(suite_results.keys())
    rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows), dpi=140)
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        res = suite_results[name]
        if res["p4_pso"].get("best_history"):
            ax.plot(res["p4_pso"]["best_history"], label="P4: MealPy PSO", linewidth=2)
        if res["p1_binary"] and res["p1_binary"].get("best_history"):
            ax.plot(res["p1_binary"]["best_history"], label="P1: Binary GA", linestyle="--")
        if res["p2_real"] and res["p2_real"].get("best_history"):
            ax.plot(res["p2_real"]["best_history"], label="P2: Real GA", linestyle=":")

        ax.set_title(_scenario_label(name), pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fitness")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[len(names):]:
        fig.delaxes(ax)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_execution_time(suite_results: dict, out_path: Path) -> None:
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25

    p4_times = [suite_results[name]["p4_pso"]["elapsed"] for name in scenarios]
    p1_times = [suite_results[name]["p1_binary"]["elapsed"] if suite_results[name]["p1_binary"] else np.nan for name in scenarios]
    p2_times = [suite_results[name]["p2_real"]["elapsed"] if suite_results[name]["p2_real"] else np.nan for name in scenarios]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.bar(x - width, p4_times, width, label="P4: MealPy PSO", color="#4C72B0")
    ax.bar(x, p1_times, width, label="P1: Binary GA", color="#55A868")
    ax.bar(x + width, p2_times, width, label="P2: Real GA", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_title("Execution Time Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_final_accuracy(suite_results: dict, out_path: Path) -> None:
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))

    p4_fitness = [suite_results[name]["p4_pso"]["best_fitness"] for name in scenarios]
    p1_fitness = [suite_results[name]["p1_binary"]["best_fitness"] if suite_results[name]["p1_binary"] else np.nan for name in scenarios]
    p2_fitness = [suite_results[name]["p2_real"]["best_fitness"] if suite_results[name]["p2_real"] else np.nan for name in scenarios]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    _plot_algorithm_line(ax, x, p4_fitness, "P4: MealPy PSO", "#4C72B0", "o", "-")
    if not np.all(np.isnan(p1_fitness)):
        _plot_algorithm_line(ax, x, p1_fitness, "P1: Binary GA", "#55A868", "s", "--")
    if not np.all(np.isnan(p2_fitness)):
        _plot_algorithm_line(ax, x, p2_fitness, "P2: Real GA", "#C44E52", "^", ":")

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Final best fitness")
    ax.set_title("Final Accuracy Comparison")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_speed(suite_results: dict, out_path: Path, threshold: float = 1e-2) -> None:
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25

    p4_speed = [
        _epoch_to_threshold(suite_results[name]["p4_pso"].get("best_history", []), threshold)
        for name in scenarios
    ]
    p1_speed = [
        _epoch_to_threshold(suite_results[name]["p1_binary"].get("best_history", []), threshold)
        if suite_results[name]["p1_binary"] else np.nan
        for name in scenarios
    ]
    p2_speed = [
        _epoch_to_threshold(suite_results[name]["p2_real"].get("best_history", []), threshold)
        if suite_results[name]["p2_real"] else np.nan
        for name in scenarios
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.bar(x - width, p4_speed, width, label=f"P4: MealPy PSO <= {threshold}", color="#4C72B0")
    ax.bar(x, p1_speed, width, label=f"P1: Binary GA <= {threshold}", color="#55A868")
    ax.bar(x + width, p2_speed, width, label=f"P2: Real GA <= {threshold}", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Epochs to threshold")
    ax.set_title(f"Epochs to reach fitness <= {threshold}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_from_start(suite_results: dict, out_path: Path) -> None:
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25

    p4_improvement = [
        _start_to_end_improvement(suite_results[name]["p4_pso"].get("best_history", []))
        for name in scenarios
    ]
    p1_improvement = [
        _start_to_end_improvement(suite_results[name]["p1_binary"].get("best_history", []))
        if suite_results[name]["p1_binary"] else np.nan
        for name in scenarios
    ]
    p2_improvement = [
        _start_to_end_improvement(suite_results[name]["p2_real"].get("best_history", []))
        if suite_results[name]["p2_real"] else np.nan
        for name in scenarios
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.bar(x - width, p4_improvement, width, label="P4: MealPy PSO", color="#4C72B0")
    ax.bar(x, p1_improvement, width, label="P1: Binary GA", color="#55A868")
    ax.bar(x + width, p2_improvement, width, label="P2: Real GA", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Improvement from start (%)")
    ax.set_title("Fitness improvement from first to last epoch")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_best_vs_time(suite_results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    for algorithm, label, color in [
        ("p4_pso", "P4: MealPy PSO", "#4C72B0"),
        ("p1_binary", "P1: Binary GA", "#55A868"),
        ("p2_real", "P2: Real GA", "#C44E52"),
    ]:
        x = []
        y = []
        for res in suite_results.values():
            entry = res.get(algorithm)
            if entry and entry.get("best_fitness") is not None:
                x.append(entry.get("elapsed", np.nan))
                y.append(entry["best_fitness"])
        if x:
            ax.scatter(x, y, label=label, color=color, s=90, alpha=0.85, edgecolors="w", linewidth=0.8)

    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Final best fitness")
    ax.set_yscale("log")
    ax.set_title("Time vs Final Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_relative_improvement(suite_results: dict, out_path: Path) -> None:
    names = []
    p4_vs_p1 = []
    p4_vs_p2 = []

    for name, res in suite_results.items():
        if res["p1_binary"] and res["p4_pso"]["best_fitness"] is not None:
            baseline = res["p1_binary"]["best_fitness"]
            if baseline != 0:
                names.append(name)
                p4_vs_p1.append(100.0 * (baseline - res["p4_pso"]["best_fitness"]) / abs(baseline))
            else:
                names.append(name)
                p4_vs_p1.append(0.0)
        if res["p2_real"] and res["p4_pso"]["best_fitness"] is not None:
            baseline = res["p2_real"]["best_fitness"]
            if baseline != 0:
                p4_vs_p2.append(100.0 * (baseline - res["p4_pso"]["best_fitness"]) / abs(baseline))
            else:
                p4_vs_p2.append(0.0)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    idx = np.arange(len(names))
    if p4_vs_p1:
        ax.bar(idx - 0.15, p4_vs_p1, width=0.3, label="P4 vs P1", color="#4C72B0")
    if p4_vs_p2:
        ax.bar(idx + 0.15, p4_vs_p2[: len(idx)], width=0.3, label="P4 vs P2", color="#C44E52")

    ax.set_xticks(idx)
    ax.set_xticklabels([_scenario_label(name) for name in names], rotation=30, ha="right")
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title("P4 Improvement over Baselines")
    ax.axhline(0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_all_results(suite_results: dict):
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output"
    _style()
    _ensure_output_dir(output_dir)

    plot_convergence_panel(suite_results, output_dir / "01_convergence_panel.png")
    plot_execution_time(suite_results, output_dir / "02_execution_time.png")
    plot_final_accuracy(suite_results, output_dir / "03_final_accuracy.png")
    plot_best_vs_time(suite_results, output_dir / "04_best_vs_time.png")
    plot_relative_improvement(suite_results, output_dir / "05_relative_improvement.png")
    plot_threshold_speed(suite_results, output_dir / "06_threshold_speed.png")
    plot_improvement_from_start(suite_results, output_dir / "07_improvement_from_start.png")

    print(f"\nPomyślnie wygenerowano pliki analizy porównawczej w folderze '{output_dir}/':")
    print(" - 01_convergence_panel.png")
    print(" - 02_execution_time.png")
    print(" - 03_final_accuracy.png")
    print(" - 04_best_vs_time.png")
    print(" - 05_relative_improvement.png")
    print(" - 06_threshold_speed.png")
    print(" - 07_improvement_from_start.png")
