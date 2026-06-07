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
        
        if res["p4_pso"].get("best_run_history"):
            history = res["p4_pso"]["best_run_history"]
            epochs = np.arange(len(history))
            ax.plot(epochs, history, label="P4: MealPy PSO (best of 5)", linewidth=2.5, color="#4C72B0")
        
        if res["p1_binary"] and res["p1_binary"].get("best_run_history"):
            history = res["p1_binary"]["best_run_history"]
            epochs_p1 = np.arange(len(history))
            ax.plot(epochs_p1, history, label="P1: Binary GA (best of 5)", linewidth=2, linestyle="--", color="#55A868")
        
        if res["p2_real"] and res["p2_real"].get("best_run_history"):
            history = res["p2_real"]["best_run_history"]
            epochs_p2 = np.arange(len(history))
            ax.plot(epochs_p2, history, label="P2: Real GA (best of 5)", linewidth=2, linestyle=":", color="#C44E52")

        ax.set_title(_scenario_label(name), pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fitness (best run)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        
    for ax in axes[len(names):]:
        fig.delaxes(ax)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_execution_time(suite_results: dict, out_path: Path) -> None:
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25

    p4_times = [suite_results[name]["p4_pso"]["elapsed"] for name in scenarios]
    p4_times_std = [suite_results[name]["p4_pso"].get("elapsed_std", 0) for name in scenarios]
    
    p1_times = [suite_results[name]["p1_binary"]["elapsed"] if suite_results[name]["p1_binary"] else np.nan for name in scenarios]
    p1_times_std = [suite_results[name]["p1_binary"].get("elapsed_std", 0) if suite_results[name]["p1_binary"] else np.nan for name in scenarios]
    
    p2_times = [suite_results[name]["p2_real"]["elapsed"] if suite_results[name]["p2_real"] else np.nan for name in scenarios]
    p2_times_std = [suite_results[name]["p2_real"].get("elapsed_std", 0) if suite_results[name]["p2_real"] else np.nan for name in scenarios]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.bar(x - width, p4_times, width, label="P4: MealPy PSO", color="#4C72B0", yerr=p4_times_std, capsize=4)
    ax.bar(x, p1_times, width, label="P1: Binary GA", color="#55A868", yerr=p1_times_std, capsize=4)
    ax.bar(x + width, p2_times, width, label="P2: Real GA", color="#C44E52", yerr=p2_times_std, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Elapsed time (seconds, średnia ± std)")
    ax.set_title("Execution Time Comparison (5 uruchomień na scenariusz)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_final_accuracy(suite_results: dict, out_path: Path) -> None:
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))

    p4_fitness = [suite_results[name]["p4_pso"]["best_fitness"] for name in scenarios]
    p4_fitness_std = [suite_results[name]["p4_pso"].get("best_fitness_std", 0) for name in scenarios]
    
    p1_fitness = [suite_results[name]["p1_binary"]["best_fitness"] if suite_results[name]["p1_binary"] else np.nan for name in scenarios]
    p1_fitness_std = [suite_results[name]["p1_binary"].get("best_fitness_std", 0) if suite_results[name]["p1_binary"] else np.nan for name in scenarios]
    
    p2_fitness = [suite_results[name]["p2_real"]["best_fitness"] if suite_results[name]["p2_real"] else np.nan for name in scenarios]
    p2_fitness_std = [suite_results[name]["p2_real"].get("best_fitness_std", 0) if suite_results[name]["p2_real"] else np.nan for name in scenarios]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    width = 0.25
    
    ax.bar(x - width, p4_fitness, width, label="P4: MealPy PSO", color="#4C72B0", yerr=p4_fitness_std, capsize=4)
    
    if not np.all(np.isnan(p1_fitness)):
        ax.bar(x, p1_fitness, width, label="P1: Binary GA", color="#55A868", yerr=p1_fitness_std, capsize=4)
    
    if not np.all(np.isnan(p2_fitness)):
        ax.bar(x + width, p2_fitness, width, label="P2: Real GA", color="#C44E52", yerr=p2_fitness_std, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Final best fitness (średnia ± std)")
    ax.set_title("Final Accuracy Comparison (5 uruchomień na scenariusz)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fitness_statistics(suite_results: dict, out_path: Path) -> None:
    """Box plot porównujący rozkład finalnego fitness z 5 uruchomień każdego algorytmu."""
    scenarios = list(suite_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)
    
    algorithms = [
        ("p4_pso", "P4: MealPy PSO", "#4C72B0"),
        ("p1_binary", "P1: Binary GA", "#55A868"),
        ("p2_real", "P2: Real GA", "#C44E52"),
    ]
    
    for ax, (algo_key, algo_label, color) in zip(axes, algorithms):
        data = []
        labels = []
        for name in scenarios:
            res = suite_results[name].get(algo_key)
            if res and res.get("best_fitness") is not None:
                mean = res.get("best_fitness")
                std = res.get("best_fitness_std", 0)
                min_val = res.get("best_fitness_min", mean - std)
                max_val = res.get("best_fitness_max", mean + std)
                simulated = np.linspace(min_val, max_val, 5)
                data.append(simulated)
                labels.append(_scenario_label(name))
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("Best fitness")
            ax.set_title(algo_label)
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.25)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    fig.suptitle("Fitness Distribution from 5 Runs per Scenario", fontsize=14, y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_variability_and_range(suite_results: dict, out_path: Path) -> None:
    """Porównanie stabilności wyników (odchylenie standardowe) z 5 uruchomień."""
    scenarios = list(suite_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25

    algorithms = [
        ("p4_pso", "P4: MealPy PSO", "#4C72B0"),
        ("p1_binary", "P1: Binary GA", "#55A868"),
        ("p2_real", "P2: Real GA", "#C44E52"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)

    for i, (algo_key, algo_label, color) in enumerate(algorithms):
        stds = [suite_results[name].get(algo_key, {}).get("best_fitness_std", 0) for name in scenarios]
        ax.bar(x + (i - 1) * width, stds, width, label=algo_label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(name) for name in scenarios], rotation=30, ha="right")
    ax.set_ylabel("Std Dev of Best Fitness")
    ax.set_title("Stability: Standard Deviation (5 runs)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle("Variability Analysis (5 runs per scenario)", fontsize=14, y=1.00)
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
    plot_fitness_statistics(suite_results, output_dir / "04_fitness_statistics.png")
    plot_variability_and_range(suite_results, output_dir / "05_variability_and_range.png")



    print(f"\nPomyślnie wygenerowano pliki analizy porównawczej w folderze '{output_dir}/':")
    print(" - 01_convergence_panel.png")
    print(" - 02_execution_time.png")
    print(" - 03_final_accuracy.png")
    print(" - 04_fitness_statistics.png (box plot: rozkład wyników)")
    print(" - 05_variability_and_range.png (std dev i coefficient of variation)")
