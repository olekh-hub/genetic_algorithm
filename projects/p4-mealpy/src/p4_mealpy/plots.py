import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_output_dir():
    os.makedirs("output", exist_ok=True)

def plot_all_results(suite_results: dict):
    ensure_output_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(suite_results.items()):
        ax = axes[idx]
        
        if res["p4_pso"]["best_history"]:
            ax.plot(res["p4_pso"]["best_history"], label="P4: MealPy PSO", linewidth=2)
        if res["p1_binary"] and res["p1_binary"].get("best_history"):
            ax.plot(res["p1_binary"]["best_history"], label="P1: Binary GA", linestyle="--")
        if res["p2_real"] and res["p2_real"].get("best_history"):
            ax.plot(res["p2_real"]["best_history"], label="P2: Real GA", linestyle=":")
            
        ax.set_title(f"Konwergencja: {name}")
        ax.set_xlabel("Epoki")
        ax.set_ylabel("Fitness (Skala log)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("output/01_convergence_panel.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    scenarios = list(suite_results.keys())
    
    p4_times = [res["p4_pso"]["elapsed"] for res in suite_results.values()]
    p1_times = [res["p1_binary"]["elapsed"] if res["p1_binary"] else 0.0 for res in suite_results.values()]
    p2_times = [res["p2_real"]["elapsed"] if res["p2_real"] else 0.0 for res in suite_results.values()]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    plt.bar(x - width, p4_times, width, label="P4: MealPy PSO")
    plt.bar(x, p1_times, width, label="P1: Binary GA")
    plt.bar(x + width, p2_times, width, label="P2: Real GA")
    
    plt.xticks(x, scenarios, rotation=15)
    plt.ylabel("Czas wykonania (sekundy)")
    plt.title("Porównanie kosztu obliczeniowego algorytmów")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/02_execution_time_benchmark.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    p4_finals = [res["p4_pso"]["best_fitness"] for res in suite_results.values()]
    p2_finals = [res["p2_real"]["best_fitness"] if res["p2_real"] else 1e9 for res in suite_results.values()]
    
    plt.plot(scenarios, p4_finals, marker='o', markersize=8, linewidth=2, label="P4: MealPy PSO")
    plt.plot(scenarios, p2_finals, marker='s', markersize=8, linestyle="--", label="P2: Real GA")
    
    plt.yscale("log")
    plt.ylabel("Najlepsza (najmniejsza) wartość końcowa (log)")
    plt.title("Dokładność optymalizacji funkcji Rosenbrocka")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/03_final_accuracy_comparison.png")
    plt.close()
    
    print("\nPomyślnie wygenerowano 3 pliki analizy porównawczej w folderze 'output/':")
    print(" - output/01_convergence_panel.png")
    print(" - output/02_execution_time_benchmark.png")
    print(" - output/03_final_accuracy_comparison.png")