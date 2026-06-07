import argparse
from p4_mealpy.suite import ExperimentSuite
from p4_mealpy.plots import plot_all_results

def main():
    parser = argparse.ArgumentParser(description="P4 Moduł testowy MealPy PSO vs P1 vs P2")
    parser.add_argument("--epochs", type=int, default=120, help="Maksymalna liczba epok")
    parser.add_argument("--dims", type=int, default=2, help="Wymiarowość funkcji celu")
    args = parser.parse_ok() if hasattr(parser, 'parse_ok') else parser.parse_args()

    print(f"====================================================")
    print(f"  URUCHAMIANIE SUITE EKSPERYMENTALNEGO: PROJEKT 4   ")
    print(f"  Optymalizacja funkcji Rosenbrocka ({args.dims}D)    ")
    print(f"====================================================\n")

    suite = ExperimentSuite(epochs=args.epochs, dims=args.dims)
    suite_results = suite.run_all()

    print("\nPrzetwarzanie wyników i generowanie zaawansowanych wykresów...")
    plot_all_results(suite_results)
    print("\nEksperyment ukończony pomyślnie.")

if __name__ == "__main__":
    main()