from p4_mealpy.config import ExperimentSpec, PSOConfig
from p4_mealpy.runner import run_mealpy_pso, run_baseline_p1, run_baseline_p2

class ExperimentSuite:
    def __init__(self, epochs: int = 100, dims: int = 2):
        self.epochs = epochs
        self.dims = dims

    def prepare_scenarios(self) -> list[ExperimentSpec]:
        return [
            # Scenariusz 1: Standardowe parametry (Bazowy)
            ExperimentSpec(
                name="S1_Standard_PSO", n_dims=self.dims, epochs=self.epochs, pop_size=50,
                pso_params=PSOConfig(w=0.75, c1=2.05, c2=2.05)
            ),
            # Scenariusz 2: Wysoka eksploracja - zachowanie indywidualne
            ExperimentSpec(
                name="S2_High_Exploration", n_dims=self.dims, epochs=self.epochs, pop_size=50,
                pso_params=PSOConfig(w=0.9, c1=2.5, c2=0.5)
            ),
            # Scenariusz 3: Wysoka eksploatacja - zachowanie stadne
            ExperimentSpec(
                name="S3_High_Exploitation", n_dims=self.dims, epochs=self.epochs, pop_size=50,
                pso_params=PSOConfig(w=0.4, c1=0.5, c2=2.8)
            )
        ]

    def run_all(self) -> dict:
        scenarios = self.prepare_scenarios()
        results = {}
        
        print("Uruchamianie P1 Binary GA (5 razy)...")
        p1_results = run_baseline_p1(n_dims=self.dims, epochs=self.epochs, pop_size=50, n_runs=5)
        
        print("Uruchamianie P2 Real GA (5 razy)...")
        p2_results = run_baseline_p2(n_dims=self.dims, epochs=self.epochs, pop_size=50, n_runs=5)

        for spec in scenarios:
            print(f"Uruchamianie scenariusza: {spec.name}...")
            results[spec.name] = {
                "spec": spec,
                "p4_pso": run_mealpy_pso(spec),
                "p1_binary": p1_results,  
                "p2_real": p2_results,    
            }
        return results