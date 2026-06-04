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
                pso_params=PSOConfig(w=0.729, c1=2.05, c2=2.05)
            ),
            # Scenariusz 2: Wysoka eksploracja (Większa bezwładność, mniejsza składowa społeczna)
            ExperimentSpec(
                name="S2_High_Exploration", n_dims=self.dims, epochs=self.epochs, pop_size=50,
                pso_params=PSOConfig(w=1.2, c1=2.5, c2=0.5)
            ),
            # Scenariusz 3: Wysoka eksploatacja (Szybka zbieżność do lidera roju)
            ExperimentSpec(
                name="S3_High_Exploitation", n_dims=self.dims, epochs=self.epochs, pop_size=50,
                pso_params=PSOConfig(w=0.4, c1=0.5, c2=2.8)
            ),
            # Scenariusz 4: Mała populacja (Wymagający test wydajnościowy)
            ExperimentSpec(
                name="S4_Small_Swarm", n_dims=self.dims, epochs=self.epochs, pop_size=15,
                pso_params=PSOConfig(w=0.729, c1=2.05, c2=2.05)
            )
        ]

    def run_all(self) -> dict:
        scenarios = self.prepare_scenarios()
        results = {}

        for spec in scenarios:
            print(f"Uruchamianie scenariusza: {spec.name}...")
            results[spec.name] = {
                "spec": spec,
                "p4_pso": run_mealpy_pso(spec),
                "p1_binary": run_baseline_p1(spec) if spec.run_p1 else None,
                "p2_real": run_baseline_p2(spec) if spec.run_p2 else None,
            }
        return results