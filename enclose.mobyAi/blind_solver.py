"""
Blind Solver

Attempts to find the optimum using statistical saturation method
without knowing the target optimal area beforehand.
"""

from typing import Dict, List, Tuple

from models import MapState
from genetic_solver import GeneticSolver
from optimizer import optimize_solution


class BlindSolver:
    """
    Finds the optimal solution by running repetitive Genetic Algorithms (GA) without knowing the target area.

    Stopping Criteria: Stop if no improvement is observed after a certain number of attempts.
    """

    # --- CONSTANTS ---
    DEFAULT_PATIENCE = 50
    DEFAULT_MAX_ATTEMPTS = 200
    GA_POPULATION_SIZE = 1000
    GA_MUTATION_RATE = 0.5
    GA_GENERATIONS = 200

    def __init__(self, map_state: MapState):
        """
        Args:
            map_state: The state of the map
        """
        self.map_state = map_state

    def find_optimum(
        self,
        patience: int = DEFAULT_PATIENCE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        target_area: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Attempts to find the optimal solution.

        Args:
            patience: How many attempts to wait for improvement before stopping
            max_attempts: Maximum number of attempts
            target_area: Target area (if known, for early stopping)
            verbose: Whether to print output

        Returns:
            {
                'area': int,
                'walls': List[Tuple[int, int]],
                'attempts': int,
                'converged': bool
            }
        """
        best_area = 0
        best_walls: List[Tuple[int, int]] = []
        no_improvement_count = 0
        attempt = 0

        if verbose:
            self._print_header(patience)

        while attempt < max_attempts:
            attempt += 1

            # Run new GA instance
            result = self._run_genetic_algorithm()
            current_area = result['optimal_area']

            # Evaluate result
            if current_area > best_area:
                best_area = current_area
                best_walls = result['solution']
                no_improvement_count = 0
                status = "🚀 NEW RECORD"
            else:
                no_improvement_count += 1
                status = f"Stable ({no_improvement_count}/{patience})"

            if verbose:
                print(f"Attempt {attempt}: {current_area} | Best: {best_area} | {status}")

            # Early stopping checks
            if target_area and best_area >= target_area:
                if verbose:
                    print(f"\n🎉 TARGET REACHED! Found {target_area} points!")
                break

            if no_improvement_count >= patience:
                if verbose:
                    print(f"\n⚠️ STAGNATION DETECTED at {best_area} points.")

                # Eğer hedefi (109) bulamadıysak ama hala deneme hakkımız varsa (max_attempts),
                # RADİKAL BİR KARAR AL: Popülasyonu sıfırla!
                if target_area and best_area < target_area and attempt < max_attempts * 0.8:
                    print("♻️ PERFORMING SOFT RESTART (Killing population to escape local optimum)...")

                    # Sabır sayacını sıfırla ki hemen çıkmasın
                    no_improvement_count = 0

                    # Solver'ı yeniden başlatmak için map_state'i bozmadan
                    # sadece döngünün devam etmesini sağla.
                    # (Not: Gerçek bir 'reset' için GeneticSolver her tur yeniden yaratılıyor zaten,
                    # bu yüzden sadece 'continue' diyerek sabrı sıfırlamak yeterli olacaktır
                    # çünkü GA her döngüde sıfırdan başlar.)
                    continue
                else:
                    # Artık gerçekten yapacak bir şey yok, bitir.
                    if verbose:
                        print(f"✅ SATURATION REACHED: Could not beat {best_area}.")
                    break

        return {
            'area': best_area,
            'walls': best_walls,
            'attempts': attempt,
            'converged': no_improvement_count >= patience
        }

    def _run_genetic_algorithm(self) -> Dict:
        """Runs a single GA instance and optimizes the result."""
        ga = GeneticSolver(
            self.map_state,
            population_size=self.GA_POPULATION_SIZE,
            mutation_rate=self.GA_MUTATION_RATE
        )
        result = ga.solve(generations=self.GA_GENERATIONS)

        # GA çözümünü optimize et
        if result['solvable'] and result['solution']:
            optimized_walls, optimized_area = optimize_solution(
                result['solution'],
                self.map_state.grid,
                self.map_state.moby_pos,
                self.map_state.water_cells,
                max_iterations=15,
                verbose=False  # Her GA'da çıktı vermesin
            )

            if optimized_area > result['optimal_area']:
                result['solution'] = optimized_walls
                result['optimal_area'] = optimized_area

        return result

    def _print_header(self, patience: int) -> None:
        """Prints the header."""
        print("=" * 60)
        print("BLIND SOLVER: SOFT PENALTY & STOCHASTIC BLOCKER")
        print(f"Strategy: Preserve 'almost there' solutions, place walls smartly.")
        print(f"Stopping Criteria: If no improvement for {patience} consecutive attempts.")
        print("=" * 60)