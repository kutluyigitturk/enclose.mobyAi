"""
Genetic Algorithm based solver.
"""

from typing import List, Tuple, Dict
import random
import numpy as np

from models import MapState
from analysis import place_walls, analyze_escape, get_stochastic_escape_path


class GeneticSolver:
    """
    Genetic Algorithm attempting to maximize the enclosed area for Moby.

    Features:
        - Soft Penalty: Evolution continues even in escape scenarios
        - Stochastic path blocking: Exploring different escape routes
        - Tournament selection: Preserving diversity
    """

    # --- FITNESS CONSTANTS ---
    FITNESS_TRAP_BASE = 20000          # Base score for successful entrapment
    FITNESS_AREA_MULTIPLIER = 100      # Multiplier per area unit
    FITNESS_SOFT_BASE = 5000           # Base score for escape scenarios
    FITNESS_FRONTIER_PENALTY = 500     # Penalty for each escape frontier
    FITNESS_MIN_SCORE = 100            # Minimum fitness score

    # --- MUTATION STRATEGY RATIOS ---
    MUTATION_PATH_BLOCK_THRESHOLD = 0.40    # 40% path blocking
    MUTATION_GEOMETRY_THRESHOLD = 0.70      # 30% geometry (0.70 - 0.40)
    # Remaining 30% is random

    # --- OTHER CONSTANTS ---
    ELITE_RATIO = 0.10                 # Elite individual ratio
    TOURNAMENT_SIZE = 5                # Tournament selection size
    EDGE_BIAS_PROBABILITY = 0.50       # Edge-focused start probability
    EDGE_COLUMN_COUNT = 4              # Number of columns considered as edge

    def __init__(
        self,
        map_state: MapState,
        population_size: int = 300,
        mutation_rate: float = 0.3
    ):
        """
        Args:
            map_state: The state of the map
            population_size: Size of the population
            mutation_rate: Mutation rate (0.0 - 1.0)
        """
        self.map_state = map_state
        self.grid = map_state.grid
        self.rows, self.cols = self.grid.shape
        self.moby_pos = map_state.moby_pos
        self.max_walls = map_state.max_walls
        self.water_cells = map_state.water_cells
        self.water_cells_set = map_state.water_cells_set

        self.population_size = population_size
        self.mutation_rate = mutation_rate

    # --- FITNESS CALCULATION ---

    def fitness(self, walls: List[Tuple[int, int]]) -> float:
        """
        Calculates the fitness score of a wall configuration.

        Soft Penalty approach:
        - Entrapment successful: High score + area bonus
        - Escape exists: Low score but evolution continues

        Args:
            walls: List of wall coordinates

        Returns:
            Fitness score (higher is better)
        """
        test_grid = place_walls(self.grid, walls)
        analysis = analyze_escape(test_grid, self.moby_pos)

        if not analysis.escaped:
            # Entrapment successful - high score
            return self.FITNESS_TRAP_BASE + (analysis.area * self.FITNESS_AREA_MULTIPLIER)
        else:
            # Escape exists - soft penalty
            frontier_penalty = analysis.frontier_count * self.FITNESS_FRONTIER_PENALTY
            area_score = (self.map_state.total_water_count - analysis.area) * 10
            score = self.FITNESS_SOFT_BASE + area_score - frontier_penalty
            return max(self.FITNESS_MIN_SCORE, score)

    # --- INDIVIDUAL CREATION ---

    def create_individual(self) -> List[Tuple[int, int]]:
        """
        Creates a new individual (wall configuration).

        Strategy: Semi-random, semi-edge focused start.
        """
        walls = []
        max_attempts = self.max_walls * 10  # Infinite loop protection
        attempts = 0

        while len(walls) < self.max_walls and attempts < max_attempts:
            attempts += 1

            if random.random() < self.EDGE_BIAS_PROBABILITY:
                # Choose a spot near the right edge
                pos = self._get_random_edge_position()
            else:
                # Completely random
                pos = random.choice(self.water_cells)

            if pos is not None and pos not in walls:
                walls.append(pos)

        # Fill remaining if missing
        self._fill_remaining_walls(walls)

        return walls[:self.max_walls]

    def _get_random_edge_position(self) -> Tuple[int, int]:
        """Returns a random water cell near the right edge."""
        edge_start = self.cols - self.EDGE_COLUMN_COUNT

        for _ in range(20):  # Max attempts
            rx = random.randint(edge_start, self.cols - 1)
            ry = random.randint(0, self.rows - 1)
            if (rx, ry) in self.water_cells_set:
                return (rx, ry)

        return random.choice(self.water_cells)

    def _fill_remaining_walls(self, walls: List[Tuple[int, int]]) -> None:
        """Fills missing walls randomly."""
        remaining = [cell for cell in self.water_cells if cell not in walls]
        needed = self.max_walls - len(walls)

        if needed > 0 and remaining:
            additions = random.sample(remaining, min(needed, len(remaining)))
            walls.extend(additions)

    # --- MUTATION ---

    def mutate(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Applies mutation to the individual.

        Three strategies:
        1. Path blocking (40%): Block the active escape route
        2. Geometry (30%): Move to a neighbor of an existing wall
        3. Random (30%): Change completely randomly
        """
        if random.random() >= self.mutation_rate:
            return individual

        rand_val = random.random()

        if rand_val < self.MUTATION_PATH_BLOCK_THRESHOLD:
            return self._mutate_path_block(individual)
        elif rand_val < self.MUTATION_GEOMETRY_THRESHOLD:
            return self._mutate_geometry(individual)
        else:
            return self._mutate_random(individual)

    def _mutate_path_block(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Escape path blocking mutation."""
        current_grid = place_walls(self.grid, individual)
        path = get_stochastic_escape_path(current_grid, self.moby_pos)

        if path and len(path) > 1:
            # Block a spot in the middle/end of the path
            target = path[random.randint(1, len(path) - 1)]
            idx = random.randint(0, self.max_walls - 1)
            individual[idx] = target

        return individual

    def _mutate_geometry(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Neighbor wall mutation (wall brotherhood)."""
        ref_idx = random.randint(0, self.max_walls - 1)
        ref_wall = individual[ref_idx]

        # Find neighbors of the reference wall
        neighbors = self._get_water_neighbors(ref_wall, individual)

        if neighbors:
            target = random.choice(neighbors)
            move_idx = random.randint(0, self.max_walls - 1)
            if move_idx != ref_idx:
                individual[move_idx] = target

        return individual

    def _mutate_random(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Random mutation."""
        idx = random.randint(0, self.max_walls - 1)
        remaining = [cell for cell in self.water_cells if cell not in individual]

        if remaining:
            individual[idx] = random.choice(remaining)

        return individual

    def _get_water_neighbors(
        self,
        pos: Tuple[int, int],
        exclude: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Returns water neighbors of the given position."""
        neighbors = []
        x, y = pos

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.water_cells_set and (nx, ny) not in exclude:
                neighbors.append((nx, ny))

        return neighbors

    # --- CROSSOVER ---

    def crossover(
        self,
        parent1: List[Tuple[int, int]],
        parent2: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Creates a new individual from two parents (single point crossover).
        """
        crossover_point = random.randint(1, self.max_walls - 1)
        child = list(set(parent1[:crossover_point] + parent2[crossover_point:]))

        # Fill missing walls
        self._fill_remaining_walls(child)

        return child[:self.max_walls]

    # --- SELECTION ---

    def _tournament_select(
        self,
        scored_population: List[Tuple[List[Tuple[int, int]], float]]
    ) -> List[Tuple[int, int]]:
        """Selects parent using tournament selection."""
        tournament = random.sample(scored_population, self.TOURNAMENT_SIZE)
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    # --- MAIN SOLVER LOOP ---

    def solve(self, generations: int = 150) -> Dict:
        """
        Runs the genetic algorithm.

        Args:
            generations: Number of generations

        Returns:
            {
                'solvable': bool,
                'optimal_area': int,
                'solution': List[Tuple[int, int]]
            }
        """
        population = [self.create_individual() for _ in range(self.population_size)]
        best_solution = None
        best_score = -float('inf')

        for generation in range(generations):
            # Calculate Fitness
            scored_population = [(ind, self.fitness(ind)) for ind in population]

            # Update Best
            generation_best = max(scored_population, key=lambda x: x[1])
            if generation_best[1] > best_score:
                best_score = generation_best[1]
                best_solution = generation_best[0].copy()

            # Create New Population
            population = self._create_next_generation(scored_population)

        # Final check
        return self._finalize_solution(best_solution)

    def _create_next_generation(
        self,
        scored_population: List[Tuple[List[Tuple[int, int]], float]]
    ) -> List[List[Tuple[int, int]]]:
        """Creates the next generation."""
        # Sort
        scored_population.sort(key=lambda x: x[1], reverse=True)

        # Elitism: Keep the top 10%
        elite_count = int(self.population_size * self.ELITE_RATIO)
        new_population = [x[0] for x in scored_population[:elite_count]]

        # Generate remaining individuals
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(scored_population)
            parent2 = self._tournament_select(scored_population)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        return new_population

    def _finalize_solution(self, solution: List[Tuple[int, int]]) -> Dict:
        """Performs final check on the solution."""
        if not solution:
            return {'solvable': False, 'optimal_area': 0, 'solution': []}

        grid = place_walls(self.grid, solution)
        analysis = analyze_escape(grid, self.moby_pos)

        if analysis.escaped:
            return {'solvable': False, 'optimal_area': 0, 'solution': solution}

        return {
            'solvable': True,
            'optimal_area': analysis.area,
            'solution': solution
        }