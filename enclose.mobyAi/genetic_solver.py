"""
Genetic Algorithm based solver (Island Model).

Key Idea:
- Instead of one mixed population, we maintain 4 SEPARATE populations (Islands).
- Each island focuses on a different quadrant of the map.
- Migration between islands is BANNED to prevent the 87-point solution (local optimum)
  from dominating the 109-point solution (global optimum) during early generations.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Set, FrozenSet, Optional
import random
import numpy as np

from models import MapState
from analysis import (
    analyze_escape_walls,
    analyze_escape_detailed_walls,
    get_stochastic_escape_path_walls,
)


Coord = Tuple[int, int]
IndexSet = Set[int]


class GeneticSolver:
    # --- FITNESS CONSTANTS ---
    BIG_GAP = 1_000_000
    ESCAPE_FRONTIER_PENALTY = 5000

    # --- GA CONSTANTS ---
    ELITE_RATIO = 0.05
    TOURNAMENT_SIZE = 3

    # Mutasyon
    MUTATION_RATE = 0.6
    MUTATION_SHIFT_PROB = 0.90

    GENERATION_PATIENCE = 150

    def __init__(
        self,
        map_state: MapState,
        population_size: int = 400, # Her adaya 100 kişi düşecek
        mutation_rate: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.map_state = map_state
        self.grid = map_state.grid
        self.rows, self.cols = self.grid.shape
        self.moby_pos = map_state.moby_pos
        self.max_walls = map_state.max_walls

        self.water_cells: List[Coord] = map_state.water_cells
        self.n = len(self.water_cells)
        self.coord_to_idx: Dict[Coord, int] = {c: i for i, c in enumerate(self.water_cells)}

        self.population_size = population_size
        self.mutation_rate = mutation_rate

        if seed is not None:
            random.seed(seed)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _indices_to_walls(self, ind: IndexSet) -> List[Coord]:
        return [self.water_cells[i] for i in ind]

    def _indices_to_walls_set(self, ind: IndexSet) -> Set[Coord]:
        return {self.water_cells[i] for i in ind}

    def fitness(self, ind: IndexSet) -> float:
        walls_set = self._indices_to_walls_set(ind)
        a = analyze_escape_walls(self.grid, self.moby_pos, walls_set)

        if not a.escaped:
            return self.BIG_GAP + a.area
        return -self.BIG_GAP - (a.frontier_count * self.ESCAPE_FRONTIER_PENALTY) - a.area

    # ----------------------------
    # Initialization & Repair (Targeted)
    # ----------------------------

    def create_individual(self, quadrant: int = 0) -> IndexSet:
        """
        Creates an individual biased towards a specific quadrant.
        quadrant: 0=Any, 1=Top-Left, 2=Top-Right, 3=Bottom-Left, 4=Bottom-Right
        """
        ind: Set[int] = set()

        # --- Bölge Sınırlarını Belirle ---
        mid_x = self.cols // 2
        mid_y = self.rows // 2

        allowed_indices = []
        for i, (x, y) in enumerate(self.water_cells):
            if (x, y) == self.moby_pos: continue

            is_valid = True
            if quadrant == 1 and not (x <= mid_x and y <= mid_y): is_valid = False
            elif quadrant == 2 and not (x >= mid_x and y <= mid_y): is_valid = False
            elif quadrant == 3 and not (x <= mid_x and y >= mid_y): is_valid = False
            elif quadrant == 4 and not (x >= mid_x and y >= mid_y): is_valid = False

            if is_valid: allowed_indices.append(i)

        # Eğer bölge çok darsa tüm haritayı aç (fallback)
        if len(allowed_indices) < self.max_walls:
            allowed_indices = [i for i in range(self.n) if self.water_cells[i] != self.moby_pos]

        # --- Greedy Build ---
        for _ in range(self.max_walls * 2):
            if len(ind) >= self.max_walls: break

            walls_set = self._indices_to_walls_set(ind)
            analysis, visited, boundary = analyze_escape_detailed_walls(self.grid, self.moby_pos, walls_set)

            # Sadece izin verilen bölgeden seç
            remaining = [i for i in allowed_indices if i not in ind]
            if not remaining: break

            pick_pool = []

            if analysis.escaped:
                # Path Blocking (Bölge içi)
                paths = [get_stochastic_escape_path_walls(self.grid, self.moby_pos, walls_set) for _ in range(2)]
                path = max(paths, key=len, default=[])
                path = path[2:] # Moby'nin dibine koyma

                path_pool = [self.coord_to_idx[c] for c in path if c in self.coord_to_idx]
                path_pool = [i for i in path_pool if i in remaining] # Filtrele

                if path_pool: pick_pool = path_pool
                else: pick_pool = remaining
            else:
                # Trapped: Dışarı doğru genişle
                outside = [self.coord_to_idx[c] for c in self.water_cells if c not in visited and c in self.coord_to_idx]
                pick_pool = [i for i in outside if i in remaining]

            if not pick_pool: pick_pool = remaining
            ind.add(random.choice(pick_pool))

        return self._repair_to_k(ind)

    def _repair_to_k(self, ind: IndexSet) -> IndexSet:
        ind = set(ind)
        if len(ind) == self.max_walls: return ind
        if len(ind) > self.max_walls: return set(random.sample(list(ind), self.max_walls))

        # Basit tamamlama (Bölge kısıtlaması olmadan, sadece tamir etsin)
        walls_set = self._indices_to_walls_set(ind)
        analysis, visited, boundary = analyze_escape_detailed_walls(self.grid, self.moby_pos, walls_set)

        remaining = [i for i in range(self.n) if i not in ind and self.water_cells[i] != self.moby_pos]

        def add_from_pool(pool: List[int]):
            random.shuffle(pool)
            for idx in pool:
                if len(ind) >= self.max_walls: break
                ind.add(idx)

        if not analysis.escaped:
            outside = [self.coord_to_idx[c] for c in self.water_cells if c not in visited and c in self.coord_to_idx]
            outside = [i for i in outside if i in remaining]
            if outside: add_from_pool(outside)
        else:
            path = get_stochastic_escape_path_walls(self.grid, self.moby_pos, walls_set)
            path_pool = [self.coord_to_idx[c] for c in path if c in self.coord_to_idx]
            path_pool = [i for i in path_pool if i in remaining]
            if path_pool: add_from_pool(path_pool)

        if len(ind) < self.max_walls: add_from_pool(remaining)
        return ind

    # ----------------------------
    # Genetic Operators
    # ----------------------------

    def _tournament_select(self, scored: List[Tuple[IndexSet, float]]) -> IndexSet:
        tournament = random.sample(scored, self.TOURNAMENT_SIZE)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return set(tournament[0][0])

    def crossover(self, p1: IndexSet, p2: IndexSet) -> IndexSet:
        p1 = set(p1)
        p2 = set(p2)
        common = p1 & p2
        child = set(common)
        pool = list((p1 | p2) - common)
        random.shuffle(pool)
        for idx in pool:
            if len(child) >= self.max_walls: break
            child.add(idx)
        return self._repair_to_k(child)

    def mutate(self, ind: IndexSet) -> IndexSet:
        if random.random() >= self.mutation_rate: return ind
        ind = set(ind)
        if not ind: return self.create_individual(quadrant=0)

        # Shift or Swap
        if random.random() < self.MUTATION_SHIFT_PROB:
            # SHIFT
            wall_idx = random.choice(list(ind))
            wx, wy = self.water_cells[wall_idx]
            neighbors = [(wx+1, wy), (wx-1, wy), (wx, wy+1), (wx, wy-1)]
            valid_neighbors = []
            for nx, ny in neighbors:
                if (nx, ny) == self.moby_pos: continue
                n_idx = self.coord_to_idx.get((nx, ny))
                if n_idx is not None and n_idx not in ind:
                    valid_neighbors.append(n_idx)
            if valid_neighbors:
                ind.remove(wall_idx)
                ind.add(random.choice(valid_neighbors))
        else:
            # SWAP
            ind.remove(random.choice(list(ind)))
            remaining = [i for i in range(self.n) if i not in ind and self.water_cells[i] != self.moby_pos]
            if remaining: ind.add(random.choice(remaining))
            ind = self._repair_to_k(ind)

        return ind

    # ----------------------------
    # MAIN LOOP (THE ISLAND MODEL)
    # ----------------------------

    def solve(self, generations: int = 150) -> Dict:
        """
        Runs 4 parallel islands that do NOT mix.
        """
        island_size = self.population_size // 4

        # 4 Ayrı Ada Oluştur (Kuzey-Batı, Kuzey-Doğu, Güney-Batı, Güney-Doğu)
        islands = []
        for q in range(1, 5):
            island_pop = [self.create_individual(quadrant=q) for _ in range(island_size)]
            islands.append(island_pop)

        global_best_ind = None
        global_best_fit = -float("inf")

        for _gen in range(generations):

            # Her adayı kendi içinde evrimleştir
            for i in range(4):
                population = islands[i]

                # 1. Score Island
                scored = []
                for ind in population:
                    f = self.fitness(ind)
                    scored.append((ind, f))

                    # Global best check
                    if f > global_best_fit:
                        global_best_fit = f
                        global_best_ind = set(ind)

                scored.sort(key=lambda x: x[1], reverse=True)

                # 2. Elitism (Island specific)
                elite_count = max(1, int(island_size * self.ELITE_RATIO))
                new_pop = [set(x[0]) for x in scored[:elite_count]]

                # 3. Breed (Island specific - NO MIXING between islands)
                while len(new_pop) < island_size:
                    p1 = self._tournament_select(scored)
                    p2 = self._tournament_select(scored)
                    child = self.crossover(p1, p2)
                    child = self.mutate(child)
                    new_pop.append(child)

                islands[i] = new_pop

        return self._finalize_solution(global_best_ind)

    def _finalize_solution(self, best_ind: Optional[IndexSet]) -> Dict:
        if not best_ind: return {'solvable': False, 'optimal_area': 0, 'solution': []}
        walls = self._indices_to_walls(best_ind)
        walls_set = set(walls)
        a = analyze_escape_walls(self.grid, self.moby_pos, walls_set)
        if a.escaped: return {'solvable': False, 'optimal_area': 0, 'solution': walls}
        return {'solvable': True, 'optimal_area': a.area, 'solution': walls}