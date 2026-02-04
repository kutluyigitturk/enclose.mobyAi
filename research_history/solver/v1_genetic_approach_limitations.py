"""
EXPERIMENT 1: BASELINE GENETIC ALGORITHM (FAILED ATTEMPT)
---------------------------------------------------------
Status: Trapped in Local Optimum
Score: 64-65 Points (Stuck)

Methods Used:
1. Hard Penalty: Immediate negative score if escape is possible.
2. Edge Snap: Strategy to snap walls to the grid edges.
3. Refine Phase: Simulated Annealing-like post-processing.

Why Did It Fail?
The algorithm could not cross the "Valley of Death". To go from 65 to 68 points,
it needed to temporarily break the wall structure (allowing escape) to rebuild it better.
The Hard Penalty prevented this risk-taking behavior, causing the algorithm to freeze.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
import random
import time
import math

# --- 1. CONSTANTS ---
WATER = 0
LAND = 1
BUOY = 2
MOBY = 3


class MapState:
    def __init__(self, grid: np.ndarray, moby_pos: Tuple[int, int], max_walls: int):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.moby_pos = moby_pos
        self.max_walls = max_walls
        self.water_cells = []
        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y, x] == WATER and (x, y) != moby_pos:
                    self.water_cells.append((x, y))


def flood_fill(grid: np.ndarray, start: Tuple[int, int]) -> Tuple[int, bool]:
    rows, cols = grid.shape
    x, y = start
    if grid[y, x] in (LAND, BUOY): return 0, False
    visited = set()
    queue = deque([start])
    visited.add(start)
    escaped = False
    while queue:
        cx, cy = queue.popleft()
        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1: escaped = True
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in visited and grid[ny, nx] not in (LAND, BUOY):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return len(visited), escaped


def place_walls(grid: np.ndarray, walls: List[Tuple[int, int]]) -> np.ndarray:
    new_grid = grid.copy()
    for x, y in walls: new_grid[y, x] = BUOY
    return new_grid


# --- 2. GENETIC SOLVER (SIMPLIFIED + EDGE SNAP) ---
class GeneticSolver:
    def __init__(self, map_state: MapState, population_size: int = 300, mutation_rate: float = 0.2,
                 connected_prob: float = 0.7):
        # Connected Prob set back to 0.7 because disconnected walls killed performance.
        self.map_state = map_state
        self.grid = map_state.grid
        self.rows, self.cols = self.grid.shape
        self.moby_pos = map_state.moby_pos
        self.max_walls = map_state.max_walls
        self.water_cells = map_state.water_cells

        # Anchor Points (Only Land)
        self.anchor_cells = []
        for wx, wy in self.water_cells:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = wx + dx, wy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and self.grid[ny, nx] == LAND:
                    self.anchor_cells.append((wx, wy))
                    break

        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.connected_prob = connected_prob

    def fitness(self, walls: List[Tuple[int, int]]) -> float:
        """
        FITNESS: BACK TO FACTORY SETTINGS (Binary System)
        If escapes: ZERO (or negative). If trapped: SCORE = AREA.
        This system guaranteed finding 64.
        """
        test_grid = place_walls(self.grid, walls)
        area, escaped = flood_fill(test_grid, self.moby_pos)

        if escaped:
            # Very low score if escaped, but not 0 (to see progress)
            return -area
        else:
            # If trapped: Base Score + Area
            return 10000 + (area * 100)

    def create_connected_individual(self):
        walls = []
        if not self.water_cells: return []
        walls.append(random.choice(self.water_cells))
        while len(walls) < self.max_walls:
            candidates = []
            for w_x, w_y in walls:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = w_x + dx, w_y + dy
                    if (nx, ny) in self.water_cells and (nx, ny) not in walls: candidates.append((nx, ny))
            if candidates:
                walls.append(random.choice(candidates))
            else:
                rem = [c for c in self.water_cells if c not in walls]
                if rem:
                    walls.append(random.choice(rem))
                else:
                    break
        return walls

    def create_individual(self):
        if random.random() < self.connected_prob: return self.create_connected_individual()
        indices = np.random.choice(len(self.water_cells), self.max_walls, replace=False)
        return [self.water_cells[i] for i in indices]

    def crossover(self, p1, p2):
        pt = random.randint(1, self.max_walls - 1)
        c = list(set(p1[:pt] + p2[pt:]))
        while len(c) < self.max_walls:
            rem = [x for x in self.water_cells if x not in c]
            if rem:
                c.append(random.choice(rem))
            else:
                break
        return c

    def mutate(self, ind):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.max_walls - 1)
            rem = [x for x in self.water_cells if x not in ind]
            if rem: ind[idx] = random.choice(rem)
        return ind

    def refine_solution(self, walls: List[Tuple[int, int]], iterations=50000) -> Tuple[List[Tuple[int, int]], int]:
        """
        SIMPLIFIED SA: MAGNET + EDGE SNAP
        We kept only the two strategies that worked.
        """
        current_walls = walls.copy()
        current_score = self.fitness(current_walls)
        best_walls = current_walls.copy()
        best_score = current_score

        # Balanced Temperature
        temperature = 50000.0
        cooling_rate = 0.9998

        for i in range(iterations):
            new_walls = current_walls.copy()
            idx_to_move = random.randint(0, self.max_walls - 1)

            new_pos = None
            rand_val = random.random()

            # --- STRATEGY 1: EDGE SNAP (30%) ---
            # Key to reaching 68 (in theory). Snaps wall to the nearest map boundary.
            if rand_val < 0.30:
                wx, wy = new_walls[idx_to_move]
                candidates = []
                # 4 Edges: Right, Left, Top, Bottom
                if (self.cols - 1, wy) in self.water_cells and (self.cols - 1, wy) not in new_walls: candidates.append(
                    (self.cols - 1, wy))
                if (0, wy) in self.water_cells and (0, wy) not in new_walls: candidates.append((0, wy))
                if (wx, self.rows - 1) in self.water_cells and (wx, self.rows - 1) not in new_walls: candidates.append(
                    (wx, self.rows - 1))
                if (wx, 0) in self.water_cells and (wx, 0) not in new_walls: candidates.append((wx, 0))

                if candidates: new_pos = random.choice(candidates)

            # --- STRATEGY 2: MAGNET (50%) ---
            # Ensures walls stick together (Crucial for containment integrity)
            elif rand_val < 0.80:
                candidates = set()
                for wx, wy in new_walls:
                    # Look around other walls (excluding itself)
                    if (wx, wy) == new_walls[idx_to_move]: continue

                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = wx + dx, wy + dy
                        if (0 <= nx < self.cols and 0 <= ny < self.rows and
                                (nx, ny) in self.water_cells and (nx, ny) not in new_walls):
                            candidates.add((nx, ny))
                if candidates: new_pos = random.choice(list(candidates))

            # --- STRATEGY 3: RANDOM (20%) ---
            if new_pos is None:
                new_pos = random.choice(self.water_cells)
                while new_pos in new_walls:
                    new_pos = random.choice(self.water_cells)

            new_walls[idx_to_move] = new_pos
            new_score = self.fitness(new_walls)

            accept = False
            if new_score > current_score:
                accept = True
            else:
                delta = new_score - current_score
                try:
                    prob = math.exp(delta / temperature)
                except:
                    prob = 0
                if random.random() < prob: accept = True

            if accept:
                current_walls = new_walls
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_walls = current_walls.copy()

            temperature *= cooling_rate
            if temperature < 1.0: break

        final_grid = place_walls(self.grid, best_walls)
        final_area, final_escaped = flood_fill(final_grid, self.moby_pos)
        if final_escaped: return best_walls, 0
        return best_walls, final_area

    def solve(self, generations: int = 80) -> Dict:
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_sol = None;
        best_sc = -float('inf')
        for _ in range(generations):
            scs = [(i, self.fitness(i)) for i in population]
            gb = max(scs, key=lambda x: x[1])
            if gb[1] > best_sc: best_sc = gb[1];
            best_sol = gb[0]
            scs.sort(key=lambda x: x[1], reverse=True)
            pop = [x[0] for x in scs[:int(self.pop_size * 0.1)]]
            while len(pop) < self.pop_size:
                p1, p2 = random.choice(pop), random.choice(pop)
                pop.append(self.mutate(self.crossover(p1, p2)))
            population = pop
        if best_sol:
            final_walls, final_area = self.refine_solution(best_sol)
            return {'solvable': final_area > 0, 'optimal_area': final_area, 'solution': final_walls}
        return {'solvable': False, 'optimal_area': 0, 'solution': []}


# --- 3. BLIND SOLVER ---
class BlindSolver:
    def __init__(self, map_state):
        self.map_state = map_state

    def find_optimum(self, patience=5, max_attempts=50):
        best_area = 0;
        best_walls = [];
        no_imp = 0;
        att = 0
        print(f"BLIND SOLVER: FACTORY SETTINGS + EDGE SNAP")
        print(f"Goal: Guarantee 64 first, then jump to 68.")
        while att < max_attempts:
            att += 1
            ga = GeneticSolver(self.map_state, population_size=250)
            res = ga.solve(generations=80)
            cur_area = res['optimal_area']
            status = "🚀 RECORD" if cur_area > best_area else f"Stable ({no_imp + 1}/{patience})"
            if cur_area > best_area:
                best_area = cur_area;
                best_walls = res['solution'];
                no_imp = 0
            else:
                no_imp += 1
            print(f"Attempt {att}: {cur_area} | Best: {best_area} | {status}")
            if no_imp >= patience: break
        return {'area': best_area, 'walls': best_walls, 'attempts': att}


# --- 4. VISUALIZATION ---
def visualize_result(walls_found, score_found):
    walls_optimal = [(12, 0), (7, 3), (6, 4), (18, 4), (18, 5), (17, 6), (15, 8), (5, 9), (11, 11), (6, 13)]
    pathfinder_map = [
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]
    ]

    def print_grid(walls, title):
        grid = np.array(pathfinder_map)
        moby_pos = (9, 7)
        rows, cols = grid.shape
        for x, y in walls: grid[y, x] = BUOY
        print(f"\n--- {title} ---")
        for y in range(rows):
            line = ""
            for x in range(cols):
                val = grid[y, x]
                if (x, y) == moby_pos:
                    char = " @ "
                elif val == BUOY:
                    char = "[X]"
                elif val == LAND:
                    char = "###"
                else:
                    char = " . "
                line += char
            print(line)

    print_grid(walls_found, f"YOUR RESULT ({score_found} Points)")
    print_grid(walls_optimal, "TARGET (68 Points)")
    set_found = set(walls_found)
    set_optimal = set(walls_optimal)
    print("\nCommon Walls:", set_found.intersection(set_optimal))
    print("Redundant Placed:", set_found - set_optimal)
    print("Missing (Target):", set_optimal - set_found)


if __name__ == "__main__":
    pathfinder_map = [
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]
    ]
    grid = np.array(pathfinder_map)
    map_state = MapState(grid, (9, 7), 10)
    blind_solver = BlindSolver(map_state)
    result = blind_solver.find_optimum(patience=15)
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Estimated Optimal Area: {result['area']}")
    print(f"Walls: {result['walls']}")
    print(f"Total Attempts: {result['attempts']}")
    print("\n--- RESULT VISUALIZATION ---")
    visualize_result(result['walls'], result['area'])