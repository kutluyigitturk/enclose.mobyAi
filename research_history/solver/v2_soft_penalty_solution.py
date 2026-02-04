"""
EXPERIMENT 2: ADVANCED GENETIC ALGORITHM (SUCCESSFUL ATTEMPT)
-------------------------------------------------------------
Status: Global Optimum Found
Score: 68 Points (Optimal)

Improvements Implemented:
1. Soft Penalty: Even if escape is possible, give partial score if few escape routes exist.
   (This allowed the algorithm to cross the "Valley of Death").
2. Stochastic Path Blocking: Find Moby's escape path and block it directly.
   (Smart mutation instead of random placement).
3. Geometry Mutation: Tendency to cluster walls together (building blocks).
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
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


def place_walls(grid: np.ndarray, walls: List[Tuple[int, int]]) -> np.ndarray:
    new_grid = grid.copy()
    for x, y in walls: new_grid[y, x] = BUOY
    return new_grid


# --- ADVANCED ANALYSIS FUNCTIONS ---

def analyze_escape(grid: np.ndarray, start: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculates both the reachable area and the number of escape points (Frontiers).
    """
    rows, cols = grid.shape
    visited = {start}
    queue = deque([start])
    frontiers = 0
    area = 0
    escaped = False

    # Boundary cells
    boundary_cells = set()

    while queue:
        cx, cy = queue.popleft()
        area += 1

        # Reached the edge?
        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            escaped = True
            # Count distinct boundary exits (Simple heuristic)
            # Treat the boundary cell itself as a frontier
            boundary_cells.add((cx, cy))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] not in (LAND, BUOY) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return area, len(boundary_cells), escaped


def get_stochastic_escape_path(grid: np.ndarray, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Randomized BFS.
    Can find a different shortest path each time it is called.
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            return path + [(cx, cy)]

        # Shuffle directions! (Stochasticity)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] not in (LAND, BUOY) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [(cx, cy)]
                    queue.append(((nx, ny), new_path))
    return []


# --- 2. GENETIC SOLVER (SOFT PENALTY + GEOMETRY MODE) ---
class GeneticSolver:
    def __init__(self, map_state: MapState, population_size: int = 300, mutation_rate: float = 0.3):
        self.map_state = map_state
        self.grid = map_state.grid
        self.rows, self.cols = self.grid.shape
        self.moby_pos = map_state.moby_pos
        self.max_walls = map_state.max_walls
        self.water_cells = map_state.water_cells

        self.pop_size = population_size
        self.mutation_rate = mutation_rate

    def fitness(self, walls: List[Tuple[int, int]]) -> float:
        """
        NEW FITNESS: Give escapers a chance to live, but treat them poorly.
        """
        test_grid = place_walls(self.grid, walls)
        area, frontiers, escaped = analyze_escape(test_grid, self.moby_pos)

        if not escaped:
            # Trap Successful: Lion's Share
            return 20000 + (area * 100)
        else:
            # Escape Exists: "Soft Penalty"
            # The smaller the area and fewer the escape points (frontiers), the better.
            # Max score will be around ~5000 (Must remain below Trap score)

            # Frontier penalty: Each escape hole removes 500 points
            frontier_penalty = frontiers * 500

            # Area bonus (Stay in a small area even if escaping)
            area_score = (len(self.water_cells) - area) * 10

            score = 5000 + area_score - frontier_penalty
            return max(100, score)  # Never return 0 or negative, keep evolution going

    def create_individual(self):
        # Semi-Random, Semi-Edge Focused Initialization
        walls = []
        while len(walls) < self.max_walls:
            # 50% chance to pick a spot near the right edge
            if random.random() < 0.5:
                # Rightmost 3 columns
                rx = random.randint(self.cols - 4, self.cols - 1)
                ry = random.randint(0, self.rows - 1)
                if (rx, ry) in self.water_cells and (rx, ry) not in walls:
                    walls.append((rx, ry))
            else:
                choice = random.choice(self.water_cells)
                if choice not in walls: walls.append(choice)

        # Fill remaining slots randomly
        while len(walls) < self.max_walls:
            rem = [x for x in self.water_cells if x not in walls]
            if rem:
                walls.append(random.choice(rem))
            else:
                break

        return walls[:self.max_walls]

    def mutate(self, ind: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if random.random() < self.mutation_rate:
            rand_val = random.random()

            # Strategy 1: PATH BLOCKING (40%)
            if rand_val < 0.40:
                current_grid = place_walls(self.grid, ind)
                # Use stochastic path finder
                path = get_stochastic_escape_path(current_grid, self.moby_pos)
                if path and len(path) > 1:
                    # Blocking somewhere in the middle/end of the path is better
                    target = path[random.randint(1, len(path) - 1)]

                    idx = random.randint(0, self.max_walls - 1)
                    ind[idx] = target

            # Strategy 2: WALL BROTHERHOOD / GEOMETRY (30%)
            elif rand_val < 0.70:
                # Pick a random wall
                ref_idx = random.randint(0, self.max_walls - 1)
                ref_wall = ind[ref_idx]

                # Move another wall to its neighbor
                neighbors = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = ref_wall[0] + dx, ref_wall[1] + dy
                    if (nx, ny) in self.water_cells and (nx, ny) not in ind:
                        neighbors.append((nx, ny))

                if neighbors:
                    target = random.choice(neighbors)
                    move_idx = random.randint(0, self.max_walls - 1)
                    if move_idx != ref_idx:  # Not itself
                        ind[move_idx] = target

            # Strategy 3: RANDOM (30%)
            else:
                idx = random.randint(0, self.max_walls - 1)
                rem = [x for x in self.water_cells if x not in ind]
                if rem: ind[idx] = random.choice(rem)

        return ind

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

    def solve(self, generations: int = 150) -> Dict:
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_sol = None
        best_sc = -float('inf')

        for gen in range(generations):
            # Calculate Fitness
            scs = []
            for i in population:
                scs.append((i, self.fitness(i)))

            # Update Best
            gb = max(scs, key=lambda x: x[1])
            if gb[1] > best_sc:
                best_sc = gb[1]
                best_sol = gb[0].copy()

            # Elitism: Keep top 10%
            scs.sort(key=lambda x: x[1], reverse=True)
            new_pop = [x[0] for x in scs[:int(self.pop_size * 0.1)]]

            # Use "Tournament Selection" instead of Roulette Wheel (Preserves diversity)
            while len(new_pop) < self.pop_size:
                # Pick best of 5 random individuals
                parent1 = max(random.sample(scs, 5), key=lambda x: x[1])[0]
                parent2 = max(random.sample(scs, 5), key=lambda x: x[1])[0]

                child = self.mutate(self.crossover(parent1, parent2))
                new_pop.append(child)

            population = new_pop

        if best_sol:
            # Final check
            grid = place_walls(self.grid, best_sol)
            area, _, escaped = analyze_escape(grid, self.moby_pos)
            if escaped: return {'optimal_area': 0, 'solution': best_sol}  # Still escaping -> 0
            return {'optimal_area': area, 'solution': best_sol}

        return {'optimal_area': 0, 'solution': []}


# --- 3. BLIND SOLVER ---
class BlindSolver:
    def __init__(self, map_state):
        self.map_state = map_state

    def find_optimum(self, patience=15, max_attempts=50):
        best_area = 0;
        best_walls = [];
        no_imp = 0;
        att = 0
        print(f"BLIND SOLVER: SOFT PENALTY & STOCHASTIC BLOCKER")
        print(f"Strategy: Protect 'almost there' solutions, cluster walls.")

        while att < max_attempts:
            att += 1
            # High population for diversity
            ga = GeneticSolver(self.map_state, population_size=400, mutation_rate=0.4)
            res = ga.solve(generations=120)
            cur_area = res['optimal_area']

            status = "🚀 RECORD" if cur_area > best_area else f"Stable ({no_imp + 1}/{patience})"
            if cur_area > best_area:
                best_area = cur_area;
                best_walls = res['solution'];
                no_imp = 0
            else:
                no_imp += 1

            print(f"Attempt {att}: {cur_area} | Best: {best_area} | {status}")

            if best_area >= 68:
                print("\n🎉 VICTORY! 68 POINTS FOUND!")
                break

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

    def print_grid(walls_to_print, title):
        grid = np.array(pathfinder_map)
        moby_pos = (9, 7)
        rows, cols = grid.shape
        for x, y in walls_to_print: grid[y, x] = BUOY
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
    visualize_result(result['walls'], result['area'])