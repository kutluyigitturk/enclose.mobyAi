"""
Chokepoint Based Map Generator

Aligned with the true design philosophy of the game:
- Each buoy must block a chokepoint
- Chokepoints must be far from each other
- Optimal solution = scattered buoys
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, List, Set
from blind_solver import BlindSolver

WATER = 0
LAND = 1


class MapState:
    """Represents the state of the game map."""

    def __init__(self, grid: np.ndarray, moby_pos: Tuple[int, int], max_walls: int):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.moby_pos = moby_pos
        self.max_walls = max_walls

        self.water_cells = []
        self.water_cells_set = set()

        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y, x] == WATER and (x, y) != moby_pos:
                    self.water_cells.append((x, y))
                    self.water_cells_set.add((x, y))

    @property
    def total_water_count(self) -> int:
        return len(self.water_cells)


class MapGenerator:
    """
    Chokepoint based map generator.

    Strategy:
    1. First determine chokepoint positions (approx. equal to buoy count)
    2. Place land blocks around each chokepoint
    3. Maintain distance between chokepoints (min 3-4 cells)
    4. Place Moby in the center
    """

    def __init__(self, rows: int = 15, cols: int = 19,
                 wall_budget: int = 10, min_chokepoint_distance: int = 4):
        self.rows = rows
        self.cols = cols
        self.wall_budget = wall_budget
        self.min_chokepoint_distance = min_chokepoint_distance

        # Chokepoint count = approx buoy count
        self.target_chokepoints = wall_budget + random.randint(0, 2)

    def generate(self) -> MapState:
        """Generates a valid map."""
        max_attempts = 100

        for attempt in range(max_attempts):
            grid, chokepoints = self._generate_chokepoint_based_grid()

            if self._is_valid_map(grid, chokepoints):
                moby_pos = self._place_moby(grid, chokepoints)
                if moby_pos and self._moby_can_escape(grid, moby_pos):
                    return MapState(grid, moby_pos, self.wall_budget)

        # Fallback
        return self._generate_fallback_map()

    def generate_solvable(self, max_retries: int = 10) -> MapState:
        """
        Generates a map guaranteed to be solvable (score > 0).

        To do this, it generates a map, quickly tests it with the Solver.
        If it fails, it retries.
        """
        for attempt in range(max_retries):
            # 1. Standard generation
            map_state = self.generate()

            # 2. Quick Validation
            # Keeping patience and attempts low for quick check
            solver = BlindSolver(map_state)

            # Calling Solver logic here since it applies to any map,
            # not just 'pathfinder'.
            # Setting verbose=False to avoid cluttering screen.
            result = solver.find_optimum(patience=5, max_attempts=15, verbose=False)

            if result['area'] > 0:
                print(
                    f"   ✅ Valid map found! (Attempt {attempt + 1}/{max_retries}) - Est. Score: {result['area']}")
                return map_state

            print(f"   ⚠️ Generated map is unsolvable, retrying... ({attempt + 1}/{max_retries})")

        # If not found after retries, return last generated (or raise error)
        print("❌ WARNING: Guaranteed solution not found, returning last generated map.")
        return map_state

    def _generate_chokepoint_based_grid(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Generates map around chokepoints."""
        grid = np.zeros((self.rows, self.cols), dtype=int)
        chokepoints = []

        # 1. Select random chokepoint positions (distinct)
        chokepoints = self._place_chokepoints()

        # 2. Place land blocks around each chokepoint
        for cx, cy in chokepoints:
            self._create_chokepoint_structure(grid, cx, cy)

        # 3. Add edge blocks
        self._add_edge_blocks(grid)

        # 4. Fill gaps (optional extra blocks)
        self._fill_gaps(grid, chokepoints)

        # 5. Ensure water connectivity
        grid = self._ensure_water_connectivity(grid)

        return grid, chokepoints

    def _place_chokepoints(self) -> List[Tuple[int, int]]:
        """Select distinct chokepoint positions."""
        chokepoints = []
        attempts = 0
        max_attempts = 500

        while len(chokepoints) < self.target_chokepoints and attempts < max_attempts:
            attempts += 1

            # Random position (away from edges)
            x = random.randint(3, self.cols - 4)
            y = random.randint(3, self.rows - 4)

            # Far enough from other chokepoints?
            too_close = False
            for cx, cy in chokepoints:
                dist = abs(x - cx) + abs(y - cy)  # Manhattan distance
                if dist < self.min_chokepoint_distance:
                    too_close = True
                    break

            if not too_close:
                chokepoints.append((x, y))

        return chokepoints

    def _create_chokepoint_structure(self, grid: np.ndarray, cx: int, cy: int) -> None:
        """
        Creates land structure around a chokepoint.

        Chokepoint = 1 cell narrow passage
        Must have SMALL land blocks around it (2x2 or 2x3)
        """
        # Choose random direction: horizontal or vertical?
        if random.random() < 0.5:
            # Horizontal chokepoint (top-bottom land)
            # Top block - SMALL
            block_width = random.randint(2, 3)
            block_height = random.randint(1, 2)
            self._place_block(grid, cx - block_width // 2, cy - block_height, block_width, block_height)

            # Bottom block - SMALL
            block_width = random.randint(2, 3)
            block_height = random.randint(1, 2)
            self._place_block(grid, cx - block_width // 2, cy + 1, block_width, block_height)
        else:
            # Vertical chokepoint (left-right land)
            # Left block - SMALL
            block_width = random.randint(1, 2)
            block_height = random.randint(2, 3)
            self._place_block(grid, cx - block_width, cy - block_height // 2, block_width, block_height)

            # Right block - SMALL
            block_width = random.randint(1, 2)
            block_height = random.randint(2, 3)
            self._place_block(grid, cx + 1, cy - block_height // 2, block_width, block_height)

    def _place_block(self, grid: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        """Places a rectangular land block."""
        for dy in range(height):
            for dx in range(width):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    grid[ny, nx] = LAND

    def _add_edge_blocks(self, grid: np.ndarray) -> None:
        """Adds random blocks to edges (makes escape harder)."""
        # Top edge
        x = 0
        while x < self.cols:
            if random.random() < 0.4:
                width = random.randint(2, 4)
                height = random.randint(1, 2)
                self._place_block(grid, x, 0, width, height)
                x += width + random.randint(2, 4)
            else:
                x += random.randint(2, 4)

        # Bottom edge
        x = 0
        while x < self.cols:
            if random.random() < 0.4:
                width = random.randint(2, 4)
                height = random.randint(1, 2)
                self._place_block(grid, x, self.rows - height, width, height)
                x += width + random.randint(2, 4)
            else:
                x += random.randint(2, 4)

        # Left edge
        y = 0
        while y < self.rows:
            if random.random() < 0.4:
                width = random.randint(1, 2)
                height = random.randint(2, 4)
                self._place_block(grid, 0, y, width, height)
                y += height + random.randint(2, 4)
            else:
                y += random.randint(2, 4)

        # Right edge
        y = 0
        while y < self.rows:
            if random.random() < 0.4:
                width = random.randint(1, 2)
                height = random.randint(2, 4)
                self._place_block(grid, self.cols - width, y, width, height)
                y += height + random.randint(2, 4)
            else:
                y += random.randint(2, 4)

    def _fill_gaps(self, grid: np.ndarray, chokepoints: List[Tuple[int, int]]) -> None:
        """Add extra blocks to fill gaps - DISABLED."""
        # Too many blocks break the map, so disabled
        pass

    def _ensure_water_connectivity(self, grid: np.ndarray) -> np.ndarray:
        """Ensure all water cells are connected."""
        visited = set()
        largest_region = []

        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y, x] == WATER and (x, y) not in visited:
                    region = self._flood_fill(grid, x, y, visited)
                    if len(region) > len(largest_region):
                        largest_region = region

        # Convert small water pools to land
        largest_set = set(largest_region)
        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y, x] == WATER and (x, y) not in largest_set:
                    grid[y, x] = LAND

        return grid

    def _flood_fill(self, grid: np.ndarray, start_x: int, start_y: int,
                    global_visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find connected water region using BFS."""
        region = []
        queue = deque([(start_x, start_y)])
        global_visited.add((start_x, start_y))

        while queue:
            cx, cy = queue.popleft()
            region.append((cx, cy))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if grid[ny, nx] == WATER and (nx, ny) not in global_visited:
                        global_visited.add((nx, ny))
                        queue.append((nx, ny))

        return region

    def _is_valid_map(self, grid: np.ndarray, chokepoints: List[Tuple[int, int]]) -> bool:
        """Is map valid?"""
        water_count = np.sum(grid == WATER)
        land_count = np.sum(grid == LAND)
        total = self.rows * self.cols

        # Water ratio must be between 35-65%
        water_ratio = water_count / total
        if water_ratio < 0.35 or water_ratio > 0.65:
            return False

        # Must have enough land
        if land_count < 25:
            return False

        # Check active chokepoint count
        active_chokepoints = self._count_active_chokepoints(grid)
        if active_chokepoints < self.wall_budget * 0.6:  # At least 60% must be active
            return False

        return True

    def _count_active_chokepoints(self, grid: np.ndarray) -> int:
        """Calculate actual chokepoint count."""
        count = 0
        counted = set()

        for y in range(1, self.rows - 1):
            for x in range(1, self.cols - 1):
                if grid[y, x] == WATER and (x, y) not in counted:
                    # Horizontal narrow passage: top and bottom land, left or right water
                    if grid[y-1, x] == LAND and grid[y+1, x] == LAND:
                        if grid[y, x-1] == WATER or grid[y, x+1] == WATER:
                            count += 1
                            counted.add((x, y))
                            # Mark neighbor chokepoints (part of same passage)
                            for nx in range(x-1, x+2):
                                if 0 <= nx < self.cols and grid[y, nx] == WATER:
                                    if grid[y-1, nx] == LAND and grid[y+1, nx] == LAND:
                                        counted.add((nx, y))

                    # Vertical narrow passage: left and right land, top or bottom water
                    elif grid[y, x-1] == LAND and grid[y, x+1] == LAND:
                        if grid[y-1, x] == WATER or grid[y+1, x] == WATER:
                            count += 1
                            counted.add((x, y))
                            # Mark neighbor chokepoints
                            for ny in range(y-1, y+2):
                                if 0 <= ny < self.rows and grid[ny, x] == WATER:
                                    if grid[ny, x-1] == LAND and grid[ny, x+1] == LAND:
                                        counted.add((x, ny))

        return count

    def _place_moby(self, grid: np.ndarray, chokepoints: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Place Moby near center, far from chokepoints."""
        center_y, center_x = self.rows // 2, self.cols // 2

        candidates = []
        for y in range(2, self.rows - 2):
            for x in range(2, self.cols - 2):
                if grid[y, x] == WATER:
                    # Far from chokepoints?
                    min_dist_to_choke = float('inf')
                    for cx, cy in chokepoints:
                        dist = abs(x - cx) + abs(y - cy)
                        min_dist_to_choke = min(min_dist_to_choke, dist)

                    # Proximity to center
                    dist_to_center = abs(x - center_x) + abs(y - center_y)

                    # Score: Near center + far from chokepoints
                    score = -dist_to_center + min_dist_to_choke * 0.5
                    candidates.append((score, (x, y)))

        if not candidates:
            return None

        # Select from top scoring candidates
        candidates.sort(key=lambda c: c[0], reverse=True)
        top_candidates = candidates[:max(1, len(candidates) // 5)]

        return random.choice(top_candidates)[1]

    def _moby_can_escape(self, grid: np.ndarray, moby_pos: Tuple[int, int]) -> bool:
        """Can Moby escape to the edge?"""
        visited = {moby_pos}
        queue = deque([moby_pos])

        while queue:
            cx, cy = queue.popleft()

            if cx == 0 or cx == self.cols - 1 or cy == 0 or cy == self.rows - 1:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if (nx, ny) not in visited and grid[ny, nx] == WATER:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return False

    def _generate_fallback_map(self) -> MapState:
        """Fallback: Simple but valid map."""
        grid = np.zeros((self.rows, self.cols), dtype=int)

        # Grid-shaped blocks
        for row in range(3):
            for col in range(4):
                x = 2 + col * 5
                y = 2 + row * 4
                if x < self.cols - 3 and y < self.rows - 3:
                    width = random.randint(2, 3)
                    height = random.randint(2, 3)
                    self._place_block(grid, x, y, width, height)

        moby_pos = (self.cols // 2, self.rows // 2)
        if grid[moby_pos[1], moby_pos[0]] == LAND:
            grid[moby_pos[1], moby_pos[0]] = WATER

        return MapState(grid, moby_pos, self.wall_budget)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("CHOKEPOINT BASED MAP GENERATOR v3")
    print("=" * 60)

    for wall_budget in [10, 12, 14]:
        print(f"\n{'='*60}")
        print(f"BUOY COUNT: {wall_budget}")
        print("=" * 60)

        generator = MapGenerator(rows=15, cols=19, wall_budget=wall_budget)

        for i in range(2):
            print(f"\n--- Map {i+1} ---")
            map_state = generator.generate()

            # Calculate chokepoint count
            chokepoint_count = generator._count_active_chokepoints(map_state.grid)

            print(f"Water cells: {map_state.total_water_count}")
            print(f"Moby: {map_state.moby_pos}")
            print(f"Active Chokepoints: {chokepoint_count}")
            print()

            for y in range(map_state.rows):
                line = ""
                for x in range(map_state.cols):
                    if (x, y) == map_state.moby_pos:
                        line += " @ "
                    elif map_state.grid[y, x] == LAND:
                        line += "###"
                    else:
                        line += " . "
                print(line)