"""
Map analysis functions.
"""

from collections import deque
from typing import List, Tuple
import random
import numpy as np

from constants import LAND, BUOY
from models import EscapeAnalysis


def place_walls(grid: np.ndarray, walls: List[Tuple[int, int]]) -> np.ndarray:
    """
    Places walls at the given coordinates.

    Args:
        grid: Original map matrix
        walls: List of wall coordinates [(x, y), ...]

    Returns:
        New map matrix with added walls
    """
    new_grid = grid.copy()
    for x, y in walls:
        new_grid[y, x] = BUOY
    return new_grid


def flood_fill(grid: np.ndarray, start: Tuple[int, int]) -> Tuple[int, bool]:
    """
    BFS calculates the storage space Moby can access.
    
    Args:
        grid: Map matrix
        start: Starting position (x, y)
    
    Returns:
        (area_size, escape_can) tuple
    """
    rows, cols = grid.shape
    x, y = start

    if grid[y, x] in (LAND, BUOY):
        return 0, False

    visited = {start}
    queue = deque([start])
    escaped = False

    while queue:
        cx, cy = queue.popleft()
        
        # If it reached the edges, it has escaped.
        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            escaped = True

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in visited and grid[ny, nx] not in (LAND, BUOY):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return len(visited), escaped


def analyze_escape(grid: np.ndarray, start: Tuple[int, int]) -> EscapeAnalysis:
    """
    It calculates the area and returns the number of escape points (frontiers).

    Args:
        grid: Map matrix
        start: Starting position (x, y)

    Returns:
        EscapeAnalysis object
    """
    rows, cols = grid.shape

    x, y = start
    if grid[y, x] in (LAND, BUOY):
        return EscapeAnalysis(area=0, frontier_count=0, escaped=False)

    visited = {start}
    queue = deque([start])
    escaped = False
    boundary_cells = set()

    while queue:
        cx, cy = queue.popleft()

        # Has it reached the edge?
        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            escaped = True
            boundary_cells.add((cx, cy))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] not in (LAND, BUOY) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return EscapeAnalysis(
        area=len(visited),
        frontier_count=len(boundary_cells),
        escaped=escaped
    )


def get_stochastic_escape_path(grid: np.ndarray, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    It finds an escape path using a randomized BFS.
    It can find a different shortest path each time it is called.

    Args:
        grid: Map matrix
        start: Starting position (x, y)

    Returns:
        List of escape path coordinates (empty list = no escape path)
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        # If Moby reached the edge, turn back.
        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            return path + [(cx, cy)]

        # Mix up the directions (stochastic behavior)
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


# --- analysis.py dosyasının içine eklenecek ---

def get_shortest_escape_path_length(grid: np.ndarray, start: Tuple[int, int]) -> int:
    """
    Calculates the minimum number of steps Moby needs to take to reach the map boundaries.

    Args:
        grid: Map matrix
        start: Starting position (x, y)

    Returns:
        int: Number of steps (If trapped, it returns to infinity)
    """
    rows, cols = grid.shape

    # If it's already on the edge
    if start[0] == 0 or start[0] == cols - 1 or start[1] == 0 or start[1] == rows - 1:
        return 0

    queue = deque([(start, 0)])  # (position, distance)
    visited = {start}

    while queue:
        (cx, cy), dist = queue.popleft()

        # Has it reached the edge?
        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            return dist

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy

            if 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in visited and grid[ny, nx] not in (LAND, BUOY):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

    return float('inf')


# ============================================================
# Wall-set (no-grid-copy) analysis helpers for fast solvers
# ============================================================

from typing import Set, Optional, Iterable, Dict as _Dict, Any as _Any

def analyze_escape_walls(
    grid: np.ndarray,
    start: Tuple[int, int],
    walls_set: Set[Tuple[int, int]]
) -> EscapeAnalysis:
    """Fast escape/area analysis without copying the grid.

    Treats LAND in grid as blocked; BUOYs are provided via walls_set.
    """
    rows, cols = grid.shape
    sx, sy = start
    if (sx, sy) in walls_set or grid[sy, sx] in (LAND, BUOY):

        return EscapeAnalysis(area=0, frontier_count=0, escaped=False)

    visited = {start}
    queue = deque([start])
    escaped = False
    boundary_cells = set()

    while queue:
        cx, cy = queue.popleft()

        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            escaped = True
            boundary_cells.add((cx, cy))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (
                        (nx, ny) not in visited
                        and (nx, ny) not in walls_set
                        and grid[ny, nx] not in (LAND, BUOY)
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return EscapeAnalysis(area=len(visited), frontier_count=len(boundary_cells), escaped=escaped)


def analyze_escape_detailed_walls(
    grid: np.ndarray,
    start: Tuple[int, int],
    walls_set: Set[Tuple[int, int]]
) -> Tuple[EscapeAnalysis, Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Like analyze_escape_walls, but also returns (visited, boundary_cells)."""
    rows, cols = grid.shape
    sx, sy = start
    if (sx, sy) in walls_set or grid[sy, sx] in (LAND, BUOY):
        analysis = EscapeAnalysis(area=0, frontier_count=0, escaped=False)
        return analysis, set(), set()

    visited = {start}
    queue = deque([start])
    escaped = False
    boundary_cells = set()

    while queue:
        cx, cy = queue.popleft()

        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            escaped = True
            boundary_cells.add((cx, cy))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (
                        (nx, ny) not in visited
                        and (nx, ny) not in walls_set
                        and grid[ny, nx] not in (LAND, BUOY)
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    analysis = EscapeAnalysis(area=len(visited), frontier_count=len(boundary_cells), escaped=escaped)
    return analysis, visited, boundary_cells


def get_stochastic_escape_path_walls(
    grid: np.ndarray,
    start: Tuple[int, int],
    walls_set: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Randomized BFS escape path finder without copying the grid."""
    rows, cols = grid.shape
    if start in walls_set or grid[start[1], start[0]] in (LAND, BUOY):

        return []

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        if cx == 0 or cx == cols - 1 or cy == 0 or cy == rows - 1:
            return path + [(cx, cy)]

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (
                        (nx, ny) not in visited
                        and (nx, ny) not in walls_set
                        and grid[ny, nx] not in (LAND, BUOY)
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(cx, cy)]))

    return []

