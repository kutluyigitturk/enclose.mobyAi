"""
Data models and basic structures.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set
import numpy as np

from constants import WATER, MOBY


@dataclass
class EscapeAnalysis:
    """A data class that holds the results of escape analysis."""
    area: int
    frontier_count: int
    escaped: bool


class MapState:
    """
    Represents the state of the game map.

    Attributes:
        grid: Map matrix (numpy array)
        rows: Number of rows
        cols: Number of columns
        moby_pos: Moby's position (x, y)
        max_walls: Maximum number of walls
        water_cells: List of water cells (for sequential access)
        water_cells_set: Set of water cells (for quick lookup)
    """
    
    def __init__(self, grid: np.ndarray, moby_pos: Tuple[int, int], max_walls: int):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.moby_pos = moby_pos
        self.max_walls = max_walls
        
        # Keep the water cells both as a list and as a set
        self.water_cells: List[Tuple[int, int]] = []
        self.water_cells_set: Set[Tuple[int, int]] = set()
        
        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y, x] == WATER and (x, y) != moby_pos:
                    self.water_cells.append((x, y))
                    self.water_cells_set.add((x, y))
    
    @property
    def total_water_count(self) -> int:
        """Total number of water cells."""
        return len(self.water_cells)
    
    def is_water(self, pos: Tuple[int, int]) -> bool:
        """Is the given position water? O(1) lookup."""
        return pos in self.water_cells_set
    
    def __repr__(self) -> str:
        return f"MapState({self.rows}x{self.cols}, moby={self.moby_pos}, walls={self.max_walls})"
