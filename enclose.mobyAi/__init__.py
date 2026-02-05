"""
Moby Dick Puzzle Solver

Genetic algorithm-based level solver.
"""

from constants import WATER, LAND, BUOY, MOBY
from models import MapState, EscapeAnalysis
from analysis import flood_fill, analyze_escape, place_walls, get_stochastic_escape_path
from genetic_solver import GeneticSolver
from blind_solver import BlindSolver
from visualization import visualize_grid, compare_solutions

__all__ = [
    # Constants
    'WATER', 'LAND', 'BUOY', 'MOBY',
    
    # Models
    'MapState', 'EscapeAnalysis',
    
    # Analysis
    'flood_fill', 'analyze_escape', 'place_walls', 'get_stochastic_escape_path',
    
    # Solvers
    'GeneticSolver', 'BlindSolver',
    
    # Visualization
    'visualize_grid', 'compare_solutions'
]

__version__ = '0.1.0'
