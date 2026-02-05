#!/usr/bin/env python3
"""
Moby Dick Puzzle Solver - Main Execution File

Usage:
    python main.py                      # Solve the default Pathfinder map
    python main.py --generate           # NEW: Generate and solve a random map
    python main.py --generate --rows 20 # Generate with specified dimensions
"""

import argparse
import numpy as np

from constants import (
    PATHFINDER_MAP,
    PATHFINDER_MOBY_POS,
    PATHFINDER_MAX_WALLS,
    PATHFINDER_OPTIMAL_AREA
)
from models import MapState
from blind_solver import BlindSolver
from visualization import compare_solutions, print_solution_summary, visualize_grid
from generator import MapGenerator  # NEW MODULE


def create_pathfinder_map_state() -> MapState:
    """Creates MapState for the fixed Pathfinder map."""
    grid = np.array(PATHFINDER_MAP)
    return MapState(grid, PATHFINDER_MOBY_POS, PATHFINDER_MAX_WALLS)


def run_generated_mode(rows: int, cols: int, walls: int, density: float, verbose: bool):
    """Runs the generator mode."""
    print(f"\n🌍 GENERATING MAP ({rows}x{cols})...")

    # 1. Generate Map
    generator = MapGenerator(rows=rows, cols=cols, wall_budget=walls)
    map_state = generator.generate_solvable()

    # --- UPDATED SECTION: EMBED MOBY INTO ARRAY ---
    print("\n" + "=" * 50)
    print("📋 GAME READY DATA (Copy & Paste)")
    print("=" * 50)

    # Create a temporary copy for printing
    display_grid = map_state.grid.copy()
    mx, my = map_state.moby_pos
    display_grid[my, mx] = 3  # 3 = MOBY (Embedding into the array)

    print("GENERATED_MAP = [")
    for row in display_grid.tolist():
        print(f"    {row},")
    print("]")
    print(f"# Moby (3) placed into the grid at position {map_state.moby_pos}.")
    print("=" * 50 + "\n")
    # -----------------------------------------------------------

    # Show generated map as ASCII
    visualize_grid([], "GENERATED MAP (VISUAL)", map_state.grid, map_state.moby_pos)

    if verbose:
        print(f"Water cell count: {map_state.total_water_count}")
        print(f"Max wall budget: {map_state.max_walls}")

    # 2. Solve
    print("\n🧠 SOLVER ENGAGING...")
    run_solver_logic(map_state, verbose=verbose, show_comparison=False)


def run_solver_logic(
        map_state: MapState,
        patience: int = 15,
        max_attempts: int = 50,
        target_area: int = None,
        verbose: bool = True,
        show_comparison: bool = True
) -> dict:
    """Runs the solver for a given map state."""

    solver = BlindSolver(map_state)
    result = solver.find_optimum(
        patience=patience,
        max_attempts=max_attempts,
        target_area=target_area,
        verbose=verbose
    )

    if verbose:
        print_solution_summary(result)

    if show_comparison and result['walls']:
        # Compare only on known maps like Pathfinder
        compare_solutions(result['walls'], result['area'])
    elif result['walls']:
        # Only draw result in Generate mode
        visualize_grid(result['walls'], f"SOLUTION FOUND ({result['area']} Points)", map_state.grid, map_state.moby_pos)

    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Moby Dick Puzzle Solver & Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Mode Selection ---
    parser.add_argument('--generate', '-g', action='store_true', help='Random map generation mode')

    # --- Solver Settings ---
    parser.add_argument('--patience', '-p', type=int, default=15, help='Patience parameter')
    parser.add_argument('--max-attempts', '-m', type=int, default=50, help='Max attempts')
    parser.add_argument('--target', '-t', type=int, default=None, help='Target area (Not used in Generate mode)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')

    # --- Generator Settings ---
    parser.add_argument('--rows', type=int, default=22, help='Map row count')
    parser.add_argument('--cols', type=int, default=19, help='Map column count')
    parser.add_argument('--walls', type=int, default=12, help='Wall budget')
    # Add to "Generator Settings" section in main.py:
    parser.add_argument('--density', type=float, default=0.55, help='Land density (between 0.0 - 1.0)')

    args = parser.parse_args()

    if args.generate:
        # GENERATOR MODE
        run_generated_mode(args.rows, args.cols, args.walls, args.density, not args.quiet)
    else:
        # CLASSIC PATHFINDER MODE
        map_state = create_pathfinder_map_state()
        result = run_solver_logic(
            map_state,
            patience=args.patience,
            max_attempts=args.max_attempts,
            target_area=args.target,
            verbose=not args.quiet,
            show_comparison=True
        )
        # Exit code
        if result['area'] >= PATHFINDER_OPTIMAL_AREA:
            exit(0)
        else:
            exit(1)


if __name__ == "__main__":
    main()