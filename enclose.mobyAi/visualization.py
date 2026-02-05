"""
Visualization and comparison functions.
"""

from typing import List, Tuple, Set
import numpy as np

from constants import (
    WATER, LAND, BUOY, MOBY,
    PATHFINDER_MAP, PATHFINDER_MOBY_POS, PATHFINDER_OPTIMAL_SOLUTION
)


def visualize_grid(
    walls: List[Tuple[int, int]],
    title: str,
    grid_data: List[List[int]] = None,
    moby_pos: Tuple[int, int] = None
) -> None:
    """
    Visualizes the map as ASCII.

    Args:
        walls: Wall coordinates
        title: Title
        grid_data: Map data (if None, PATHFINDER_MAP is used)
        moby_pos: Moby position (if None, PATHFINDER_MOBY_POS is used)
    """
    if grid_data is None:
        grid_data = PATHFINDER_MAP
    if moby_pos is None:
        moby_pos = PATHFINDER_MOBY_POS

    grid = np.array(grid_data)
    rows, cols = grid.shape

    # Place walls
    for x, y in walls:
        grid[y, x] = BUOY

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
            else:  # WATER
                char = " . "
            line += char
        print(line)


def compare_solutions(
    found_walls: List[Tuple[int, int]],
    found_score: int,
    optimal_walls: List[Tuple[int, int]] = None,
    optimal_score: int = None
) -> None:
    """
    Compares the found solution with the optimal solution.

    Args:
        found_walls: Found wall coordinates
        found_score: Found score
        optimal_walls: Optimal wall coordinates (if None, PATHFINDER_OPTIMAL_SOLUTION is used)
        optimal_score: Optimal score (if None, 68 is used)
    """
    if optimal_walls is None:
        optimal_walls = PATHFINDER_OPTIMAL_SOLUTION
    if optimal_score is None:
        optimal_score = 68

    # Visualize
    visualize_grid(found_walls, f"FOUND SOLUTION ({found_score} Points)")
    visualize_grid(optimal_walls, f"OPTIMAL SOLUTION ({optimal_score} Points)")

    # Comparison analysis
    set_found = set(found_walls)
    set_optimal = set(optimal_walls)

    common = set_found.intersection(set_optimal)
    extra = set_found - set_optimal
    missing = set_optimal - set_found

    print("\n" + "=" * 40)
    print("COMPARISON ANALYSIS")
    print("=" * 40)
    print(f"Common Walls ({len(common)}): {common}")
    print(f"Extra Placed ({len(extra)}): {extra}")
    print(f"Missing ({len(missing)}): {missing}")

    # Success evaluation
    accuracy = len(common) / len(optimal_walls) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")

    if found_score >= optimal_score:
        print("✅ OPTIMAL OR BETTER!")
    elif found_score >= optimal_score * 0.95:
        print("🟡 Very close to optimal (95%+)")
    elif found_score >= optimal_score * 0.90:
        print("🟠 Good result (90%+)")
    else:
        print(f"🔴 Needs improvement (Target: {optimal_score})")


def print_solution_summary(result: dict) -> None:
    """
    Prints the solution summary.

    Args:
        result: Solver result
    """
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Found Area: {result.get('area', result.get('optimal_area', 0))}")
    print(f"Walls: {result.get('walls', result.get('solution', []))}")
    print(f"Attempt Count: {result.get('attempts', 'N/A')}")

    if 'converged' in result:
        status = "Saturation reached" if result['converged'] else "Max attempts reached"
        print(f"Status: {status}")