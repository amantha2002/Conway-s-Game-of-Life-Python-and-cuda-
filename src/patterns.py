"""
Pattern definitions and grid generator for Conway's Game of Life.
Provides standard patterns: Random, Glider, Glider Gun, R-pentomino, and Blinker.
"""

from typing import Dict, List, Tuple
import numpy as np
from src.config import PatternType, SimulationConfig

# Pre-defined pattern coordinates or bitmasks
GLIDER_COORDS = [
    (0, 1),
    (1, 2),
    (2, 0), (2, 1), (2, 2)
]

BLINKER_COORDS = [
    (0, 0), (0, 1), (0, 2)
]

R_PENTOMINO_COORDS = [
    (0, 1), (0, 2),
    (1, 0), (1, 1),
    (2, 1)
]

# Gosper's Glider Gun (36 wide, 9 high)
GOSPER_GLIDER_GUN_COORDS = [
    (4, 0), (4, 1), (5, 0), (5, 1),  # Left 2x2 square
    (2, 12), (2, 13), (3, 11), (4, 10), (5, 10), (6, 10), (7, 11), (8, 12), (8, 13),  # Left circle
    (5, 14), (3, 15), (7, 15), (4, 16), (5, 16), (6, 16), (5, 17),  # Middle node
    (2, 20), (3, 20), (4, 20), (2, 21), (3, 21), (4, 21), (1, 22), (5, 22), (0, 24), (1, 24), (5, 24), (6, 24),  # Right wings
    (2, 34), (3, 34), (2, 35), (3, 35)  # Right 2x2 square
]

PATTERN_NAMES = [
    PatternType.RANDOM.value,
    PatternType.GLIDER.value,
    PatternType.GLIDER_GUN.value,
    PatternType.R_PENTOMINO.value,
    PatternType.BLINKER.value,
]


def _stamp_coordinates(grid: np.ndarray, coords: List[Tuple[int, int]], offset_r: int, offset_c: int) -> None:
    """Stamp a list of relative (r, c) coordinates onto a grid at a given offset."""
    rows, cols = grid.shape
    for r, c in coords:
        nr = offset_r + r
        nc = offset_c + c
        if 0 <= nr < rows and 0 <= nc < cols:
            grid[nr, nc] = 1


def create_initial_grid(config: SimulationConfig) -> np.ndarray:
    """
    Generate the initial 2D numpy uint8 array based on the given configuration.
    
    Args:
        config: SimulationConfig instance
        
    Returns:
        np.ndarray: 2D array of shape (grid_size, grid_size) with 0s (dead) and 1s (alive).
    """
    size = config.grid_size
    grid = np.zeros((size, size), dtype=np.uint8)
    pattern = config.pattern

    if pattern == PatternType.RANDOM.value:
        rng = np.random.default_rng(config.random_seed)
        density = float(config.random_density)
        grid = (rng.random((size, size)) < density).astype(np.uint8)

    elif pattern == PatternType.GLIDER.value:
        # Place near the top-left or centered
        start_r = max(2, size // 4)
        start_c = max(2, size // 4)
        _stamp_coordinates(grid, GLIDER_COORDS, start_r, start_c)

    elif pattern == PatternType.GLIDER_GUN.value:
        # Gosper Glider Gun is ~36 wide x 9 high
        # Place near the upper-left with enough space to produce gliders
        start_r = max(5, size // 6)
        start_c = max(5, size // 8)
        _stamp_coordinates(grid, GOSPER_GLIDER_GUN_COORDS, start_r, start_c)

    elif pattern == PatternType.R_PENTOMINO.value:
        # Center R-pentomino
        start_r = size // 2 - 1
        start_c = size // 2 - 1
        _stamp_coordinates(grid, R_PENTOMINO_COORDS, start_r, start_c)

    elif pattern == PatternType.BLINKER.value:
        # Center Blinker
        start_r = size // 2
        start_c = size // 2 - 1
        _stamp_coordinates(grid, BLINKER_COORDS, start_r, start_c)

    else:
        # Default fallback to random
        rng = np.random.default_rng(config.random_seed)
        grid = (rng.random((size, size)) < config.random_density).astype(np.uint8)

    return grid
