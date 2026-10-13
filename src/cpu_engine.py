"""
Pure CPU Implementation of Conway's Game of Life.
Supports both Wrap-around (toroidal) and Dead edges (zero-padding) boundary conditions.
Includes high-performance vectorized NumPy execution and per-step stepping.
"""

import time
from typing import Tuple, Optional
import numpy as np
from src.config import EdgeBehavior


class CPUEngine:
    """CPU-based Game of Life simulation engine."""

    def __init__(self, edge_behavior: str = EdgeBehavior.WRAP_AROUND.value):
        self.edge_behavior = edge_behavior

    def count_neighbors_wrap(self, grid: np.ndarray) -> np.ndarray:
        """Count 8-neighbors with periodic (wrap-around / toroidal) boundary conditions."""
        return (
            np.roll(grid, 1, axis=0) +
            np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) +
            np.roll(grid, -1, axis=1) +
            np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +
            np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
        )

    def count_neighbors_dead_edges(self, grid: np.ndarray) -> np.ndarray:
        """Count 8-neighbors with dead edges (zero-padded boundaries)."""
        padded = np.pad(grid, 1, mode="constant", constant_values=0)
        return (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2]                    + padded[1:-1, 2:] +
            padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
        )

    def step(self, grid: np.ndarray) -> np.ndarray:
        """
        Advance the simulation by 1 generation on CPU.
        
        Rules:
        - Live cell with 2 or 3 live neighbors survives.
        - Dead cell with exactly 3 live neighbors becomes live.
        - All other cells die or stay dead.
        """
        if self.edge_behavior == EdgeBehavior.WRAP_AROUND.value:
            neighbors = self.count_neighbors_wrap(grid)
        else:
            neighbors = self.count_neighbors_dead_edges(grid)

        # Vectorized Conway's rule:
        # alive in next gen if: (current == 1 and (neighbors == 2 or 3)) or (current == 0 and neighbors == 3)
        next_grid = np.zeros_like(grid, dtype=np.uint8)
        survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
        born = (grid == 0) & (neighbors == 3)
        next_grid[survive | born] = 1

        return next_grid

    def run(self, initial_grid: np.ndarray, iterations: int) -> Tuple[np.ndarray, float]:
        """
        Run simulation for given number of iterations.
        
        Args:
            initial_grid: Starting 2D uint8 grid
            iterations: Number of generations to advance
            
        Returns:
            Tuple[np.ndarray, float]: (final_grid, elapsed_time_seconds)
        """
        current_grid = initial_grid.copy()
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            current_grid = self.step(current_grid)
        elapsed_time = time.perf_counter() - start_time

        return current_grid, elapsed_time

    def run_stepwise(self, initial_grid: np.ndarray, iterations: Optional[int] = None):
        """
        Generator that yields timing data after each generation step.
        If iterations is None or <= 0, runs indefinitely until stopped.

        Yields:
            Tuple[int, np.ndarray, float]: (generation_number, grid_snapshot, step_time_ms)
        """
        current_grid = initial_grid.copy()
        gen = 0
        while True:
            gen += 1
            if iterations is not None and iterations > 0 and gen > iterations:
                break
            step_start = time.perf_counter()
            current_grid = self.step(current_grid)
            step_ms = (time.perf_counter() - step_start) * 1000.0
            yield gen, current_grid, step_ms
