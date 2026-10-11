"""
Conway's Game of Life - High Performance Computing (CPU vs CUDA)
Modular package containing simulation engines, pattern generators, UI, and benchmarking tools.
"""

from src.config import SimulationConfig, PatternType, EdgeBehavior, RunMode, DisplayMode
from src.patterns import create_initial_grid, PATTERN_NAMES
from src.cpu_engine import CPUEngine
from src.cuda_engine import CUDAEngine
from src.benchmark import run_benchmark, BenchmarkResult

__all__ = [
    "SimulationConfig",
    "PatternType",
    "EdgeBehavior",
    "RunMode",
    "DisplayMode",
    "create_initial_grid",
    "PATTERN_NAMES",
    "CPUEngine",
    "CUDAEngine",
    "run_benchmark",
    "BenchmarkResult",
]
