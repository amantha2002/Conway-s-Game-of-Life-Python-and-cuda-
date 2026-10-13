"""
Configuration dataclasses and constants for Conway's Game of Life simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class PatternType(str, Enum):
    RANDOM = "Random"
    GLIDER = "Glider"
    GLIDER_GUN = "Glider Gun"
    R_PENTOMINO = "R-pentomino"
    BLINKER = "Blinker"


class EdgeBehavior(str, Enum):
    WRAP_AROUND = "Wrap-around"
    DEAD_EDGES = "Dead edges"


class RunMode(str, Enum):
    BOTH = "Both"
    CPU_ONLY = "CPU only"
    GPU_ONLY = "GPU only"


class DisplayMode(str, Enum):
    LIVE_VIEW = "Live view"
    BENCHMARK = "Benchmark (no drawing)"


AVAILABLE_GRID_SIZES = [64, 128, 256, 512, 1024]
AVAILABLE_BLOCK_SIZES = [(8, 8), (16, 16), (32, 32)]
BLOCK_SIZE_STRINGS = ["8x8", "16x16", "32x32"]


def parse_block_size(val: str) -> Tuple[int, int]:
    """Parse block size representation like '16x16' or '16×16' to tuple (w, h).
    Accepts only values defined in AVAILABLE_BLOCK_SIZES.
    """
    val = str(val).replace("×", "x").strip().lower()
    allowed = {f"{w}x{h}" for w, h in AVAILABLE_BLOCK_SIZES}
    if val in allowed:
        w, h = map(int, val.split("x"))
        return (w, h)
    raise ValueError(f"Invalid block size '{val}'. Must be one of {sorted(allowed)}")


@dataclass
class SimulationConfig:
    """Holds all simulation parameters and runtime settings."""
    grid_size: int = 128
    iterations: int = 100
    is_unlimited: bool = False
    pattern: str = PatternType.RANDOM.value
    random_density: float = 0.25  # 10% to 50%
    random_seed: int = 42
    edge_behavior: str = EdgeBehavior.WRAP_AROUND.value
    run_mode: str = RunMode.BOTH.value
    display_mode: str = DisplayMode.LIVE_VIEW.value
    cell_size: int = 4
    cuda_block_size: Tuple[int, int] = (16, 16)

    def validate(self) -> None:
        """Validate configuration values against constraints."""
        if self.grid_size not in AVAILABLE_GRID_SIZES:
            raise ValueError(f"grid_size must be one of {AVAILABLE_GRID_SIZES}, got {self.grid_size}")
        if not self.is_unlimited and self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if not (0.05 <= self.random_density <= 0.90):
            raise ValueError(f"random_density should be between 0.05 and 0.90, got {self.random_density}")
        if self.cell_size < 1:
            raise ValueError(f"cell_size must be >= 1, got {self.cell_size}")
        if self.cuda_block_size not in AVAILABLE_BLOCK_SIZES:
            raise ValueError(f"cuda_block_size must be one of {AVAILABLE_BLOCK_SIZES}, got {self.cuda_block_size}")

    @property
    def block_size_str(self) -> str:
        return f"{self.cuda_block_size[0]}x{self.cuda_block_size[1]}"
