"""
Benchmark module comparing CPU vs CUDA performance for Conway's Game of Life.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.config import SimulationConfig, RunMode
from src.patterns import create_initial_grid
from src.cpu_engine import CPUEngine
from src.cuda_engine import CUDAEngine


@dataclass
class BenchmarkResult:
    """Stores benchmark metrics and comparison data."""
    grid_size: int
    iterations: int
    edge_behavior: str
    block_size_str: str
    
    cpu_time_sec: Optional[float] = None
    cpu_steps_per_sec: Optional[float] = None
    
    cuda_kernel_time_sec: Optional[float] = None
    cuda_total_time_sec: Optional[float] = None
    cuda_steps_per_sec: Optional[float] = None
    
    kernel_speedup: Optional[float] = None
    total_speedup: Optional[float] = None
    grids_match: Optional[bool] = None
    error_message: Optional[str] = None

    def summary_table(self) -> str:
        """Format results as a readable markdown-like table."""
        lines = [
            f"=== Benchmark Summary: Grid {self.grid_size}x{self.grid_size}, {self.iterations} Iterations ===",
            f"Edge Behavior: {self.edge_behavior} | CUDA Block Size: {self.block_size_str}",
            "-" * 65,
        ]
        if self.cpu_time_sec is not None:
            lines.append(f"CPU Time:            {self.cpu_time_sec * 1000:8.2f} ms ({self.cpu_steps_per_sec:8.1f} steps/s)")
        if self.cuda_kernel_time_sec is not None:
            lines.append(f"CUDA Kernel Time:    {self.cuda_kernel_time_sec * 1000:8.2f} ms ({self.cuda_steps_per_sec:8.1f} steps/s)")
            lines.append(f"CUDA Total Time:     {self.cuda_total_time_sec * 1000:8.2f} ms (incl. HtoD + DtoH)")
        if self.kernel_speedup is not None:
            lines.append(f"Speedup (Kernel):    {self.kernel_speedup:8.2f}x faster on GPU")
        if self.total_speedup is not None:
            lines.append(f"Speedup (Total):     {self.total_speedup:8.2f}x faster on GPU")
        if self.grids_match is not None:
            status = "MATCH (Identical output)" if self.grids_match else "MISMATCH"
            lines.append(f"Result Verification: {status}")
        if self.error_message:
            lines.append(f"Notice: {self.error_message}")
        lines.append("-" * 65)
        return "\n".join(lines)


def run_benchmark(config: SimulationConfig) -> BenchmarkResult:
    """
    Execute CPU and/or CUDA benchmark based on configuration.
    
    Args:
        config: SimulationConfig instance
        
    Returns:
        BenchmarkResult with performance metrics
    """
    initial_grid = create_initial_grid(config)
    result = BenchmarkResult(
        grid_size=config.grid_size,
        iterations=config.iterations,
        edge_behavior=config.edge_behavior,
        block_size_str=config.block_size_str,
    )

    cpu_final_grid = None
    gpu_final_grid = None

    # Run CPU if requested
    if config.run_mode in (RunMode.BOTH.value, RunMode.CPU_ONLY.value):
        cpu_engine = CPUEngine(edge_behavior=config.edge_behavior)
        cpu_final_grid, cpu_time = cpu_engine.run(initial_grid, config.iterations)
        result.cpu_time_sec = cpu_time
        result.cpu_steps_per_sec = config.iterations / cpu_time if cpu_time > 0 else float("inf")

    # Run GPU if requested
    if config.run_mode in (RunMode.BOTH.value, RunMode.GPU_ONLY.value):
        if not CUDAEngine.is_available():
            result.error_message = "CUDA is not available on this device; GPU benchmark was skipped."
        else:
            try:
                gpu_engine = CUDAEngine(
                    edge_behavior=config.edge_behavior,
                    block_size=config.cuda_block_size
                )
                gpu_final_grid, kernel_time, total_time = gpu_engine.run(initial_grid, config.iterations)
                result.cuda_kernel_time_sec = kernel_time
                result.cuda_total_time_sec = total_time
                result.cuda_steps_per_sec = config.iterations / kernel_time if kernel_time > 0 else float("inf")
            except Exception as e:
                result.error_message = f"GPU execution error: {e}"

    # Calculate speedups and verify match if both ran
    if result.cpu_time_sec is not None and result.cuda_kernel_time_sec is not None:
        if result.cuda_kernel_time_sec > 0:
            result.kernel_speedup = result.cpu_time_sec / result.cuda_kernel_time_sec
        if result.cuda_total_time_sec and result.cuda_total_time_sec > 0:
            result.total_speedup = result.cpu_time_sec / result.cuda_total_time_sec

    if cpu_final_grid is not None and gpu_final_grid is not None:
        result.grids_match = bool(np.array_equal(cpu_final_grid, gpu_final_grid))

    return result
