"""
Benchmark module comparing CPU vs CUDA performance for Conway's Game of Life.
"""

import platform
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from src.config import SimulationConfig, RunMode
from src.patterns import create_initial_grid
from src.cpu_engine import CPUEngine
from src.cuda_engine import CUDAEngine


def get_hardware_info() -> dict:
    """
    Collect hardware/software information for the benchmark report.
    Returns a dict with cpu_name, gpu_name, cuda_version keys.
    """
    cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
    # Try to get a friendlier CPU name on macOS / Linux
    try:
        import subprocess
        if sys.platform == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if out:
                cpu_name = out
        elif sys.platform == "linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    gpu_info = CUDAEngine.get_gpu_info()

    return {
        "cpu_name": cpu_name,
        "gpu_name": gpu_info.get("gpu_name", "N/A"),
        "cuda_version": gpu_info.get("cuda_version", "N/A"),
        "vram_bytes": gpu_info.get("vram_bytes", 0),
    }


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

    # --- Extended stats for the Results Window ---
    # Per-step timing lists (for dynamic graph)
    per_step_times_cpu: List[float] = field(default_factory=list)
    per_step_times_gpu: List[float] = field(default_factory=list)

    # Derived stats (populated after simulation completes)
    avg_ms_per_gen_cpu: Optional[float] = None
    avg_ms_per_gen_gpu: Optional[float] = None
    min_ms_per_gen_cpu: Optional[float] = None
    min_ms_per_gen_gpu: Optional[float] = None
    max_ms_per_gen_cpu: Optional[float] = None
    max_ms_per_gen_gpu: Optional[float] = None

    throughput_cpu: Optional[float] = None   # cells/sec
    throughput_gpu: Optional[float] = None   # cells/sec

    gpu_transfer_time_sec: Optional[float] = None  # total_time - kernel_time
    memory_bytes_cpu: Optional[int] = None
    memory_bytes_gpu: Optional[int] = None

    final_alive_cpu: Optional[int] = None
    final_alive_gpu: Optional[int] = None

    # Hardware info
    cpu_name: str = ""
    gpu_name: str = ""
    cuda_version: str = ""

    def compute_derived_stats(self):
        """Calculate derived statistics from raw timing data."""
        total_cells = self.grid_size * self.grid_size

        if self.per_step_times_cpu:
            self.avg_ms_per_gen_cpu = sum(self.per_step_times_cpu) / len(self.per_step_times_cpu)
            self.min_ms_per_gen_cpu = min(self.per_step_times_cpu)
            self.max_ms_per_gen_cpu = max(self.per_step_times_cpu)
            self.cpu_time_sec = sum(self.per_step_times_cpu) / 1000.0
            self.cpu_steps_per_sec = len(self.per_step_times_cpu) / self.cpu_time_sec if self.cpu_time_sec > 0 else float("inf")
            # Throughput: cells updated per second  = (total_cells * generations) / total_time
            self.throughput_cpu = (total_cells * len(self.per_step_times_cpu)) / self.cpu_time_sec if self.cpu_time_sec > 0 else 0

        if self.per_step_times_gpu:
            self.avg_ms_per_gen_gpu = sum(self.per_step_times_gpu) / len(self.per_step_times_gpu)
            self.min_ms_per_gen_gpu = min(self.per_step_times_gpu)
            self.max_ms_per_gen_gpu = max(self.per_step_times_gpu)
            self.cuda_kernel_time_sec = sum(self.per_step_times_gpu) / 1000.0
            self.cuda_steps_per_sec = len(self.per_step_times_gpu) / self.cuda_kernel_time_sec if self.cuda_kernel_time_sec > 0 else float("inf")
            self.throughput_gpu = (total_cells * len(self.per_step_times_gpu)) / self.cuda_kernel_time_sec if self.cuda_kernel_time_sec > 0 else 0

        # GPU transfer overhead
        if self.cuda_total_time_sec is not None and self.cuda_kernel_time_sec is not None:
            self.gpu_transfer_time_sec = max(0, self.cuda_total_time_sec - self.cuda_kernel_time_sec)

        # Speedups
        if self.cpu_time_sec and self.cuda_kernel_time_sec and self.cuda_kernel_time_sec > 0:
            self.kernel_speedup = self.cpu_time_sec / self.cuda_kernel_time_sec
        if self.cpu_time_sec and self.cuda_total_time_sec and self.cuda_total_time_sec > 0:
            self.total_speedup = self.cpu_time_sec / self.cuda_total_time_sec

        # Memory estimates (uint8 grid)
        self.memory_bytes_cpu = total_cells  # 1 byte per cell (uint8)
        self.memory_bytes_gpu = total_cells  # same grid size on GPU

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
