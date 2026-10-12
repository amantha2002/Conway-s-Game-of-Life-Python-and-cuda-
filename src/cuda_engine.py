"""
CUDA Implementation of Conway's Game of Life using Numba CUDA kernels.
Supports configurable 2D block dimensions (8x8, 16x16, 32x32),
wrap-around vs dead-edges boundary conditions, and device-memory double buffering.
"""

import time
import math
from typing import Tuple, Optional
import numpy as np
from src.config import EdgeBehavior, AVAILABLE_BLOCK_SIZES

# Flag to verify CUDA availability
_CUDA_READY = False
try:
    from numba import cuda
    if cuda.is_available():
        _CUDA_READY = True
except Exception:
    _CUDA_READY = False


# Numba CUDA Kernels
def _create_kernels():
    """Define CUDA kernels dynamically or statically if cuda is available."""
    if not _CUDA_READY:
        return None, None

    @cuda.jit
    def life_kernel_wrap(current, nxt, width, height):
        x, y = cuda.grid(2)
        if x < width and y < height:
            neighbors = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx + width) % width
                    ny = (y + dy + height) % height
                    neighbors += current[nx, ny]

            cell = current[x, y]
            if cell == 1:
                nxt[x, y] = 1 if (neighbors == 2 or neighbors == 3) else 0
            else:
                nxt[x, y] = 1 if neighbors == 3 else 0

    @cuda.jit
    def life_kernel_dead_edges(current, nxt, width, height):
        x, y = cuda.grid(2)
        if x < width and y < height:
            neighbors = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors += current[nx, ny]

            cell = current[x, y]
            if cell == 1:
                nxt[x, y] = 1 if (neighbors == 2 or neighbors == 3) else 0
            else:
                nxt[x, y] = 1 if neighbors == 3 else 0

    return life_kernel_wrap, life_kernel_dead_edges


# Precompile/cache kernels when CUDA is ready
_kernel_wrap, _kernel_dead_edges = _create_kernels()


class CUDAEngine:
    """CUDA GPU simulation engine for Conway's Game of Life."""

    def __init__(
        self,
        edge_behavior: str = EdgeBehavior.WRAP_AROUND.value,
        block_size: Tuple[int, int] = (16, 16)
    ):
        self.edge_behavior = edge_behavior
        self.block_size = block_size
        self._check_cuda()

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA runtime and device are available."""
        global _CUDA_READY
        try:
            from numba import cuda
            return cuda.is_available()
        except Exception:
            return False

    def _check_cuda(self):
        global _CUDA_READY, _kernel_wrap, _kernel_dead_edges
        if not _CUDA_READY:
            try:
                from numba import cuda
                if cuda.is_available():
                    _CUDA_READY = True
                    _kernel_wrap, _kernel_dead_edges = _create_kernels()
            except Exception:
                pass

    def run(
        self,
        initial_grid: np.ndarray,
        iterations: int
    ) -> Tuple[np.ndarray, float, float]:
        """
        Run simulation on CUDA GPU.
        
        Args:
            initial_grid: Starting 2D uint8 numpy array
            iterations: Number of generations to advance
            
        Returns:
            Tuple[np.ndarray, float, float]:
                - final_grid: Result numpy array
                - kernel_time: Time spent in GPU compute (excluding memory transfers)
                - total_time: Total time including HtoD and DtoH memory copies
        """
        self._check_cuda()
        if not _CUDA_READY:
            raise RuntimeError(
                "CUDA is not available on this system. Please check NVIDIA driver & cudatoolkit."
            )

        width, height = initial_grid.shape
        bw, bh = self.block_size
        threads_per_block = (bw, bh)
        blocks_per_grid_x = math.ceil(width / bw)
        blocks_per_grid_y = math.ceil(height / bh)
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        kernel = _kernel_wrap if self.edge_behavior == EdgeBehavior.WRAP_AROUND.value else _kernel_dead_edges

        # Warm-up compile if needed
        # Track Total Time (Host-to-Device + Execution + Device-to-Host)
        total_start = time.perf_counter()

        # Host-to-Device transfer
        d_current = cuda.to_device(np.ascontiguousarray(initial_grid, dtype=np.uint8))
        d_next = cuda.device_array_like(d_current)

        cuda.synchronize()
        kernel_start = time.perf_counter()

        # Double-buffering loop on GPU device memory
        for _ in range(iterations):
            kernel[blocks_per_grid, threads_per_block](d_current, d_next, width, height)
            d_current, d_next = d_next, d_current

        cuda.synchronize()
        kernel_time = time.perf_counter() - kernel_start

        # Device-to-Host transfer
        final_grid = d_current.copy_to_host()
        cuda.synchronize()
        total_time = time.perf_counter() - total_start

        return final_grid, kernel_time, total_time

    def step(self, grid: np.ndarray) -> np.ndarray:
        """Advance single step on GPU (useful for live view / debug)."""
        final_grid, _, _ = self.run(grid, iterations=1)
        return final_grid

    def run_stepwise(self, initial_grid: np.ndarray, iterations: int):
        """
        Generator that yields timing data after each generation step on GPU.

        For live visualization this transfers the grid back to host each step.
        This adds overhead compared to bulk run but enables real-time display.

        Yields:
            Tuple[int, np.ndarray, float]: (generation_number, grid_snapshot, step_time_ms)
        """
        self._check_cuda()
        if not _CUDA_READY:
            raise RuntimeError("CUDA is not available on this system.")

        width, height = initial_grid.shape
        bw, bh = self.block_size
        threads_per_block = (bw, bh)
        blocks_per_grid = (math.ceil(width / bw), math.ceil(height / bh))

        kernel = _kernel_wrap if self.edge_behavior == EdgeBehavior.WRAP_AROUND.value else _kernel_dead_edges

        d_current = cuda.to_device(np.ascontiguousarray(initial_grid, dtype=np.uint8))
        d_next = cuda.device_array_like(d_current)

        for gen in range(1, iterations + 1):
            cuda.synchronize()
            step_start = time.perf_counter()

            kernel[blocks_per_grid, threads_per_block](d_current, d_next, width, height)
            cuda.synchronize()

            step_ms = (time.perf_counter() - step_start) * 1000.0

            d_current, d_next = d_next, d_current

            # Copy back to host for live display
            host_grid = d_current.copy_to_host()
            yield gen, host_grid, step_ms

    @staticmethod
    def get_gpu_info() -> dict:
        """
        Return a dict with GPU device name, CUDA version, and VRAM.
        Returns empty/fallback strings when CUDA is not available.
        """
        info = {"gpu_name": "N/A", "cuda_version": "N/A", "vram_bytes": 0}
        try:
            from numba import cuda as _cuda
            if _cuda.is_available():
                device = _cuda.get_current_device()
                info["gpu_name"] = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
                # CUDA compute capability
                cc = device.compute_capability
                info["cuda_version"] = f"Compute {cc[0]}.{cc[1]}"
                # Total memory
                ctx = device.primary_context
                meminfo = _cuda.current_context().get_memory_info()
                info["vram_bytes"] = meminfo.total
        except Exception:
            pass
        return info
