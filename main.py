"""
Main entry point for Conway's Game of Life (HPC CPU vs CUDA).
Runs the native desktop Settings & Preview window.
"""

import sys
from src.gui.tk_window import open_settings_window
from src.config import SimulationConfig
from src.benchmark import run_benchmark


def on_run(config: SimulationConfig):
    """
    Callback when user clicks the Run button in the UI.
    User can expand this to launch their custom simulation loop or visualization window.
    """
    print(f"\n[RUN TRIGGERED] Running Conway's Game of Life with config:")
    print(f"  Grid Size:       {config.grid_size}x{config.grid_size}")
    print(f"  Iterations:      {config.iterations}")
    print(f"  Pattern:         {config.pattern}")
    print(f"  Random Density:  {config.random_density * 100:.1f}%")
    print(f"  Random Seed:     {config.random_seed}")
    print(f"  Edge Behavior:   {config.edge_behavior}")
    print(f"  Run Mode:        {config.run_mode}")
    print(f"  Display Mode:    {config.display_mode}")
    print(f"  Cell Size:       {config.cell_size}px")
    print(f"  CUDA Block Size: {config.block_size_str}")
    
    # Execute benchmark / simulation
    result = run_benchmark(config)
    print("\n" + result.summary_table())


if __name__ == "__main__":
    print("Launching Conway's Game of Life Settings & Preview Window...")
    open_settings_window(on_run_callback=on_run)
