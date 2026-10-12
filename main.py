"""
Main entry point for Conway's Game of Life (HPC CPU vs CUDA).
Runs the native desktop Settings & Preview window.
"""

import sys

try:
    from src.gui.tk_window import open_settings_window
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    print("Tkinter is required for the desktop GUI.", file=sys.stderr)
    print("Install it with:  brew install python-tk@3.14", file=sys.stderr)
    sys.exit(1)

from src.config import SimulationConfig


if __name__ == "__main__":
    print("Launching Conway's Game of Life Settings & Preview Window...")
    # No custom callback — the default Run handler inside tk_window
    # will open the ResultsWindow with live CPU vs CUDA simulation.
    open_settings_window()

