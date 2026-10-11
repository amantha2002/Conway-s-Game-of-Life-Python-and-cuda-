"""
GUI subpackage providing desktop Tkinter and Jupyter notebook widget interfaces.
"""

from src.gui.tk_window import GameOfLifeSettingsWindow, open_settings_window
from src.gui.ipy_widget import create_notebook_dashboard

__all__ = [
    "GameOfLifeSettingsWindow",
    "open_settings_window",
    "create_notebook_dashboard",
]
