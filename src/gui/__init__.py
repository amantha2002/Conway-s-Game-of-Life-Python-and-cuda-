"""
GUI subpackage providing desktop Tkinter and Jupyter notebook widget interfaces.
"""

# Optional Tkinter GUI import – may fail if Tk is not installed
try:
    from src.gui.tk_window import GameOfLifeSettingsWindow, open_settings_window
except Exception:  # pragma: no cover
    GameOfLifeSettingsWindow = None
    open_settings_window = None
# Optional Jupyter notebook dashboard import – may fail if ipywidgets not installed
try:
    from src.gui.ipy_widget import create_notebook_dashboard
except Exception:  # pragma: no cover
    create_notebook_dashboard = None
# Optional results window import – also depends on Tkinter
try:
    from src.gui.results_window import ResultsWindow, open_results_window
except Exception:  # pragma: no cover
    ResultsWindow = None
    open_results_window = None

__all__ = [
    "GameOfLifeSettingsWindow",
    "open_settings_window",
    "create_notebook_dashboard",
    "ResultsWindow",
    "open_results_window",
]

