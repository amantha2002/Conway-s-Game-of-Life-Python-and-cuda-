"""
Interactive ipywidgets Dashboard for Jupyter Notebook (.ipynb).
Renders top settings panel, real-time preview canvas/image below, and the Run button.
"""

import io
from typing import Callable, Optional
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from src.config import (
    SimulationConfig,
    AVAILABLE_GRID_SIZES,
    BLOCK_SIZE_STRINGS,
    PatternType,
    EdgeBehavior,
    RunMode,
    DisplayMode,
    parse_block_size,
)
from src.patterns import create_initial_grid, PATTERN_NAMES
from src.preview import render_preview_image
from src.benchmark import run_benchmark


def create_notebook_dashboard(on_run_callback: Optional[Callable[[SimulationConfig], None]] = None):
    """
    Constructs and displays the interactive Jupyter Notebook widget dashboard.
    
    Layout:
    1. Header title & info
    2. Top snippet: Settings controls in an organized grid
    3. Below snippet: Live state preview image and statistics
    4. Action: Run button and output log area
    """
    # 1. Widgets definition for all settings
    w_grid_size = widgets.Dropdown(
        options=AVAILABLE_GRID_SIZES,
        value=128,
        description="Grid size:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_iterations = widgets.IntText(
        value=100,
        description="Iterations:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_pattern = widgets.Dropdown(
        options=PATTERN_NAMES,
        value=PatternType.RANDOM.value,
        description="Initial pattern:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_density = widgets.IntSlider(
        value=25,
        min=10,
        max=50,
        step=1,
        description="Random density (%):",
        style={"description_width": "130px"},
        layout=widgets.Layout(width="340px")
    )

    w_seed = widgets.IntText(
        value=42,
        description="Random seed:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_edge = widgets.Dropdown(
        options=[EdgeBehavior.WRAP_AROUND.value, EdgeBehavior.DEAD_EDGES.value],
        value=EdgeBehavior.WRAP_AROUND.value,
        description="Edge behavior:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_run_mode = widgets.Dropdown(
        options=[RunMode.BOTH.value, RunMode.CPU_ONLY.value, RunMode.GPU_ONLY.value],
        value=RunMode.BOTH.value,
        description="Run mode:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_display_mode = widgets.Dropdown(
        options=[DisplayMode.LIVE_VIEW.value, DisplayMode.BENCHMARK.value],
        value=DisplayMode.LIVE_VIEW.value,
        description="Display mode:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="260px")
    )

    w_cell_size = widgets.IntSlider(
        value=4,
        min=1,
        max=16,
        step=1,
        description="Cell size (px):",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="340px")
    )

    w_block_size = widgets.Dropdown(
        options=BLOCK_SIZE_STRINGS,
        value="16x16",
        description="CUDA block size:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="260px")
    )

    # Preview widgets
    w_stats_label = widgets.HTML(
        value="<b>Pattern Preview:</b> Initializing...",
        layout=widgets.Layout(margin="6px 0px")
    )
    w_preview_image = widgets.Image(
        format="png",
        width=320,
        height=320,
        layout=widgets.Layout(border="2px solid #334155", margin="6px auto")
    )

    # Run button & output area
    w_btn_run = widgets.Button(
        description="▶ Run Simulation / Benchmark",
        button_style="info",
        tooltip="Execute Game of Life simulation or benchmark",
        icon="play",
        layout=widgets.Layout(width="320px", height="42px", margin="10px 0px")
    )
    w_output = widgets.Output(layout=widgets.Layout(border="1px solid #1e293b", padding="8px", margin="8px 0px"))

    def get_current_config() -> SimulationConfig:
        return SimulationConfig(
            grid_size=int(w_grid_size.value),
            iterations=int(w_iterations.value),
            pattern=str(w_pattern.value),
            random_density=float(w_density.value) / 100.0,
            random_seed=int(w_seed.value),
            edge_behavior=str(w_edge.value),
            run_mode=str(w_run_mode.value),
            display_mode=str(w_display_mode.value),
            cell_size=int(w_cell_size.value),
            cuda_block_size=parse_block_size(str(w_block_size.value)),
        )

    def update_preview_display(*args):
        try:
            cfg = get_current_config()
            grid = create_initial_grid(cfg)
            total = cfg.grid_size * cfg.grid_size
            alive = int(np.sum(grid))
            pct = (alive / total) * 100.0

            w_stats_label.value = (
                f"<div style='font-family: monospace; color: #38bdf8; background: #0f172a; padding: 6px; border-radius: 4px;'>"
                f"Pattern: <b>{cfg.pattern}</b> | Grid: <b>{cfg.grid_size}×{cfg.grid_size}</b> | "
                f"Alive Cells: <b>{alive:,}</b> ({pct:.1f}%) | Seed: <b>{cfg.random_seed}</b>"
                f"</div>"
            )

            pil_img = render_preview_image(grid, target_size=320)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            w_preview_image.value = buf.getvalue()
        except Exception as e:
            w_stats_label.value = f"<span style='color:red;'>Preview Error: {e}</span>"

    # Hook change listeners to update preview live
    w_grid_size.observe(update_preview_display, names="value")
    w_pattern.observe(update_preview_display, names="value")
    w_density.observe(update_preview_display, names="value")
    w_seed.observe(update_preview_display, names="value")
    w_edge.observe(update_preview_display, names="value")
    w_cell_size.observe(update_preview_display, names="value")

    def on_run_clicked(b):
        cfg = get_current_config()
        with w_output:
            w_output.clear_output()
            print(f"🚀 Running Conway's Game of Life ({cfg.run_mode}) with Grid {cfg.grid_size}x{cfg.grid_size}, {cfg.iterations} steps...")
            if on_run_callback:
                on_run_callback(cfg)
            else:
                res = run_benchmark(cfg)
                print(res.summary_table())

    w_btn_run.on_click(on_run_clicked)

    # Initial rendering
    update_preview_display()

    # Layout Assembly: Top Settings, Below Preview, Bottom Run
    settings_row1 = widgets.HBox([w_grid_size, w_iterations, w_pattern])
    settings_row2 = widgets.HBox([w_density, w_seed, w_edge])
    settings_row3 = widgets.HBox([w_run_mode, w_display_mode, w_block_size])
    settings_row4 = widgets.HBox([w_cell_size])

    settings_box = widgets.VBox(
        [
            widgets.HTML("<h3 style='color: #0284c7; margin-bottom: 4px;'>⚙ Simulation Configuration</h3>"),
            settings_row1,
            settings_row2,
            settings_row3,
            settings_row4,
        ],
        layout=widgets.Layout(
            border="1px solid #334155",
            padding="12px",
            border_radius="8px",
            margin="4px 0px 12px 0px",
            background_color="#0b0f19"
        )
    )

    preview_box = widgets.VBox(
        [
            widgets.HTML("<h3 style='color: #10b981; margin-bottom: 4px;'>👁 Initial Pattern Preview</h3>"),
            w_stats_label,
            w_preview_image,
        ],
        layout=widgets.Layout(
            border="1px solid #334155",
            padding="12px",
            border_radius="8px",
            margin="4px 0px 12px 0px",
            background_color="#0b0f19",
            align_items="center"
        )
    )

    action_box = widgets.VBox([w_btn_run, w_output], layout=widgets.Layout(align_items="center"))

    dashboard = widgets.VBox([settings_box, preview_box, action_box])
    display(dashboard)
    return dashboard
