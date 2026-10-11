"""
Tkinter Native Desktop Settings & Preview Window for Conway's Game of Life.
Contains top settings panel, real-time preview canvas below, and a Run button.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional
import numpy as np
from PIL import ImageTk

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


class GameOfLifeSettingsWindow:
    """Tkinter Desktop Window with top settings and bottom real-time preview."""

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        on_run_callback: Optional[Callable[[SimulationConfig], None]] = None
    ):
        self.root = root if root is not None else tk.Tk()
        self.root.title("Conway's Game of Life — HPC Settings & Preview")
        self.root.geometry("640x780")
        self.root.minsize(580, 720)
        self.root.configure(bg="#121316")

        self.on_run_callback = on_run_callback
        self.current_grid: Optional[np.ndarray] = None
        self.photo_img: Optional[ImageTk.PhotoImage] = None

        # Build UI
        self._init_styles()
        self._build_header()
        self._build_settings_panel()
        self._build_preview_panel()
        self._build_action_panel()

        # Initial preview rendering
        self.update_preview()

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Global dark palette
        style.configure(".", background="#121316", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("TLabel", background="#121316", foreground="#cbd5e1", font=("Segoe UI", 9))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#38bdf8", background="#121316")
        style.configure("Subheader.TLabel", font=("Segoe UI", 9), foreground="#94a3b8", background="#121316")
        style.configure("Section.TLabelframe", background="#18191e", foreground="#38bdf8", bordercolor="#2d3748")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#38bdf8", background="#18191e")

        style.configure("TCombobox", fieldbackground="#1e2026", background="#2d3748", foreground="#f8fafc")
        style.map("TCombobox", fieldbackground=[("readonly", "#1e2026")], foreground=[("readonly", "#f8fafc")])

        style.configure("TScale", background="#18191e", troughcolor="#27272a")

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg="#121316")
        header_frame.pack(fill="x", padx=16, pady=(12, 6))

        title = ttk.Label(header_frame, text="Conway's Game of Life", style="Header.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header_frame,
            text="High Performance Computing: CUDA GPU vs Standard CPU Benchmark",
            style="Subheader.TLabel"
        )
        subtitle.pack(anchor="w")

    def _build_settings_panel(self):
        """Top snippet: Settings frame containing all required controls."""
        settings_frame = ttk.LabelFrame(self.root, text=" ⚙ Configuration Settings ", style="Section.TLabelframe", padding=10)
        settings_frame.pack(fill="x", padx=16, pady=6)

        # 3-column responsive grid layout
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(2, weight=1)

        # Row 0: Grid Size, Iterations, Initial Pattern
        # Grid Size
        f_grid = tk.Frame(settings_frame, bg="#18191e")
        f_grid.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_grid, text="Grid size:").pack(anchor="w")
        self.var_grid_size = tk.IntVar(value=128)
        self.cb_grid_size = ttk.Combobox(
            f_grid, values=AVAILABLE_GRID_SIZES, textvariable=self.var_grid_size, state="readonly", width=12
        )
        self.cb_grid_size.pack(fill="x", pady=2)
        self.cb_grid_size.bind("<<ComboboxSelected>>", lambda e: self.update_preview())

        # Iterations
        f_iter = tk.Frame(settings_frame, bg="#18191e")
        f_iter.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_iter, text="Iterations:").pack(anchor="w")
        self.var_iterations = tk.IntVar(value=100)
        self.sp_iterations = tk.Spinbox(
            f_iter, from_=1, to=10000, textvariable=self.var_iterations,
            bg="#1e2026", fg="#f8fafc", insertbackground="white", width=12
        )
        self.sp_iterations.pack(fill="x", pady=2)

        # Initial Pattern
        f_pat = tk.Frame(settings_frame, bg="#18191e")
        f_pat.grid(row=0, column=2, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_pat, text="Initial pattern:").pack(anchor="w")
        self.var_pattern = tk.StringVar(value=PatternType.RANDOM.value)
        self.cb_pattern = ttk.Combobox(
            f_pat, values=PATTERN_NAMES, textvariable=self.var_pattern, state="readonly", width=14
        )
        self.cb_pattern.pack(fill="x", pady=2)
        self.cb_pattern.bind("<<ComboboxSelected>>", lambda e: self.update_preview())

        # Row 1: Random Density, Random Seed, Edge Behavior
        # Random Density Slider
        f_den = tk.Frame(settings_frame, bg="#18191e")
        f_den.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.lbl_density = ttk.Label(f_den, text="Random density: 25%")
        self.lbl_density.pack(anchor="w")
        self.var_density = tk.IntVar(value=25)
        self.slider_density = ttk.Scale(
            f_den, from_=10, to=50, variable=self.var_density, orient="horizontal",
            command=self._on_density_change
        )
        self.slider_density.pack(fill="x", pady=2)

        # Random Seed
        f_seed = tk.Frame(settings_frame, bg="#18191e")
        f_seed.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_seed, text="Random seed:").pack(anchor="w")
        self.var_seed = tk.IntVar(value=42)
        self.sp_seed = tk.Spinbox(
            f_seed, from_=0, to=999999, textvariable=self.var_seed,
            bg="#1e2026", fg="#f8fafc", insertbackground="white", width=12,
            command=self.update_preview
        )
        self.sp_seed.pack(fill="x", pady=2)
        self.sp_seed.bind("<KeyRelease>", lambda e: self.update_preview())

        # Edge Behavior
        f_edge = tk.Frame(settings_frame, bg="#18191e")
        f_edge.grid(row=1, column=2, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_edge, text="Edge behavior:").pack(anchor="w")
        self.var_edge = tk.StringVar(value=EdgeBehavior.WRAP_AROUND.value)
        self.cb_edge = ttk.Combobox(
            f_edge, values=[EdgeBehavior.WRAP_AROUND.value, EdgeBehavior.DEAD_EDGES.value],
            textvariable=self.var_edge, state="readonly", width=14
        )
        self.cb_edge.pack(fill="x", pady=2)
        self.cb_edge.bind("<<ComboboxSelected>>", lambda e: self.update_preview())

        # Row 2: Run Mode, Display Mode, Cell size & CUDA Block Size
        # Run Mode
        f_run = tk.Frame(settings_frame, bg="#18191e")
        f_run.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_run, text="Run mode:").pack(anchor="w")
        self.var_run_mode = tk.StringVar(value=RunMode.BOTH.value)
        self.cb_run_mode = ttk.Combobox(
            f_run, values=[RunMode.BOTH.value, RunMode.CPU_ONLY.value, RunMode.GPU_ONLY.value],
            textvariable=self.var_run_mode, state="readonly", width=12
        )
        self.cb_run_mode.pack(fill="x", pady=2)

        # Display Mode
        f_disp = tk.Frame(settings_frame, bg="#18191e")
        f_disp.grid(row=2, column=1, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_disp, text="Display mode:").pack(anchor="w")
        self.var_disp_mode = tk.StringVar(value=DisplayMode.LIVE_VIEW.value)
        self.cb_disp_mode = ttk.Combobox(
            f_disp, values=[DisplayMode.LIVE_VIEW.value, DisplayMode.BENCHMARK.value],
            textvariable=self.var_disp_mode, state="readonly", width=12
        )
        self.cb_disp_mode.pack(fill="x", pady=2)

        # CUDA Block Size
        f_blk = tk.Frame(settings_frame, bg="#18191e")
        f_blk.grid(row=2, column=2, sticky="nsew", padx=6, pady=4)
        ttk.Label(f_blk, text="CUDA block size:").pack(anchor="w")
        self.var_block_size = tk.StringVar(value="16x16")
        self.cb_block_size = ttk.Combobox(
            f_blk, values=BLOCK_SIZE_STRINGS, textvariable=self.var_block_size, state="readonly", width=14
        )
        self.cb_block_size.pack(fill="x", pady=2)

        # Row 3: Cell Size Slider across bottom of settings
        f_cell = tk.Frame(settings_frame, bg="#18191e")
        f_cell.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=6, pady=(4, 2))
        self.lbl_cell_size = ttk.Label(f_cell, text="Cell size: 4 px")
        self.lbl_cell_size.pack(anchor="w")
        self.var_cell_size = tk.IntVar(value=4)
        self.slider_cell_size = ttk.Scale(
            f_cell, from_=1, to=16, variable=self.var_cell_size, orient="horizontal",
            command=self._on_cell_size_change
        )
        self.slider_cell_size.pack(fill="x", pady=2)

    def _on_density_change(self, val):
        density_pct = int(float(val))
        self.lbl_density.config(text=f"Random density: {density_pct}%")
        self.update_preview()

    def _on_cell_size_change(self, val):
        cell_px = int(float(val))
        self.lbl_cell_size.config(text=f"Cell size: {cell_px} px")

    def _build_preview_panel(self):
        """Below snippet: Real-time preview canvas of the starting grid state."""
        preview_frame = ttk.LabelFrame(self.root, text=" 👁 Initial State Preview ", style="Section.TLabelframe", padding=8)
        preview_frame.pack(fill="both", expand=True, padx=16, pady=6)

        # Stats banner
        self.lbl_stats = ttk.Label(
            preview_frame,
            text="Pattern: Random | Grid: 128×128 | Active Cells: 0 (0.0%)",
            font=("Consolas", 9),
            foreground="#38bdf8"
        )
        self.lbl_stats.pack(anchor="w", pady=(0, 4))

        # Canvas for the live image
        self.preview_canvas = tk.Canvas(
            preview_frame, width=320, height=320, bg="#18181b", highlightthickness=1, highlightbackground="#27272a"
        )
        self.preview_canvas.pack(pady=4)

    def _build_action_panel(self):
        """Action button section with the requested Run button."""
        action_frame = tk.Frame(self.root, bg="#121316")
        action_frame.pack(fill="x", padx=16, pady=(6, 14))

        # Run button
        self.btn_run = tk.Button(
            action_frame,
            text="▶  RUN SIMULATION / BENCHMARK",
            font=("Segoe UI", 11, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=8,
            command=self._on_run_click
        )
        self.btn_run.pack(fill="x", pady=4)

        # Status output line
        self.lbl_run_status = ttk.Label(
            action_frame,
            text="Status: Ready (Configure settings above and click Run)",
            font=("Segoe UI", 9, "italic"),
            foreground="#94a3b8"
        )
        self.lbl_run_status.pack(anchor="w", pady=2)

    def get_config(self) -> SimulationConfig:
        """Read and validate current UI values into SimulationConfig."""
        cfg = SimulationConfig(
            grid_size=int(self.var_grid_size.get()),
            iterations=int(self.var_iterations.get()),
            pattern=self.var_pattern.get(),
            random_density=float(self.var_density.get()) / 100.0,
            random_seed=int(self.var_seed.get()),
            edge_behavior=self.var_edge.get(),
            run_mode=self.var_run_mode.get(),
            display_mode=self.var_disp_mode.get(),
            cell_size=int(self.var_cell_size.get()),
            cuda_block_size=parse_block_size(self.var_block_size.get()),
        )
        return cfg

    def update_preview(self):
        """Regenerate the initial grid and update the preview canvas."""
        try:
            config = self.get_config()
            self.current_grid = create_initial_grid(config)

            # Update stats
            total_cells = config.grid_size * config.grid_size
            alive_cells = int(np.sum(self.current_grid))
            ratio = (alive_cells / total_cells) * 100.0
            self.lbl_stats.config(
                text=f"Pattern: {config.pattern} | Grid: {config.grid_size}×{config.grid_size} | Alive: {alive_cells:,} ({ratio:.1f}%)"
            )

            # Render image
            pil_img = render_preview_image(self.current_grid, target_size=320)
            self.photo_img = ImageTk.PhotoImage(pil_img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(160, 160, image=self.photo_img)

        except Exception as e:
            self.lbl_stats.config(text=f"Preview Error: {e}")

    def _on_run_click(self):
        """Triggered when the user clicks the Run button."""
        config = self.get_config()
        self.lbl_run_status.config(
            text=f"Status: Executing {config.run_mode} ({config.grid_size}x{config.grid_size}, {config.iterations} steps)...",
            foreground="#38bdf8"
        )
        self.root.update_idletasks()

        if self.on_run_callback:
            # Delegate to custom callback if provided
            self.on_run_callback(config)
        else:
            # Default action: execute benchmark and display results
            try:
                res = run_benchmark(config)
                speedup_msg = f" | Speedup: {res.kernel_speedup:.2f}x" if res.kernel_speedup else ""
                match_msg = " [Results Match!]" if res.grids_match else ""
                self.lbl_run_status.config(
                    text=f"Status: Done! CPU: {res.cpu_time_sec*1000 if res.cpu_time_sec else 0:.1f}ms | GPU: {res.cuda_kernel_time_sec*1000 if res.cuda_kernel_time_sec else 0:.1f}ms{speedup_msg}{match_msg}",
                    foreground="#4ade80"
                )
                print(res.summary_table())
            except Exception as e:
                self.lbl_run_status.config(text=f"Status Error: {e}", foreground="#f87171")


def open_settings_window(on_run_callback: Optional[Callable[[SimulationConfig], None]] = None):
    """Convenience function to open standalone Tkinter settings window."""
    app = GameOfLifeSettingsWindow(on_run_callback=on_run_callback)
    app.root.mainloop()
    return app
