"""
Tkinter Native Desktop Settings & Preview Window for Conway's Game of Life.
Contains top settings panel, real-time preview canvas below, and a Run button.
"""

import threading
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
from src.cuda_engine import CUDAEngine
from src.gui.results_window import ResultsWindow


class GameOfLifeSettingsWindow:
    """Tkinter Desktop Window with top settings and bottom real-time preview."""

    CANVAS_SIZE = 240

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        on_run_callback: Optional[Callable[[SimulationConfig], None]] = None
    ):
        self.root = root if root is not None else tk.Tk()
        self.root.title("Conway's Game of Life — HPC Settings & Preview")

        # Responsive initial dimensions calculated to fit safely inside screen bounds
        # Automatically accounts for 1080p displays with 125%/150% scaling (864px/720px height)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(620, max(520, screen_w - 60))
        win_h = min(660, max(500, screen_h - 120))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2 - 20)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(480, 420)
        self.root.configure(bg="#121316")

        self.on_run_callback = on_run_callback
        self.current_grid: Optional[np.ndarray] = None
        self.photo_img: Optional[ImageTk.PhotoImage] = None
        self._last_cuda_diag = None

        # Build UI
        self._init_styles()

        # 1. Action panel PINNED at the bottom first:
        # Tkinter pack order ensures bottom-pinned widgets NEVER disappear or get squeezed out when resized.
        self._build_action_panel()

        # 2. Responsive scrollable container occupying all remaining upper window space
        self._build_scrollable_container()

        # 3. Mount content panels inside the scrollable content frame
        self._build_header()
        self._build_cuda_status_panel()
        self._build_settings_panel()
        self._build_preview_panel()

        # 4. Bind mouse wheel scrolling across the entire window
        self._bind_mousewheel()

        # Initial preview rendering & automatic initial CUDA detection
        self.update_preview()
        self._on_check_cuda_click()

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

        # Dark theme scrollbar styling
        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background="#1e2026",
            darkcolor="#1e2026",
            lightcolor="#1e2026",
            troughcolor="#121316",
            bordercolor="#18191e",
            arrowcolor="#64748b"
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", "#2d3748"), ("pressed", "#38bdf8")]
        )

    def _build_scrollable_container(self):
        """Scrollable container so all settings and preview remain accessible on any display size."""
        self.container = tk.Frame(self.root, bg="#121316")
        self.container.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.container,
            bg="#121316",
            highlightthickness=0,
            bd=0
        )
        self.scrollbar = ttk.Scrollbar(
            self.container,
            orient="vertical",
            style="Vertical.TScrollbar",
            command=self.canvas.yview
        )
        self.content_frame = tk.Frame(self.canvas, bg="#121316")

        # Dynamically compute scrollable region whenever content size updates
        self.content_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window_id = self.canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Expand content frame width when window/canvas width changes
        def _on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window_id, width=event.width)

        self.canvas.bind("<Configure>", _on_canvas_configure)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

    def _bind_mousewheel(self):
        """Enable mouse wheel scrolling smoothly across the window."""
        def _on_mousewheel(event):
            try:
                if not hasattr(self, "canvas") or not self.canvas.winfo_exists():
                    return
                # Do not intercept wheel if inside an active combobox dropdown
                widget = event.widget
                if isinstance(widget, (ttk.Combobox, tk.Listbox)) and widget.focus_get() == widget:
                    return
                if event.delta:
                    self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.canvas.yview_scroll(1, "units")
            except Exception:
                pass

        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

        def _on_destroy(event):
            if event.widget == self.root:
                try:
                    self.root.unbind_all("<MouseWheel>")
                    self.root.unbind_all("<Button-4>")
                    self.root.unbind_all("<Button-5>")
                except Exception:
                    pass

        self.root.bind("<Destroy>", _on_destroy)

    def _build_header(self):
        header_frame = tk.Frame(self.content_frame, bg="#121316")
        header_frame.pack(fill="x", padx=14, pady=(8, 4))

        title = ttk.Label(header_frame, text="Conway's Game of Life", style="Header.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header_frame,
            text="High Performance Computing: CUDA GPU vs Standard CPU Benchmark",
            style="Subheader.TLabel"
        )
        subtitle.pack(anchor="w")

    def _build_cuda_status_panel(self):
        """Card containing the 'Check CUDA Available' button and live status display."""
        cuda_card = ttk.LabelFrame(
            self.content_frame, text=" ⚡ CUDA Hardware Acceleration ", style="Section.TLabelframe", padding=10
        )
        cuda_card.pack(fill="x", padx=14, pady=(4, 6))

        content = tk.Frame(cuda_card, bg="#18191e")
        content.pack(fill="x")
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=0)

        # 1. Action Button: Check CUDA Available
        self.btn_check_cuda = tk.Button(
            content,
            text="Check CUDA Available",
            font=("Segoe UI", 9, "bold"),
            bg="#1e293b",
            fg="#f8fafc",
            activebackground="#334155",
            activeforeground="#38bdf8",
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self._on_check_cuda_click
        )
        self.btn_check_cuda.grid(row=0, column=0, rowspan=2, padx=(0, 14), sticky="ns")

        # 2. Main Status Label (CUDA is available in green / No CUDA found in red)
        self.lbl_cuda_status = tk.Label(
            content,
            text="Checking CUDA status...",
            font=("Segoe UI", 11, "bold"),
            fg="#94a3b8",
            bg="#18191e"
        )
        self.lbl_cuda_status.grid(row=0, column=1, sticky="w")

        # 3. Sub-detail Label
        self.lbl_cuda_details = tk.Label(
            content,
            text="Click 'Check CUDA Available' to probe hardware.",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#18191e"
        )
        self.lbl_cuda_details.grid(row=1, column=1, sticky="w", pady=(2, 0))

        # 4. Info Modal Button
        self.btn_cuda_info = tk.Button(
            content,
            text="Specs",
            font=("Segoe UI", 8, "bold"),
            bg="#1e293b",
            fg="#38bdf8",
            activebackground="#334155",
            activeforeground="#38bdf8",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=3,
            command=self._show_cuda_details_modal
        )
        self.btn_cuda_info.grid(row=0, column=2, rowspan=2, padx=(8, 0), sticky="e")
        self.btn_cuda_info.grid_remove()

    def _on_check_cuda_click(self):
        """Probe CUDA hardware and update status."""
        self.lbl_cuda_status.config(text="Probing CUDA hardware & drivers...", fg="#38bdf8")
        self.lbl_cuda_details.config(text="Querying NVIDIA Driver & runtime APIs...", fg="#94a3b8")
        self.btn_check_cuda.config(state="disabled")
        self.root.update_idletasks()

        def _do_check():
            try:
                diag = CUDAEngine.check_cuda_detailed()
                self._apply_cuda_diag(diag)
            except Exception as e:
                self.lbl_cuda_status.config(text="No CUDA found", fg="#ef4444")
                self.lbl_cuda_details.config(text=f"Check error: {e}", fg="#fca5a5")
            finally:
                self.btn_check_cuda.config(state="normal")

        self.root.after(20, _do_check)

    def _apply_cuda_diag(self, diag: dict):
        self.btn_check_cuda.config(state="normal")
        self._last_cuda_diag = diag

        if diag["is_available"]:
            # GREEN COLOR as requested: "CUDA is available"
            self.lbl_cuda_status.config(
                text="CUDA is available",
                fg="#22c55e"  # Vibrant green
            )
            sub = f"{diag['gpu_name']}  |  Compute {diag['compute_capability']}  |  {diag['vram_mb'] / 1024:.1f} GB VRAM  |  CUDA {diag['driver_version']}"
            self.lbl_cuda_details.config(text=sub, fg="#a7f3d0")
            self.btn_cuda_info.grid()
        else:
            # RED COLOR as requested: "No CUDA found"
            self.lbl_cuda_status.config(
                text="No CUDA found",
                fg="#ef4444"  # Vibrant red
            )
            self.lbl_cuda_details.config(
                text="No NVIDIA GPU or CUDA driver detected on this system.",
                fg="#fca5a5"
            )
            self.btn_cuda_info.grid_remove()

    def _show_cuda_details_modal(self):
        """Show full CUDA diagnostic details in a sleek modal dialog."""
        diag = self._last_cuda_diag or CUDAEngine.check_cuda_detailed()
        top = tk.Toplevel(self.root)
        top.title("CUDA Hardware Diagnostics")
        top.geometry("460x360")
        top.configure(bg="#121316")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()

        tk.Label(
            top, text="⚡ CUDA Hardware & Driver Profile",
            font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg="#121316"
        ).pack(pady=(16, 8))

        card = tk.Frame(top, bg="#18191e", highlightbackground="#2a2d3e", highlightthickness=1)
        card.pack(fill="both", expand=True, padx=16, pady=8)

        status_text = "CUDA is available" if diag["is_available"] else "No CUDA found"
        status_color = "#22c55e" if diag["is_available"] else "#ef4444"

        items = [
            ("Status", status_text, status_color),
            ("Device Name", diag.get("gpu_name", "N/A"), "#f8fafc"),
            ("Compute Capability", diag.get("compute_capability", "N/A"), "#38bdf8"),
            ("Total VRAM", f"{diag.get('vram_mb', 0)/1024:.2f} GB ({diag.get('vram_mb', 0):.0f} MB)", "#f8fafc"),
            ("CUDA Driver Version", diag.get("driver_version", "N/A"), "#f8fafc"),
            ("Device Count", str(diag.get("device_count", 0)), "#f8fafc"),
            ("JIT Kernel Compiler", "Ready (Numba CUDA)" if diag.get("jit_ready") else "Driver Ready (Active)", "#a7f3d0"),
            ("Recommended Block", "16×16 (256 threads/block)", "#fb923c"),
        ]

        for label, val, color in items:
            row = tk.Frame(card, bg="#18191e")
            row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=f"{label}:", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#18191e", width=22, anchor="w").pack(side="left")
            tk.Label(row, text=val, font=("Segoe UI", 9), fg=color, bg="#18191e", anchor="w").pack(side="left")

        tk.Button(
            top, text="Close", font=("Segoe UI", 9, "bold"),
            bg="#1e293b", fg="#f8fafc", activebackground="#334155",
            relief="flat", cursor="hand2", padx=18, pady=4,
            command=top.destroy
        ).pack(pady=(4, 12))

    def _build_settings_panel(self):
        """Top snippet: Settings frame containing all required controls."""
        settings_frame = ttk.LabelFrame(self.content_frame, text=" ⚙ Configuration Settings ", style="Section.TLabelframe", padding=8)
        settings_frame.pack(fill="x", padx=14, pady=(4, 6))

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
        preview_frame = ttk.LabelFrame(self.content_frame, text=" 👁 Initial State Preview ", style="Section.TLabelframe", padding=8)
        preview_frame.pack(fill="x", padx=14, pady=(4, 6))

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
            preview_frame, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="#18181b", highlightthickness=1, highlightbackground="#27272a"
        )
        self.preview_canvas.pack(pady=4)

    def _build_action_panel(self):
        """Action button section with the requested Run button pinned to the bottom."""
        action_frame = tk.Frame(self.root, bg="#121316")
        action_frame.pack(side="bottom", fill="x", padx=14, pady=(4, 10))

        # Subtle divider separating settings from action footer
        sep = tk.Frame(action_frame, height=1, bg="#27272a")
        sep.pack(fill="x", pady=(0, 6))

        # Run button — pinned and always visible
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
        self.btn_run.pack(fill="x", pady=(2, 4))

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
            pil_img = render_preview_image(self.current_grid, target_size=self.CANVAS_SIZE)
            self.photo_img = ImageTk.PhotoImage(pil_img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                self.CANVAS_SIZE // 2,
                self.CANVAS_SIZE // 2,
                image=self.photo_img
            )

        except Exception as e:
            self.lbl_stats.config(text=f"Preview Error: {e}")

    def _on_run_click(self):
        """Triggered when the user clicks the Run button."""
        config = self.get_config()
        self.lbl_run_status.config(
            text=f"Status: Launching {config.run_mode} ({config.grid_size}x{config.grid_size}, {config.iterations} steps)...",
            foreground="#38bdf8"
        )
        self.root.update_idletasks()

        if self.on_run_callback:
            # Delegate to custom callback if provided
            self.on_run_callback(config)
        else:
            # Default action: open the Results Window with live simulation
            try:
                ResultsWindow(self.root, config)
                self.lbl_run_status.config(
                    text="Status: Results window opened — simulation running.",
                    foreground="#4ade80"
                )
            except Exception as e:
                self.lbl_run_status.config(text=f"Status Error: {e}", foreground="#f87171")


def open_settings_window(on_run_callback: Optional[Callable[[SimulationConfig], None]] = None):
    """Convenience function to open standalone Tkinter settings window."""
    app = GameOfLifeSettingsWindow(on_run_callback=on_run_callback)
    app.root.mainloop()
    return app
