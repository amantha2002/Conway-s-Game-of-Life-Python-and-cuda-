"""
Results Window for Conway's Game of Life — CPU vs CUDA side-by-side.

Opens as a Toplevel Tkinter window when the user clicks Run.
Shows:
  1. Split-view grids (CPU left, CUDA right) updating live
  2. Dynamic matplotlib graph of ms/generation
  3. Full benchmark stats table
  4. Control buttons: Pause, Reset, Export CSV, Save Graphs
"""

import csv
import math
import platform
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.config import SimulationConfig, RunMode, EdgeBehavior
from src.patterns import create_initial_grid
from src.preview import render_preview_image
from src.cpu_engine import CPUEngine
from src.cuda_engine import CUDAEngine
from src.benchmark import BenchmarkResult, get_hardware_info


# ─── Color Palette ───────────────────────────────────────────────────
BG_DARK       = "#0f1117"
BG_PANEL      = "#161821"
BG_CARD       = "#1c1e2a"
BORDER_COLOR  = "#2a2d3e"
TEXT_PRIMARY  = "#e2e8f0"
TEXT_SECONDARY = "#8892a8"
ACCENT_CYAN   = "#38bdf8"
ACCENT_GREEN  = "#34d399"
ACCENT_ORANGE = "#fb923c"
ACCENT_RED    = "#f87171"
ACCENT_PURPLE = "#a78bfa"
BTN_BG        = "#1e293b"
BTN_HOVER     = "#334155"
GRAPH_CPU     = "#60a5fa"   # Soft blue
GRAPH_GPU     = "#34d399"   # Emerald green


def _format_bytes(b: int) -> str:
    """Format byte count to human-readable string."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def _safe_fmt(val, fmt=".2f", suffix=""):
    """Format a numeric value safely, returning 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val:{fmt}}{suffix}"


class ResultsWindow:
    """
    A Toplevel window showing live CPU vs CUDA simulation side-by-side,
    a dynamic performance graph, benchmark stats table, and control buttons.
    """

    CANVAS_SIZE = 280  # pixels per grid canvas

    def __init__(self, parent: tk.Tk, config: SimulationConfig):
        self.parent = parent
        self.config = config
        self.cuda_available = CUDAEngine.is_available()

        # ── Simulation state ──
        self.initial_grid = create_initial_grid(config)
        self.generation = 0
        self.is_paused = False
        self.is_running = False
        self.is_finished = False
        self._stop_event = threading.Event()
        self._sim_thread: Optional[threading.Thread] = None

        # ── Timing data ──
        self.cpu_step_times: List[float] = []
        self.gpu_step_times: List[float] = []
        self.cpu_current_grid: Optional[np.ndarray] = None
        self.gpu_current_grid: Optional[np.ndarray] = None
        self.cpu_photo: Optional[ImageTk.PhotoImage] = None
        self.gpu_photo: Optional[ImageTk.PhotoImage] = None

        # ── Result holder ──
        self.result = BenchmarkResult(
            grid_size=config.grid_size,
            iterations=config.iterations,
            edge_behavior=config.edge_behavior,
            block_size_str=config.block_size_str,
        )

        # ── Build window ──
        self.win = tk.Toplevel(parent)
        self.win.title("Conway's Game of Life — Simulation Results")
        self.win.configure(bg=BG_DARK)
        self.win.minsize(960, 820)
        self.win.geometry("1060x920")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._start_simulation()

    # ───────────────────────── UI CONSTRUCTION ─────────────────────────

    def _build_ui(self):
        """Assemble all UI sections from top to bottom."""
        # Main scrollable container
        self.main_frame = tk.Frame(self.win, bg=BG_DARK)
        self.main_frame.pack(fill="both", expand=True)

        self._build_header()
        self._build_split_view()
        self._build_graph_section()
        self._build_stats_table()
        self._build_control_buttons()

    def _build_header(self):
        """Top header with simulation info."""
        hdr = tk.Frame(self.main_frame, bg=BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(14, 6))

        title = tk.Label(
            hdr, text="⚡ Simulation Results — CPU vs CUDA",
            font=("Segoe UI", 15, "bold"), fg=ACCENT_CYAN, bg=BG_DARK
        )
        title.pack(anchor="w")

        mode_text = self.config.run_mode
        if not self.cuda_available and mode_text != RunMode.CPU_ONLY.value:
            mode_text += " (CUDA unavailable — CPU only)"

        sub = tk.Label(
            hdr,
            text=(
                f"Grid: {self.config.grid_size}×{self.config.grid_size}  |  "
                f"Generations: {self.config.iterations}  |  "
                f"Pattern: {self.config.pattern}  |  "
                f"Mode: {mode_text}"
            ),
            font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_DARK
        )
        sub.pack(anchor="w", pady=(2, 0))

    def _build_split_view(self):
        """Two side-by-side panels: CPU (left) and CUDA (right)."""
        split = tk.Frame(self.main_frame, bg=BG_DARK)
        split.pack(fill="x", padx=20, pady=8)
        split.columnconfigure(0, weight=1)
        split.columnconfigure(1, weight=1)

        # ── CPU Panel ──
        cpu_card = tk.Frame(split, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        cpu_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        cpu_title = tk.Label(
            cpu_card, text="🐍  CPU (Python / NumPy)",
            font=("Segoe UI", 11, "bold"), fg=GRAPH_CPU, bg=BG_CARD
        )
        cpu_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.lbl_cpu_gen = tk.Label(
            cpu_card, text="Gen: 0  |  — ms/step",
            font=("Consolas", 9), fg=TEXT_SECONDARY, bg=BG_CARD
        )
        self.lbl_cpu_gen.pack(anchor="w", padx=12)

        self.cpu_canvas = tk.Canvas(
            cpu_card, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="#18181b", highlightthickness=0
        )
        self.cpu_canvas.pack(padx=12, pady=(6, 12))

        # ── CUDA Panel ──
        gpu_card = tk.Frame(split, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        gpu_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        gpu_title = tk.Label(
            gpu_card, text="🚀  CUDA (GPU Kernel)",
            font=("Segoe UI", 11, "bold"), fg=GRAPH_GPU, bg=BG_CARD
        )
        gpu_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.lbl_gpu_gen = tk.Label(
            gpu_card, text="Gen: 0  |  — ms/step",
            font=("Consolas", 9), fg=TEXT_SECONDARY, bg=BG_CARD
        )
        self.lbl_gpu_gen.pack(anchor="w", padx=12)

        self.gpu_canvas = tk.Canvas(
            gpu_card, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="#18181b", highlightthickness=0
        )
        self.gpu_canvas.pack(padx=12, pady=(6, 12))

        if not self.cuda_available and self.config.run_mode != RunMode.CPU_ONLY.value:
            self.gpu_canvas.create_text(
                self.CANVAS_SIZE // 2, self.CANVAS_SIZE // 2,
                text="CUDA Not Available\non this device",
                fill=ACCENT_RED, font=("Segoe UI", 11, "bold"), justify="center"
            )

        # ── Progress bar ──
        prog_frame = tk.Frame(self.main_frame, bg=BG_DARK)
        prog_frame.pack(fill="x", padx=20, pady=(0, 4))

        self.lbl_progress = tk.Label(
            prog_frame, text="Generation 0 / 0",
            font=("Consolas", 9), fg=ACCENT_CYAN, bg=BG_DARK
        )
        self.lbl_progress.pack(anchor="w")

        self.progress_bar = ttk.Progressbar(
            prog_frame, mode="determinate", maximum=self.config.iterations, length=400
        )
        self.progress_bar.pack(fill="x", pady=(2, 0))

    def _build_graph_section(self):
        """Embedded matplotlib chart for ms/generation over time."""
        graph_frame = tk.Frame(self.main_frame, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        graph_frame.pack(fill="x", padx=20, pady=8)

        tk.Label(
            graph_frame, text="📊  Performance — ms / generation",
            font=("Segoe UI", 10, "bold"), fg=ACCENT_CYAN, bg=BG_CARD
        ).pack(anchor="w", padx=12, pady=(8, 0))

        self.fig = Figure(figsize=(9.5, 2.8), dpi=100, facecolor=BG_CARD)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.line_cpu, = self.ax.plot([], [], color=GRAPH_CPU, linewidth=1.5, label="CPU (Python)", alpha=0.9)
        self.line_gpu, = self.ax.plot([], [], color=GRAPH_GPU, linewidth=1.5, label="CUDA (GPU)", alpha=0.9)
        self.ax.legend(
            loc="upper right", fontsize=8, framealpha=0.7,
            facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY
        )

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.fig_canvas.get_tk_widget().pack(fill="x", padx=8, pady=(4, 10))

    def _style_axes(self):
        """Apply dark theme to matplotlib axes."""
        self.ax.set_facecolor(BG_PANEL)
        self.ax.set_xlabel("Generation", fontsize=8, color=TEXT_SECONDARY)
        self.ax.set_ylabel("ms / step", fontsize=8, color=TEXT_SECONDARY)
        self.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER_COLOR)
        self.ax.set_xlim(0, max(self.config.iterations, 10))
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, color=BORDER_COLOR, alpha=0.3, linewidth=0.5)

    def _build_stats_table(self):
        """Treeview table showing all 11 benchmark metrics."""
        table_frame = tk.Frame(self.main_frame, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        table_frame.pack(fill="x", padx=20, pady=8)

        tk.Label(
            table_frame, text="📋  Benchmark Statistics",
            font=("Segoe UI", 10, "bold"), fg=ACCENT_CYAN, bg=BG_CARD
        ).pack(anchor="w", padx=12, pady=(8, 4))

        style = ttk.Style()
        style.configure(
            "Stats.Treeview",
            background=BG_PANEL, foreground=TEXT_PRIMARY, fieldbackground=BG_PANEL,
            font=("Consolas", 9), rowheight=24, borderwidth=0
        )
        style.configure(
            "Stats.Treeview.Heading",
            background=BG_CARD, foreground=ACCENT_CYAN,
            font=("Segoe UI", 9, "bold"), borderwidth=0
        )
        style.map("Stats.Treeview", background=[("selected", BORDER_COLOR)])

        cols = ("metric", "cpu_value", "gpu_value", "description")
        self.tree = ttk.Treeview(
            table_frame, columns=cols, show="headings", height=11, style="Stats.Treeview"
        )
        self.tree.heading("metric", text="Metric", anchor="w")
        self.tree.heading("cpu_value", text="CPU (Python)", anchor="center")
        self.tree.heading("gpu_value", text="CUDA (GPU)", anchor="center")
        self.tree.heading("description", text="Description", anchor="w")
        self.tree.column("metric", width=180, anchor="w")
        self.tree.column("cpu_value", width=160, anchor="center")
        self.tree.column("gpu_value", width=160, anchor="center")
        self.tree.column("description", width=260, anchor="w")

        self.tree.pack(fill="x", padx=12, pady=(0, 10))

        # Insert placeholder rows
        self._metric_rows = [
            ("total_time",      "Total time",               "—", "—", "Whole run time"),
            ("avg_ms",          "Avg ms/generation",        "—", "—", "Total ÷ generations"),
            ("min_max_ms",      "Min / Max ms/gen",         "—", "—", "Fastest & slowest step"),
            ("throughput",      "Throughput (cells/sec)",    "—", "—", "Cells updated per second"),
            ("speedup",         "Speedup",                  "—", "—", "CPU time ÷ GPU time"),
            ("gpu_transfer",    "GPU transfer time",        "—", "—", "CPU ↔ GPU data copy"),
            ("gpu_kernel",      "GPU kernel time",          "—", "—", "Pure compute on GPU"),
            ("memory",          "Memory used",              "—", "—", "Grid size in bytes"),
            ("alive_cells",     "Final alive cells",        "—", "—", "Population at last gen"),
            ("results_match",   "Results match?",           "—", "—", "CPU grid == GPU grid"),
            ("hardware",        "Hardware info",            "—", "—", "CPU / GPU names"),
        ]
        self._tree_iids = {}
        for key, metric, cpu_v, gpu_v, desc in self._metric_rows:
            iid = self.tree.insert("", "end", values=(metric, cpu_v, gpu_v, desc))
            self._tree_iids[key] = iid

    def _build_control_buttons(self):
        """Pause, Reset, Export CSV, Save Graphs."""
        btn_frame = tk.Frame(self.main_frame, bg=BG_DARK)
        btn_frame.pack(fill="x", padx=20, pady=(6, 16))

        btn_style = dict(
            font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
            bg=BTN_BG, fg=TEXT_PRIMARY, activebackground=BTN_HOVER,
            activeforeground=TEXT_PRIMARY, padx=14, pady=6
        )

        self.btn_pause = tk.Button(btn_frame, text="⏸  Pause", command=self._toggle_pause, **btn_style)
        self.btn_pause.pack(side="left", padx=(0, 8))

        self.btn_reset = tk.Button(btn_frame, text="🔄  Reset", command=self._reset_simulation, **btn_style)
        self.btn_reset.pack(side="left", padx=(0, 8))

        self.btn_csv = tk.Button(btn_frame, text="📄  Export CSV", command=self._export_csv, **btn_style)
        self.btn_csv.pack(side="left", padx=(0, 8))

        self.btn_save = tk.Button(btn_frame, text="💾  Save Graphs", command=self._save_graphs, **btn_style)
        self.btn_save.pack(side="left")

        # Status label
        self.lbl_status = tk.Label(
            btn_frame, text="⏳ Simulation running...",
            font=("Segoe UI", 9, "italic"), fg=ACCENT_ORANGE, bg=BG_DARK
        )
        self.lbl_status.pack(side="right")

    # ───────────────────── SIMULATION LOGIC ─────────────────────

    def _start_simulation(self):
        """Launch simulation in a background thread."""
        self.is_running = True
        self.is_finished = False
        self.is_paused = False
        self._stop_event.clear()
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def _simulation_loop(self):
        """
        Run CPU and CUDA step-by-step in lockstep.
        Posts results back to the UI thread via root.after().
        """
        config = self.config
        initial = self.initial_grid.copy()
        run_cpu = config.run_mode in (RunMode.BOTH.value, RunMode.CPU_ONLY.value)
        run_gpu = config.run_mode in (RunMode.BOTH.value, RunMode.GPU_ONLY.value) and self.cuda_available

        # Create engines
        cpu_engine = CPUEngine(edge_behavior=config.edge_behavior) if run_cpu else None
        gpu_engine = None
        if run_gpu:
            try:
                gpu_engine = CUDAEngine(
                    edge_behavior=config.edge_behavior,
                    block_size=config.cuda_block_size
                )
            except Exception:
                gpu_engine = None

        # Set up stepwise generators
        cpu_gen = cpu_engine.run_stepwise(initial, config.iterations) if cpu_engine else None
        gpu_gen = gpu_engine.run_stepwise(initial, config.iterations) if gpu_engine else None

        # Measure total GPU time (for transfer overhead calculation)
        gpu_total_start = time.perf_counter() if gpu_engine else None

        for step_idx in range(1, config.iterations + 1):
            # Handle pause
            while self.is_paused and not self._stop_event.is_set():
                time.sleep(0.05)
            if self._stop_event.is_set():
                return

            cpu_grid = None
            gpu_grid = None
            cpu_ms = 0.0
            gpu_ms = 0.0

            # CPU step
            if cpu_gen:
                try:
                    _, cpu_grid, cpu_ms = next(cpu_gen)
                except StopIteration:
                    cpu_gen = None

            # GPU step
            if gpu_gen:
                try:
                    _, gpu_grid, gpu_ms = next(gpu_gen)
                except StopIteration:
                    gpu_gen = None

            # Post update to UI thread
            self.win.after(
                0,
                self._on_step_complete,
                step_idx, cpu_grid, gpu_grid, cpu_ms, gpu_ms
            )

            # Small throttle so UI can breathe (skip if grid is large → steps are slow enough)
            if cpu_ms + gpu_ms < 5:
                time.sleep(0.008)

        # Record total GPU time
        gpu_total_sec = (time.perf_counter() - gpu_total_start) if gpu_total_start else None

        # Finalize on UI thread
        self.win.after(0, self._on_simulation_done, gpu_total_sec)

    def _on_step_complete(self, gen: int, cpu_grid, gpu_grid, cpu_ms: float, gpu_ms: float):
        """Called on the main thread after each generation step."""
        if self._stop_event.is_set():
            return

        self.generation = gen

        # Store timing
        if cpu_grid is not None:
            self.cpu_step_times.append(cpu_ms)
            self.cpu_current_grid = cpu_grid
        if gpu_grid is not None:
            self.gpu_step_times.append(gpu_ms)
            self.gpu_current_grid = gpu_grid

        # Update canvases
        self._render_grid(self.cpu_canvas, cpu_grid, self.cpu_photo, "cpu")
        self._render_grid(self.gpu_canvas, gpu_grid, self.gpu_photo, "gpu")

        # Update gen labels
        cpu_ms_str = f"{cpu_ms:.2f}" if cpu_grid is not None else "—"
        gpu_ms_str = f"{gpu_ms:.2f}" if gpu_grid is not None else "—"
        self.lbl_cpu_gen.config(text=f"Gen: {gen}  |  {cpu_ms_str} ms/step")
        self.lbl_gpu_gen.config(text=f"Gen: {gen}  |  {gpu_ms_str} ms/step")

        # Update progress
        self.progress_bar["value"] = gen
        pct = (gen / self.config.iterations) * 100
        self.lbl_progress.config(text=f"Generation {gen} / {self.config.iterations}  ({pct:.0f}%)")

        # Update graph every few steps (avoid matplotlib overhead every single step)
        if gen % max(1, self.config.iterations // 100) == 0 or gen == self.config.iterations:
            self._update_graph()

    def _render_grid(self, canvas: tk.Canvas, grid, photo_ref, tag: str):
        """Render a grid array onto a Tkinter canvas."""
        if grid is None:
            return
        try:
            pil_img = render_preview_image(grid, target_size=self.CANVAS_SIZE)
            photo = ImageTk.PhotoImage(pil_img)
            canvas.delete("all")
            canvas.create_image(
                self.CANVAS_SIZE // 2, self.CANVAS_SIZE // 2, image=photo
            )
            # Keep reference to prevent garbage collection
            if tag == "cpu":
                self.cpu_photo = photo
            else:
                self.gpu_photo = photo
        except Exception:
            pass

    def _update_graph(self):
        """Redraw the matplotlib chart with current timing data."""
        try:
            if self.cpu_step_times:
                x_cpu = list(range(1, len(self.cpu_step_times) + 1))
                self.line_cpu.set_data(x_cpu, self.cpu_step_times)

            if self.gpu_step_times:
                x_gpu = list(range(1, len(self.gpu_step_times) + 1))
                self.line_gpu.set_data(x_gpu, self.gpu_step_times)

            # Auto-scale Y axis
            all_times = self.cpu_step_times + self.gpu_step_times
            if all_times:
                y_max = max(all_times) * 1.2
                y_min = 0
                self.ax.set_ylim(y_min, max(y_max, 0.1))

            x_max = max(len(self.cpu_step_times), len(self.gpu_step_times), 10)
            self.ax.set_xlim(0, max(x_max, self.config.iterations))

            self.fig_canvas.draw_idle()
        except Exception:
            pass

    def _on_simulation_done(self, gpu_total_sec: Optional[float]):
        """Finalize: compute derived stats and populate the table."""
        self.is_running = False
        self.is_finished = True

        # Build result
        r = self.result
        r.per_step_times_cpu = self.cpu_step_times[:]
        r.per_step_times_gpu = self.gpu_step_times[:]
        r.cuda_total_time_sec = gpu_total_sec

        # Final alive cells
        if self.cpu_current_grid is not None:
            r.final_alive_cpu = int(np.sum(self.cpu_current_grid))
        if self.gpu_current_grid is not None:
            r.final_alive_gpu = int(np.sum(self.gpu_current_grid))

        # Grids match?
        if self.cpu_current_grid is not None and self.gpu_current_grid is not None:
            r.grids_match = bool(np.array_equal(self.cpu_current_grid, self.gpu_current_grid))

        # Hardware info
        hw = get_hardware_info()
        r.cpu_name = hw["cpu_name"]
        r.gpu_name = hw["gpu_name"]
        r.cuda_version = hw["cuda_version"]

        r.compute_derived_stats()

        # Final graph update
        self._update_graph()

        # Populate table
        self._populate_stats_table()

        # Status
        speedup_str = ""
        if r.kernel_speedup is not None:
            speedup_str = f"  |  Speedup: {r.kernel_speedup:.2f}×"
        self.lbl_status.config(
            text=f"✅ Done in {(r.cpu_time_sec or 0)*1000 + (r.cuda_kernel_time_sec or 0)*1000:.0f} ms total{speedup_str}",
            fg=ACCENT_GREEN
        )
        self.btn_pause.config(state="disabled")

    def _populate_stats_table(self):
        """Fill the Treeview with computed benchmark metrics."""
        r = self.result

        def _update_row(key, cpu_val, gpu_val, desc=None):
            iid = self._tree_iids.get(key)
            if iid:
                old = self.tree.item(iid, "values")
                metric = old[0]
                d = desc if desc else old[3]
                self.tree.item(iid, values=(metric, cpu_val, gpu_val, d))

        # 1. Total time
        _update_row(
            "total_time",
            _safe_fmt(r.cpu_time_sec and r.cpu_time_sec * 1000, ".2f", " ms"),
            _safe_fmt(r.cuda_kernel_time_sec and r.cuda_kernel_time_sec * 1000, ".2f", " ms"),
            f"Whole run, {r.iterations} generations"
        )

        # 2. Average ms/gen
        _update_row(
            "avg_ms",
            _safe_fmt(r.avg_ms_per_gen_cpu, ".3f", " ms"),
            _safe_fmt(r.avg_ms_per_gen_gpu, ".3f", " ms"),
            f"Total ÷ {r.iterations}"
        )

        # 3. Min / Max
        cpu_mm = f"{_safe_fmt(r.min_ms_per_gen_cpu, '.3f')} / {_safe_fmt(r.max_ms_per_gen_cpu, '.3f')} ms" if r.min_ms_per_gen_cpu is not None else "N/A"
        gpu_mm = f"{_safe_fmt(r.min_ms_per_gen_gpu, '.3f')} / {_safe_fmt(r.max_ms_per_gen_gpu, '.3f')} ms" if r.min_ms_per_gen_gpu is not None else "N/A"
        _update_row("min_max_ms", cpu_mm, gpu_mm, "Shows consistency")

        # 4. Throughput
        cpu_tp = f"{r.throughput_cpu:,.0f} cells/s" if r.throughput_cpu else "N/A"
        gpu_tp = f"{r.throughput_gpu:,.0f} cells/s" if r.throughput_gpu else "N/A"
        _update_row("throughput", cpu_tp, gpu_tp, "Fair across grid sizes")

        # 5. Speedup
        speedup_str = _safe_fmt(r.kernel_speedup, ".2f", "× faster") if r.kernel_speedup else "N/A"
        _update_row("speedup", "baseline", speedup_str, "CPU time ÷ GPU time")

        # 6. GPU transfer time
        transfer_str = _safe_fmt(r.gpu_transfer_time_sec and r.gpu_transfer_time_sec * 1000, ".2f", " ms")
        _update_row("gpu_transfer", "—", transfer_str, "CPU ↔ GPU data copy")

        # 7. GPU kernel time
        kernel_str = _safe_fmt(r.cuda_kernel_time_sec and r.cuda_kernel_time_sec * 1000, ".2f", " ms")
        _update_row("gpu_kernel", "—", kernel_str, "Pure compute on GPU")

        # 8. Memory
        cpu_mem = _format_bytes(r.memory_bytes_cpu) if r.memory_bytes_cpu else "N/A"
        gpu_mem = _format_bytes(r.memory_bytes_gpu) if r.memory_bytes_gpu else "N/A"
        _update_row("memory", cpu_mem, gpu_mem, "Grid size (uint8)")

        # 9. Final alive cells
        cpu_alive = f"{r.final_alive_cpu:,}" if r.final_alive_cpu is not None else "N/A"
        gpu_alive = f"{r.final_alive_gpu:,}" if r.final_alive_gpu is not None else "N/A"
        _update_row("alive_cells", cpu_alive, gpu_alive, f"Population at gen {r.iterations}")

        # 10. Results match
        if r.grids_match is not None:
            match_str = "✅ MATCH" if r.grids_match else "❌ MISMATCH"
            _update_row("results_match", match_str, match_str, "Proves both are correct")
        else:
            _update_row("results_match", "—", "—", "Only one engine ran")

        # 11. Hardware
        _update_row("hardware", r.cpu_name or "N/A", f"{r.gpu_name} ({r.cuda_version})", "Needed in any report")

    # ────────────────────── CONTROL BUTTONS ──────────────────────

    def _toggle_pause(self):
        """Toggle pause/resume of the running simulation."""
        if self.is_finished:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶  Resume")
            self.lbl_status.config(text="⏸ Paused", fg=ACCENT_ORANGE)
        else:
            self.btn_pause.config(text="⏸  Pause")
            self.lbl_status.config(text="⏳ Simulation running...", fg=ACCENT_ORANGE)

    def _reset_simulation(self):
        """Stop current simulation and restart from generation 0."""
        # Stop existing thread
        self._stop_event.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2)

        # Reset state
        self.generation = 0
        self.cpu_step_times.clear()
        self.gpu_step_times.clear()
        self.cpu_current_grid = None
        self.gpu_current_grid = None
        self.is_finished = False

        # Reset result
        self.result = BenchmarkResult(
            grid_size=self.config.grid_size,
            iterations=self.config.iterations,
            edge_behavior=self.config.edge_behavior,
            block_size_str=self.config.block_size_str,
        )

        # Clear UI
        self.cpu_canvas.delete("all")
        self.gpu_canvas.delete("all")
        self.progress_bar["value"] = 0
        self.lbl_progress.config(text=f"Generation 0 / {self.config.iterations}  (0%)")
        self.lbl_cpu_gen.config(text="Gen: 0  |  — ms/step")
        self.lbl_gpu_gen.config(text="Gen: 0  |  — ms/step")

        # Reset graph
        self.line_cpu.set_data([], [])
        self.line_gpu.set_data([], [])
        self.ax.set_ylim(0, 1)
        self.fig_canvas.draw_idle()

        # Clear table rows
        for key, _, cpu_v, gpu_v, desc in self._metric_rows:
            iid = self._tree_iids.get(key)
            if iid:
                old = self.tree.item(iid, "values")
                self.tree.item(iid, values=(old[0], "—", "—", old[3]))

        self.btn_pause.config(text="⏸  Pause", state="normal")
        self.lbl_status.config(text="⏳ Simulation running...", fg=ACCENT_ORANGE)

        # Re-generate initial grid (in case pattern has randomness)
        self.initial_grid = create_initial_grid(self.config)

        # Restart
        self._start_simulation()

    def _export_csv(self):
        """Save stats table and per-step timings to a CSV file."""
        if not self.is_finished:
            messagebox.showwarning("Export CSV", "Please wait for the simulation to complete before exporting.", parent=self.win)
            return

        filepath = filedialog.asksaveasfilename(
            parent=self.win,
            title="Export Benchmark Results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"gol_benchmark_{self.config.grid_size}x{self.config.grid_size}.csv"
        )
        if not filepath:
            return

        try:
            r = self.result
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)

                # Section 1: Summary Table
                writer.writerow(["=== Benchmark Summary ==="])
                writer.writerow(["Metric", "CPU (Python)", "CUDA (GPU)", "Description"])
                for iid_key, metric_name, _, _, desc in self._metric_rows:
                    iid = self._tree_iids.get(iid_key)
                    if iid:
                        vals = self.tree.item(iid, "values")
                        writer.writerow(vals)
                writer.writerow([])

                # Section 2: Per-step timing data
                writer.writerow(["=== Per-Generation Timing (ms) ==="])
                writer.writerow(["Generation", "CPU ms/step", "GPU ms/step"])
                max_steps = max(len(r.per_step_times_cpu), len(r.per_step_times_gpu))
                for i in range(max_steps):
                    cpu_v = f"{r.per_step_times_cpu[i]:.4f}" if i < len(r.per_step_times_cpu) else ""
                    gpu_v = f"{r.per_step_times_gpu[i]:.4f}" if i < len(r.per_step_times_gpu) else ""
                    writer.writerow([i + 1, cpu_v, gpu_v])

            self.lbl_status.config(text=f"📄 CSV exported: {filepath}", fg=ACCENT_GREEN)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{e}", parent=self.win)

    def _save_graphs(self):
        """Save the matplotlib graph as a high-resolution PNG."""
        if not self.cpu_step_times and not self.gpu_step_times:
            messagebox.showwarning("Save Graphs", "No graph data to save yet.", parent=self.win)
            return

        filepath = filedialog.asksaveasfilename(
            parent=self.win,
            title="Save Performance Graph",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")],
            initialfile=f"gol_performance_{self.config.grid_size}x{self.config.grid_size}.png"
        )
        if not filepath:
            return

        try:
            self.fig.savefig(filepath, dpi=200, bbox_inches="tight", facecolor=BG_CARD)
            self.lbl_status.config(text=f"💾 Graph saved: {filepath}", fg=ACCENT_GREEN)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save graph:\n{e}", parent=self.win)

    def _on_close(self):
        """Clean up when the results window is closed."""
        self._stop_event.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2)
        try:
            self.win.destroy()
        except Exception:
            pass


def open_results_window(parent: tk.Tk, config: SimulationConfig) -> ResultsWindow:
    """Convenience function to create and return a ResultsWindow."""
    return ResultsWindow(parent, config)
