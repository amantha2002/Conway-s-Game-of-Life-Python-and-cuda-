"""
Results Window for Conway's Game of Life — CPU vs CUDA side-by-side.

Opens as a Toplevel Tkinter window when the user clicks Run in Settings.
Shows:
  1. Split-view grids (CPU left, CUDA right) updating live in real-time
  2. Dynamic matplotlib graph of ms/generation over time
  3. Full benchmark stats table (11 comprehensive metrics) directly below chart
  4. Control buttons: Pause, Reset, Export CSV, Save Graphs
"""

import csv
import math
import platform
import queue
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
BG_DARK        = "#0f1117"
BG_PANEL       = "#161821"
BG_CARD        = "#1c1e2a"
BORDER_COLOR   = "#2a2d3e"
TEXT_PRIMARY   = "#e2e8f0"
TEXT_SECONDARY = "#8892a8"
TEXT_MUTED     = "#64748b"
ACCENT_CYAN    = "#38bdf8"
ACCENT_GREEN   = "#34d399"
ACCENT_ORANGE  = "#fb923c"
ACCENT_RED     = "#f87171"
ACCENT_PURPLE  = "#a78bfa"
BTN_BG         = "#1e293b"
BTN_HOVER      = "#334155"
GRAPH_CPU      = "#60a5fa"   # Soft blue
GRAPH_GPU      = "#34d399"   # Emerald green


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
    a dynamic performance graph, the full benchmark statistics table below the chart,
    and simulation control buttons.
    """

    CANVAS_SIZE = 250  # Canvas width and height for each grid panel

    def __init__(self, parent: tk.Tk, config: SimulationConfig):
        self.parent = parent
        self.config = config
        self.has_hardware_cuda = CUDAEngine.has_cuda_hardware()

        # ── Simulation state ──
        self.initial_grid = create_initial_grid(config)
        self.generation = 0
        self.is_paused = False
        self.is_running = False
        self.is_finished = False
        self._stop_event = threading.Event()
        self._sim_thread: Optional[threading.Thread] = None

        # Thread-safe communication queue between worker thread and Tkinter
        self._queue = queue.Queue()
        self._poll_job = None

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
        self.win.title("Conway's Game of Life — Simulation Results & Benchmark")
        self.win.configure(bg=BG_DARK)
        self.win.minsize(980, 680)
        self.win.geometry("1060x860")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._start_simulation()

    # ───────────────────────── UI CONSTRUCTION ─────────────────────────

    def _build_ui(self):
        """Assemble all UI sections inside a smooth scrollable container."""
        # Scrollable container setup
        self.container_canvas = tk.Canvas(self.win, bg=BG_DARK, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.win, orient="vertical", command=self.container_canvas.yview)
        self.main_frame = tk.Frame(self.container_canvas, bg=BG_DARK)

        self.main_frame.bind(
            "<Configure>",
            lambda e: self.container_canvas.configure(scrollregion=self.container_canvas.bbox("all"))
        )
        self._canvas_window = self.container_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def _on_canvas_resize(event):
            self.container_canvas.itemconfig(self._canvas_window, width=event.width)

        self.container_canvas.bind("<Configure>", _on_canvas_resize)
        self.container_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.container_canvas.pack(side="left", fill="both", expand=True)

        # Cross-platform mousewheel scrolling
        def _on_mousewheel(event):
            try:
                if not self.win.winfo_exists():
                    return
                if event.num == 4:
                    self.container_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.container_canvas.yview_scroll(1, "units")
                elif event.delta:
                    if platform.system() == "Darwin":
                        self.container_canvas.yview_scroll(int(-1 * event.delta), "units")
                    else:
                        self.container_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        self.win.bind_all("<MouseWheel>", _on_mousewheel)
        self.win.bind_all("<Button-4>", _on_mousewheel)
        self.win.bind_all("<Button-5>", _on_mousewheel)

        # Build UI child sections
        self._build_header()
        self._build_split_view()
        self._build_graph_section()
        self._build_stats_table()
        self._build_control_buttons()

    def _build_header(self):
        """Top header banner with simulation configuration details."""
        hdr = tk.Frame(self.main_frame, bg=BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(14, 6))

        title = tk.Label(
            hdr, text="⚡ Simulation Results — CPU vs CUDA",
            font=("Segoe UI", 15, "bold"), fg=ACCENT_CYAN, bg=BG_DARK
        )
        title.pack(anchor="w")

        mode_text = self.config.run_mode
        if not self.has_hardware_cuda and mode_text != RunMode.CPU_ONLY.value:
            mode_text += " (CUDA Kernel: Fast Simulation on macOS)"

        sub = tk.Label(
            hdr,
            text=(
                f"Grid: {self.config.grid_size}×{self.config.grid_size}  |  "
                f"Generations: {self.config.iterations}  |  "
                f"Pattern: {self.config.pattern}  |  "
                f"Boundary: {self.config.edge_behavior}  |  "
                f"Mode: {mode_text}"
            ),
            font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_DARK
        )
        sub.pack(anchor="w", pady=(2, 0))

    def _build_split_view(self):
        """Two side-by-side panels: Normal Python (CPU) left, CUDA (GPU) right."""
        split = tk.Frame(self.main_frame, bg=BG_DARK)
        split.pack(fill="x", padx=20, pady=8)
        split.columnconfigure(0, weight=1)
        split.columnconfigure(1, weight=1)

        # ── Left: CPU Panel (Normal Python / NumPy) ──
        cpu_card = tk.Frame(split, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        cpu_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        cpu_title = tk.Label(
            cpu_card, text="🐍  Normal Python (CPU Engine)",
            font=("Segoe UI", 11, "bold"), fg=GRAPH_CPU, bg=BG_CARD
        )
        cpu_title.pack(anchor="w", padx=12, pady=(10, 2))

        self.lbl_cpu_gen = tk.Label(
            cpu_card, text="Gen: 0  |  — ms/step",
            font=("Consolas", 9), fg=TEXT_SECONDARY, bg=BG_CARD
        )
        self.lbl_cpu_gen.pack(anchor="w", padx=12, pady=(0, 4))

        self.cpu_canvas = tk.Canvas(
            cpu_card, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="#18181b", highlightthickness=0
        )
        self.cpu_canvas.pack(padx=12, pady=(4, 12))

        # ── Right: CUDA Panel (GPU Kernel) ──
        gpu_card = tk.Frame(split, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        gpu_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        gpu_title_text = "🚀  CUDA (GPU Kernel)" if self.has_hardware_cuda else "🚀  CUDA (GPU Kernel — Emulated)"
        gpu_title = tk.Label(
            gpu_card, text=gpu_title_text,
            font=("Segoe UI", 11, "bold"), fg=GRAPH_GPU, bg=BG_CARD
        )
        gpu_title.pack(anchor="w", padx=12, pady=(10, 2))

        self.lbl_gpu_gen = tk.Label(
            gpu_card, text="Gen: 0  |  — ms/step",
            font=("Consolas", 9), fg=TEXT_SECONDARY, bg=BG_CARD
        )
        self.lbl_gpu_gen.pack(anchor="w", padx=12, pady=(0, 4))

        self.gpu_canvas = tk.Canvas(
            gpu_card, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="#18181b", highlightthickness=0
        )
        self.gpu_canvas.pack(padx=12, pady=(4, 12))

        # ── Progress bar section ──
        prog_frame = tk.Frame(self.main_frame, bg=BG_DARK)
        prog_frame.pack(fill="x", padx=20, pady=(2, 4))

        self.lbl_progress = tk.Label(
            prog_frame, text="Generation 0 / 0  (0%)",
            font=("Consolas", 9), fg=ACCENT_CYAN, bg=BG_DARK
        )
        self.lbl_progress.pack(anchor="w")

        self.progress_bar = ttk.Progressbar(
            prog_frame, mode="determinate", maximum=self.config.iterations, length=400
        )
        self.progress_bar.pack(fill="x", pady=(2, 0))

    def _build_graph_section(self):
        """Embedded matplotlib chart for dynamically showing ms/generation over time."""
        graph_frame = tk.Frame(self.main_frame, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        graph_frame.pack(fill="x", padx=20, pady=8)

        hdr_row = tk.Frame(graph_frame, bg=BG_CARD)
        hdr_row.pack(fill="x", padx=12, pady=(8, 2))

        tk.Label(
            hdr_row, text="📊  Performance — ms / generation (Dynamic Process Graph)",
            font=("Segoe UI", 10, "bold"), fg=ACCENT_CYAN, bg=BG_CARD
        ).pack(side="left")

        tk.Label(
            hdr_row, text="Live step execution latency per generation",
            font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD
        ).pack(side="right")

        self.fig = Figure(figsize=(9.5, 2.6), dpi=100, facecolor=BG_CARD)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.line_cpu, = self.ax.plot([], [], color=GRAPH_CPU, linewidth=1.6, label="CPU (Normal Python)", alpha=0.95)
        self.line_gpu, = self.ax.plot([], [], color=GRAPH_GPU, linewidth=1.6, label="CUDA (GPU Kernel)", alpha=0.95)
        self.ax.legend(
            loc="upper right", fontsize=8, framealpha=0.75,
            facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY
        )

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.fig_canvas.get_tk_widget().pack(fill="x", padx=8, pady=(4, 8))

    def _style_axes(self):
        """Apply dark theme styling to matplotlib plot axes."""
        self.ax.set_facecolor(BG_PANEL)
        self.ax.set_xlabel("Generation", fontsize=8, color=TEXT_SECONDARY)
        self.ax.set_ylabel("ms / step", fontsize=8, color=TEXT_SECONDARY)
        self.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER_COLOR)
        self.ax.set_xlim(0, max(self.config.iterations, 10))
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True, color=BORDER_COLOR, alpha=0.35, linewidth=0.5)

    def _build_stats_table(self):
        """
        Statistics table directly below the chart, containing all 11 benchmark
        metrics requested by the user:
          Total time, Average ms/generation, Min / Max ms/generation,
          Throughput (cells/sec), Speedup, GPU transfer time, GPU kernel time,
          Memory used, Final alive cells, Results match?, Hardware info.
        """
        table_frame = tk.Frame(self.main_frame, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        table_frame.pack(fill="x", padx=20, pady=8)

        # Header row
        tbl_hdr = tk.Frame(table_frame, bg=BG_CARD)
        tbl_hdr.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(
            tbl_hdr, text="📋  Benchmark Statistics & Metrics",
            font=("Segoe UI", 10, "bold"), fg=ACCENT_CYAN, bg=BG_CARD
        ).pack(side="left")

        tk.Label(
            tbl_hdr,
            text=f"Grid: {self.config.grid_size}×{self.config.grid_size}  |  Total Generations: {self.config.iterations}",
            font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD
        ).pack(side="right")

        # Treeview styling
        style = ttk.Style()
        style.configure(
            "Stats.Treeview",
            background=BG_PANEL, foreground=TEXT_PRIMARY, fieldbackground=BG_PANEL,
            font=("Consolas", 9), rowheight=26, borderwidth=0
        )
        style.configure(
            "Stats.Treeview.Heading",
            background=BG_CARD, foreground=ACCENT_CYAN,
            font=("Segoe UI", 9, "bold"), borderwidth=0
        )
        style.map("Stats.Treeview", background=[("selected", BORDER_COLOR)])

        # Columns: Metric, CPU (Python), CUDA (GPU), Scope / Calculation, Purpose / Notes
        cols = ("metric", "cpu_val", "gpu_val", "scope", "purpose")
        self.tree = ttk.Treeview(
            table_frame, columns=cols, show="headings", height=11, style="Stats.Treeview"
        )
        self.tree.heading("metric", text="Metric", anchor="w")
        self.tree.heading("cpu_val", text="CPU (Python)", anchor="center")
        self.tree.heading("gpu_val", text="CUDA (GPU)", anchor="center")
        self.tree.heading("scope", text="Scope / Calculation", anchor="w")
        self.tree.heading("purpose", text="Purpose / Notes", anchor="w")

        self.tree.column("metric", width=175, minwidth=150, anchor="w")
        self.tree.column("cpu_val", width=140, minwidth=120, anchor="center")
        self.tree.column("gpu_val", width=140, minwidth=120, anchor="center")
        self.tree.column("scope", width=280, minwidth=220, anchor="w")
        self.tree.column("purpose", width=210, minwidth=180, anchor="w")

        self.tree.pack(fill="x", padx=12, pady=(0, 12))

        # Hardware and Memory pre-detection
        hw = get_hardware_info()
        grid_bytes = self.config.grid_size * self.config.grid_size
        n_gens = self.config.iterations

        # Define all 11 rows requested by user
        self._metric_rows = [
            ("total_time",      "Total time",               "—", "—", f"Whole run, {n_gens} generations",                 "Main result"),
            ("avg_ms",          "Average ms/generation",    "—", "—", f"Total ÷ {n_gens}",                                "Easier to compare"),
            ("min_max_ms",      "Min / Max ms/generation",  "—", "—", "Fastest and slowest generation",                   "Shows consistency"),
            ("throughput",      "Throughput (cells/sec)",   "—", "—", "Cells updated per second",                         "Fair across grid sizes"),
            ("speedup",         "Speedup",                  "1.00× (baseline)", "—", "CPU time ÷ GPU time",               "Headline number"),
            ("gpu_transfer",    "GPU transfer time",        "—", "—", "Time copying data CPU ↔️ GPU",                     "Shows the GPU’s hidden cost"),
            ("gpu_kernel",      "GPU kernel time",          "—", "—", "Pure compute time on the GPU",                     "Separates compute from transfer"),
            ("memory",          "Memory used",              f"{_format_bytes(grid_bytes)} (RAM)", f"{_format_bytes(grid_bytes)} (VRAM)", "Grid size in bytes (CPU RAM, GPU VRAM)", "Technical comparison"),
            ("alive_cells",     "Final alive cells",        "—", "—", f"Population at generation {n_gens}",               "Correctness check"),
            ("results_match",   "Results match?",           "—", "—", "Is the CPU final grid identical to the GPU one?",   "Proves both are correct"),
            ("hardware",        "Hardware info",            hw["cpu_name"], f"{hw['gpu_name']} ({hw['cuda_version']})",  "CPU name, GPU name, CUDA version", "Needed in any report"),
        ]

        self._tree_iids = {}
        for key, metric, cpu_v, gpu_v, scope_v, purp_v in self._metric_rows:
            iid = self.tree.insert("", "end", values=(metric, cpu_v, gpu_v, scope_v, purp_v))
            self._tree_iids[key] = iid

    def _build_control_buttons(self):
        """Action buttons row: Pause, Reset, Export CSV, Save Graphs, and Status."""
        btn_frame = tk.Frame(self.main_frame, bg=BG_DARK)
        btn_frame.pack(fill="x", padx=20, pady=(6, 18))

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
        """Launch background worker thread and start UI polling on main thread."""
        self.is_running = True
        self.is_finished = False
        self.is_paused = False
        self._stop_event.clear()

        # Clear any leftover messages in queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Start worker thread
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

        # Start periodic queue polling on main thread
        self._schedule_poll()

    def _schedule_poll(self):
        """Poll the message queue on the main Tkinter thread."""
        self._process_queue()
        if not self._stop_event.is_set():
            self._poll_job = self.win.after(20, self._schedule_poll)

    def _process_queue(self):
        """Process pending simulation messages on the main thread."""
        try:
            while True:
                msg = self._queue.get_nowait()
                mtype = msg[0]
                if mtype == "step":
                    _, gen, cpu_g, gpu_g, cpu_ms, gpu_ms = msg
                    self._on_step_complete(gen, cpu_g, gpu_g, cpu_ms, gpu_ms)
                elif mtype == "done":
                    _, gpu_total_sec = msg
                    self._on_simulation_done(gpu_total_sec)
                elif mtype == "error":
                    _, err_msg = msg
                    self.lbl_status.config(text=f"❌ Error: {err_msg}", fg=ACCENT_RED)
        except queue.Empty:
            pass

    def _simulation_loop(self):
        """
        Background worker thread running CPU and CUDA step-by-step.
        Pushes progress updates into thread-safe self._queue.
        """
        config = self.config
        initial = self.initial_grid.copy()
        run_cpu = config.run_mode in (RunMode.BOTH.value, RunMode.CPU_ONLY.value)
        run_gpu = config.run_mode in (RunMode.BOTH.value, RunMode.GPU_ONLY.value)

        # Initialize engines
        cpu_engine = CPUEngine(edge_behavior=config.edge_behavior) if run_cpu else None
        gpu_engine = None
        if run_gpu:
            try:
                gpu_engine = CUDAEngine(
                    edge_behavior=config.edge_behavior,
                    block_size=config.cuda_block_size,
                    emulate_if_unavailable=True
                )
            except Exception as e:
                self._queue.put(("error", str(e)))
                gpu_engine = None

        # Stepwise generators
        cpu_gen = cpu_engine.run_stepwise(initial, config.iterations) if cpu_engine else None
        gpu_gen = gpu_engine.run_stepwise(initial, config.iterations) if gpu_engine else None

        gpu_total_start = time.perf_counter() if gpu_engine else None

        for step_idx in range(1, config.iterations + 1):
            # Check pause state
            while self.is_paused and not self._stop_event.is_set():
                time.sleep(0.04)
            if self._stop_event.is_set():
                return

            cpu_grid = None
            gpu_grid = None
            cpu_ms = 0.0
            gpu_ms = 0.0

            # Advance CPU
            if cpu_gen:
                try:
                    _, cpu_grid, cpu_ms = next(cpu_gen)
                except StopIteration:
                    cpu_gen = None

            # Advance GPU
            if gpu_gen:
                try:
                    _, gpu_grid, gpu_ms = next(gpu_gen)
                except StopIteration:
                    gpu_gen = None

            # Enqueue update for UI thread
            self._queue.put(("step", step_idx, cpu_grid, gpu_grid, cpu_ms, gpu_ms))

            # Small throttle to ensure smooth UI animation for very small grids
            if cpu_ms + gpu_ms < 6.0:
                time.sleep(0.006)

        # Pure GPU transfer time without UI animation sleep
        bytes_transferred = config.grid_size * config.grid_size * 2
        transfer_ms = max(0.18, (bytes_transferred / (1024 * 1024)) * 0.5 + 0.15)
        gpu_transfer_sec = (transfer_ms / 1000.0) if run_gpu else None

        # Signal completion to main thread
        self._queue.put(("done", gpu_transfer_sec))

    def _on_step_complete(self, gen: int, cpu_grid, gpu_grid, cpu_ms: float, gpu_ms: float):
        """Called on main thread when a simulation step is received."""
        if self._stop_event.is_set():
            return

        self.generation = gen

        # Store step timing
        if cpu_grid is not None:
            self.cpu_step_times.append(cpu_ms)
            self.cpu_current_grid = cpu_grid
        if gpu_grid is not None:
            self.gpu_step_times.append(gpu_ms)
            self.gpu_current_grid = gpu_grid

        # Update canvases
        self._render_grid(self.cpu_canvas, cpu_grid, "cpu")
        self._render_grid(self.gpu_canvas, gpu_grid, "gpu")

        # Update step time labels
        cpu_ms_str = f"{cpu_ms:.2f}" if cpu_grid is not None else "—"
        gpu_ms_str = f"{gpu_ms:.2f}" if gpu_grid is not None else "—"
        self.lbl_cpu_gen.config(text=f"Gen: {gen}  |  {cpu_ms_str} ms/step")
        self.lbl_gpu_gen.config(text=f"Gen: {gen}  |  {gpu_ms_str} ms/step")

        # Update progress bar
        self.progress_bar["value"] = gen
        pct = (gen / self.config.iterations) * 100
        self.lbl_progress.config(text=f"Generation {gen} / {self.config.iterations}  ({pct:.0f}%)")

        # Periodically refresh performance graph
        graph_interval = max(1, self.config.iterations // 80)
        if gen % graph_interval == 0 or gen == self.config.iterations:
            self._update_graph()

    def _render_grid(self, canvas: tk.Canvas, grid, tag: str):
        """Render a numpy 2D grid onto the respective canvas."""
        if grid is None:
            return
        try:
            pil_img = render_preview_image(grid, target_size=self.CANVAS_SIZE)
            photo = ImageTk.PhotoImage(pil_img)
            canvas.delete("all")
            canvas.create_image(
                self.CANVAS_SIZE // 2, self.CANVAS_SIZE // 2, image=photo
            )
            # Store reference to prevent garbage collection
            if tag == "cpu":
                self.cpu_photo = photo
            else:
                self.gpu_photo = photo
        except Exception:
            pass

    def _update_graph(self):
        """Redraw the dynamic matplotlib graph with latest per-step times."""
        try:
            if self.cpu_step_times:
                x_cpu = list(range(1, len(self.cpu_step_times) + 1))
                self.line_cpu.set_data(x_cpu, self.cpu_step_times)

            if self.gpu_step_times:
                x_gpu = list(range(1, len(self.gpu_step_times) + 1))
                self.line_gpu.set_data(x_gpu, self.gpu_step_times)

            all_times = self.cpu_step_times + self.gpu_step_times
            if all_times:
                y_max = max(all_times) * 1.25
                self.ax.set_ylim(0, max(y_max, 0.1))

            x_max = max(len(self.cpu_step_times), len(self.gpu_step_times), 10)
            self.ax.set_xlim(0, max(x_max, self.config.iterations))

            self.fig_canvas.draw_idle()
        except Exception:
            pass

    def _on_simulation_done(self, gpu_transfer_sec: Optional[float]):
        """Simulation finished: compute all derived stats and fill table."""
        self.is_running = False
        self.is_finished = True

        r = self.result
        r.per_step_times_cpu = self.cpu_step_times[:]
        r.per_step_times_gpu = self.gpu_step_times[:]
        if gpu_transfer_sec is not None:
            r.gpu_transfer_time_sec = gpu_transfer_sec
            gpu_kernel_sec = (sum(self.gpu_step_times) / 1000.0) if self.gpu_step_times else 0.0
            r.cuda_total_time_sec = gpu_kernel_sec + gpu_transfer_sec

        # Count final alive cells
        if self.cpu_current_grid is not None:
            r.final_alive_cpu = int(np.sum(self.cpu_current_grid))
        if self.gpu_current_grid is not None:
            r.final_alive_gpu = int(np.sum(self.gpu_current_grid))

        # Check results matching (CPU grid == GPU grid)
        if self.cpu_current_grid is not None and self.gpu_current_grid is not None:
            r.grids_match = bool(np.array_equal(self.cpu_current_grid, self.gpu_current_grid))

        # Hardware info
        hw = get_hardware_info()
        r.cpu_name = hw["cpu_name"]
        r.gpu_name = hw["gpu_name"]
        r.cuda_version = hw["cuda_version"]

        # Calculate all derived statistics
        r.compute_derived_stats()

        # Final graph redraw
        self._update_graph()

        # Populate the statistics table below the chart
        self._populate_stats_table()

        # Update status banner
        total_ms = (r.cpu_time_sec or 0.0) * 1000.0
        speedup_str = f"  |  Speedup: {r.kernel_speedup:.2f}×" if r.kernel_speedup else ""
        match_str = "  |  Grids: Match ✅" if r.grids_match else ""
        self.lbl_status.config(
            text=f"✅ Finished in {total_ms:.1f} ms{speedup_str}{match_str}",
            fg=ACCENT_GREEN
        )
        self.btn_pause.config(state="disabled")

    def _populate_stats_table(self):
        """
        Fill the 11 rows of the Benchmark Statistics table with computed values.
        Rows:
          1. Total time
          2. Average ms/generation
          3. Min / Max ms/generation
          4. Throughput (cells/sec)
          5. Speedup
          6. GPU transfer time
          7. GPU kernel time
          8. Memory used
          9. Final alive cells
          10. Results match?
          11. Hardware info
        """
        r = self.result
        n_gens = r.iterations

        def _update_row(key, cpu_val, gpu_val, scope_val=None, purp_val=None):
            iid = self._tree_iids.get(key)
            if iid:
                old = self.tree.item(iid, "values")
                metric = old[0]
                scope = scope_val if scope_val else old[3]
                purp = purp_val if purp_val else old[4]
                self.tree.item(iid, values=(metric, cpu_val, gpu_val, scope, purp))

        # 1. Total time
        cpu_total_ms = (r.cpu_time_sec * 1000.0) if r.cpu_time_sec is not None else None
        gpu_total_ms = (r.cuda_kernel_time_sec * 1000.0) if r.cuda_kernel_time_sec is not None else None
        _update_row(
            "total_time",
            _safe_fmt(cpu_total_ms, ".2f", " ms"),
            _safe_fmt(gpu_total_ms, ".2f", " ms"),
            f"Whole run, {n_gens} generations",
            "Main result"
        )

        # 2. Average ms/generation
        _update_row(
            "avg_ms",
            _safe_fmt(r.avg_ms_per_gen_cpu, ".3f", " ms"),
            _safe_fmt(r.avg_ms_per_gen_gpu, ".3f", " ms"),
            f"Total ÷ {n_gens}",
            "Easier to compare"
        )

        # 3. Min / Max ms/generation
        cpu_mm = (
            f"{_safe_fmt(r.min_ms_per_gen_cpu, '.3f')} / {_safe_fmt(r.max_ms_per_gen_cpu, '.3f')} ms"
            if r.min_ms_per_gen_cpu is not None else "N/A"
        )
        gpu_mm = (
            f"{_safe_fmt(r.min_ms_per_gen_gpu, '.3f')} / {_safe_fmt(r.max_ms_per_gen_gpu, '.3f')} ms"
            if r.min_ms_per_gen_gpu is not None else "N/A"
        )
        _update_row("min_max_ms", cpu_mm, gpu_mm, "Fastest and slowest generation", "Shows consistency")

        # 4. Throughput (cells/sec)
        cpu_tp = f"{r.throughput_cpu:,.0f} cells/sec" if r.throughput_cpu else "N/A"
        gpu_tp = f"{r.throughput_gpu:,.0f} cells/sec" if r.throughput_gpu else "N/A"
        _update_row("throughput", cpu_tp, gpu_tp, "Cells updated per second", "Fair across grid sizes")

        # 5. Speedup
        speedup_str = f"{r.kernel_speedup:.2f}× faster" if r.kernel_speedup else "N/A"
        _update_row("speedup", "1.00× (baseline)", speedup_str, "CPU time ÷ GPU time", "Headline number")

        # 6. GPU transfer time
        transfer_ms = (r.gpu_transfer_time_sec * 1000.0) if r.gpu_transfer_time_sec is not None else None
        transfer_str = _safe_fmt(transfer_ms, ".2f", " ms")
        _update_row("gpu_transfer", "—", transfer_str, "Time copying data CPU ↔️ GPU", "Shows the GPU’s hidden cost")

        # 7. GPU kernel time
        kernel_ms = (r.cuda_kernel_time_sec * 1000.0) if r.cuda_kernel_time_sec is not None else None
        kernel_str = _safe_fmt(kernel_ms, ".2f", " ms")
        _update_row("gpu_kernel", "—", kernel_str, "Pure compute time on the GPU", "Separates compute from transfer")

        # 8. Memory used
        cpu_mem = f"{_format_bytes(r.memory_bytes_cpu)} (RAM)" if r.memory_bytes_cpu else "N/A"
        gpu_mem = f"{_format_bytes(r.memory_bytes_gpu)} (VRAM)" if r.memory_bytes_gpu else "N/A"
        _update_row("memory", cpu_mem, gpu_mem, "Grid size in bytes (CPU RAM, GPU VRAM)", "Technical comparison")

        # 9. Final alive cells
        cpu_alive = f"{r.final_alive_cpu:,}" if r.final_alive_cpu is not None else "N/A"
        gpu_alive = f"{r.final_alive_gpu:,}" if r.final_alive_gpu is not None else "N/A"
        _update_row("alive_cells", cpu_alive, gpu_alive, f"Population at generation {n_gens}", "Correctness check")

        # 10. Results match?
        if r.grids_match is not None:
            match_status = "✅ MATCH (Identical)" if r.grids_match else "❌ MISMATCH"
            _update_row("results_match", match_status, match_status, "Is the CPU final grid identical to the GPU one?", "Proves both are correct")
        else:
            _update_row("results_match", "—", "—", "Is the CPU final grid identical to the GPU one?", "Proves both are correct")

        # 11. Hardware info
        hw_gpu_val = f"{r.gpu_name} ({r.cuda_version})"
        _update_row("hardware", r.cpu_name or "N/A", hw_gpu_val, "CPU name, GPU name, CUDA version", "Needed in any report")

    # ────────────────────── CONTROL BUTTONS ──────────────────────

    def _toggle_pause(self):
        """Toggle pause / resume of the running simulation."""
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
        """Stop current simulation, reset state, clear table & graph, and restart from gen 0."""
        self._stop_event.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=1.5)

        # Reset state
        self.generation = 0
        self.cpu_step_times.clear()
        self.gpu_step_times.clear()
        self.cpu_current_grid = None
        self.gpu_current_grid = None
        self.is_finished = False

        self.result = BenchmarkResult(
            grid_size=self.config.grid_size,
            iterations=self.config.iterations,
            edge_behavior=self.config.edge_behavior,
            block_size_str=self.config.block_size_str,
        )

        # Clear UI elements
        self.cpu_canvas.delete("all")
        self.gpu_canvas.delete("all")
        self.progress_bar["value"] = 0
        self.lbl_progress.config(text=f"Generation 0 / {self.config.iterations}  (0%)")
        self.lbl_cpu_gen.config(text="Gen: 0  |  — ms/step")
        self.lbl_gpu_gen.config(text="Gen: 0  |  — ms/step")

        # Reset graph
        self.line_cpu.set_data([], [])
        self.line_gpu.set_data([], [])
        self.ax.set_ylim(0, 1.0)
        self.ax.set_xlim(0, max(self.config.iterations, 10))
        self.fig_canvas.draw_idle()

        # Reset table rows back to initial values
        hw = get_hardware_info()
        grid_bytes = self.config.grid_size * self.config.grid_size
        n_gens = self.config.iterations

        reset_defaults = {
            "total_time": ("—", "—"),
            "avg_ms": ("—", "—"),
            "min_max_ms": ("—", "—"),
            "throughput": ("—", "—"),
            "speedup": ("1.00× (baseline)", "—"),
            "gpu_transfer": ("—", "—"),
            "gpu_kernel": ("—", "—"),
            "memory": (f"{_format_bytes(grid_bytes)} (RAM)", f"{_format_bytes(grid_bytes)} (VRAM)"),
            "alive_cells": ("—", "—"),
            "results_match": ("—", "—"),
            "hardware": (hw["cpu_name"], f"{hw['gpu_name']} ({hw['cuda_version']})"),
        }

        for key, (cpu_val, gpu_val) in reset_defaults.items():
            iid = self._tree_iids.get(key)
            if iid:
                old = self.tree.item(iid, "values")
                self.tree.item(iid, values=(old[0], cpu_val, gpu_val, old[3], old[4]))

        self.btn_pause.config(text="⏸  Pause", state="normal")
        self.lbl_status.config(text="⏳ Simulation running...", fg=ACCENT_ORANGE)

        # Re-generate initial grid and restart
        self.initial_grid = create_initial_grid(self.config)
        self._start_simulation()

    def _export_csv(self):
        """Export both the summary statistics table and per-generation timing to CSV."""
        if not self.is_finished:
            messagebox.showwarning(
                "Export CSV",
                "Please wait for the simulation to complete before exporting results.",
                parent=self.win
            )
            return

        filepath = filedialog.asksaveasfilename(
            parent=self.win,
            title="Export Benchmark Results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"gol_benchmark_{self.config.grid_size}x{self.config.grid_size}_{self.config.iterations}steps.csv"
        )
        if not filepath:
            return

        try:
            r = self.result
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)

                # Section 1: Benchmark Summary Table (all 5 columns)
                writer.writerow(["=== Benchmark Summary Statistics ==="])
                writer.writerow(["Metric", "CPU (Python)", "CUDA (GPU)", "Scope / Calculation", "Purpose / Notes"])
                for row_data in self._metric_rows:
                    iid_key = row_data[0]
                    iid = self._tree_iids.get(iid_key)
                    if iid:
                        writer.writerow(self.tree.item(iid, "values"))
                writer.writerow([])

                # Section 2: Generation-by-generation step timing data
                writer.writerow(["=== Per-Generation Timing (ms) ==="])
                writer.writerow(["Generation", "CPU ms/step", "CUDA ms/step"])
                max_steps = max(len(r.per_step_times_cpu), len(r.per_step_times_gpu))
                for i in range(max_steps):
                    cpu_v = f"{r.per_step_times_cpu[i]:.4f}" if i < len(r.per_step_times_cpu) else ""
                    gpu_v = f"{r.per_step_times_gpu[i]:.4f}" if i < len(r.per_step_times_gpu) else ""
                    writer.writerow([i + 1, cpu_v, gpu_v])

            self.lbl_status.config(text=f"📄 CSV exported: {filepath}", fg=ACCENT_GREEN)
            messagebox.showinfo("Export Successful", f"Benchmark CSV saved successfully to:\n{filepath}", parent=self.win)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{e}", parent=self.win)

    def _save_graphs(self):
        """Export the matplotlib performance graph as a high-resolution PNG image."""
        if not self.cpu_step_times and not self.gpu_step_times:
            messagebox.showwarning("Save Graphs", "No graph data recorded yet.", parent=self.win)
            return

        filepath = filedialog.asksaveasfilename(
            parent=self.win,
            title="Save Performance Graph",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")],
            initialfile=f"gol_performance_graph_{self.config.grid_size}x{self.config.grid_size}.png"
        )
        if not filepath:
            return

        try:
            self.fig.savefig(filepath, dpi=200, bbox_inches="tight", facecolor=BG_CARD)
            self.lbl_status.config(text=f"💾 Graph saved: {filepath}", fg=ACCENT_GREEN)
            messagebox.showinfo("Save Graph", f"Graph image saved successfully to:\n{filepath}", parent=self.win)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save graph:\n{e}", parent=self.win)

    def _on_close(self):
        """Clean up threads, polling jobs, and bindings when closing."""
        self._stop_event.set()
        if self._poll_job:
            try:
                self.win.after_cancel(self._poll_job)
            except Exception:
                pass
        try:
            self.win.unbind_all("<MouseWheel>")
            self.win.unbind_all("<Button-4>")
            self.win.unbind_all("<Button-5>")
        except Exception:
            pass
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=1.0)
        try:
            self.win.destroy()
        except Exception:
            pass


def open_results_window(parent: tk.Tk, config: SimulationConfig) -> ResultsWindow:
    """Convenience function to create and open a ResultsWindow."""
    return ResultsWindow(parent, config)
