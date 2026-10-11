# 🌌 Conway's Game of Life: High-Performance Computing (CPU vs CUDA)

An assignment project designed to analyze and benchmark **GPU acceleration with NVIDIA CUDA vs standard CPU computing** on 2D cellular automata (Conway's Game of Life).

---

## 🚀 Key Objectives
1. **Parallelism & High-Performance Computing (HPC)**: Demonstrate how 2D grid/thread blocks on an NVIDIA GPU provide massive speedup over CPU execution as grid sizes scale.
2. **Strict Modularity**: Clean separation of configuration, pattern generation, CPU computation, CUDA kernel execution, preview rendering, and UI layers.
3. **Interactive Control & Live Preview**: Real-time parameter tweaking with immediate visual feedback before running the simulation.
4. **Jupyter Notebook (`.ipynb`) Integration**: Full support for running within Jupyter notebooks using interactive widgets, alongside an optional native desktop GUI.

---

## 📁 Modular Project Structure

```
Conway-s-Game-of-Life-Python-and-cuda-/
│
├── conway_game_of_life.ipynb   # 📓 Main interactive Jupyter Notebook
├── main.py                     # 🚀 Standalone desktop GUI launcher
├── requirements.txt            # 📦 Python dependencies
├── readme.md                   # 📖 Project documentation
│
└── src/                        # 🧩 Core modular package
    ├── __init__.py             # Package exports
    ├── config.py               # SimulationConfig dataclass & parameter enums
    ├── patterns.py             # Pattern generator (Random, Glider, Glider Gun, etc.)
    ├── cpu_engine.py           # Vectorized NumPy CPU simulation engine
    ├── cuda_engine.py          # Numba CUDA 2D kernel GPU simulation engine
    ├── preview.py              # Real-time RGB grid renderer for preview UI
    ├── benchmark.py            # Timing, speedup calculation & verification harness
    └── gui/
        ├── __init__.py
        ├── tk_window.py        # Desktop Tkinter GUI (Settings + Live Preview + Run)
        └── ipy_widget.py       # Jupyter Notebook ipywidgets dashboard
```

---

## ⚙️ Configuration Settings & Specification

The settings panel allows you to configure every parameter required for the HPC assignment. Whenever you adjust a setting, the **Preview** canvas updates immediately:

| Setting | Options / Default | Purpose in HPC & Assignment |
| :--- | :--- | :--- |
| **Grid size** | Dropdown: `64`, `128`, `256`, `512`, `1024` | Main independent variable for performance comparison across workload scales. |
| **Iterations** | Default: `100` | Fixed number of generations to advance for consistent benchmarking. |
| **Initial pattern** | `Random`, `Glider`, `Glider Gun`, `R-pentomino`, `Blinker` | Provides both dynamic natural chaos (Random) and deterministic patterns. |
| **Random density** | Slider: `10%` to `50%` (Default: `25%`) | Controls initial alive cell percentage; prevents immediate extinction. |
| **Random seed** | Number (Default: `42`) | Ensures identical starting grids for CPU and GPU to guarantee fair comparison. |
| **Edge behavior** | `Wrap-around` (Toroidal) / `Dead edges` | Changes neighbor boundary logic (periodic vs zero-boundary conditions). |
| **Run mode** | `Both`, `CPU only`, `GPU only` | Allows isolated testing or direct side-by-side verification. |
| **Display mode** | `Live view`, `Benchmark (no drawing)` | Benchmark mode excludes GUI drawing overhead for pure compute timings. |
| **Cell size (px)** | Slider: `1` to `16 px` (Default: `4 px`) | Keeps the display window at a sensible size regardless of grid resolution. |
| **CUDA block size** | `8x8`, `16x16`, `32x32` | Tests how GPU warp occupancy and thread block dimensions impact throughput. |

---

## 💻 How to Run

### Option 1: Using the Jupyter Notebook (`.ipynb`)
Open `conway_game_of_life.ipynb` in VS Code or Jupyter Notebook:
1. Select the project Python kernel (`./venv`).
2. Run through the cells step-by-step:
   - Cell 1-2: Environment and GPU check.
   - Cell 3: Interactive **Settings & Live Preview Dashboard** (adjust dropdowns/sliders and see immediate updates).
   - Cell 4-5: Pattern evolution and visualization with Matplotlib.
   - Cell 6-7: Multi-grid HPC benchmark comparison and speedup plots.

### Option 2: Running the Standalone Desktop Window
Run the main script from your terminal:
```bash
python main.py
```
This opens the native desktop window containing:
- **Top section**: Full settings controls.
- **Middle section**: Real-time canvas preview of the starting pattern.
- **Bottom section**: Prominent **▶ RUN SIMULATION / BENCHMARK** button ready for execution.

---

## 🔬 CUDA Implementation Details

- **2D Thread Grid**: Mapped directly to cell coordinates `(x, y)`:
  $$\text{GridDim} = \left(\left\lceil \frac{W}{B_x} \right\rceil, \left\lceil \frac{H}{B_y} \right\rceil\right)$$
- **Double Buffering**: Two device memory buffers are allocated (`d_current` and `d_next`). The kernel advances generations on the GPU without intermediate host-device transfers.
- **HPC Metrics Recorded**:
  - `CPU Compute Time`: Pure NumPy computation time.
  - `CUDA Kernel Time`: Device-only compute execution time.
  - `CUDA Total Time`: Memory copies (Host-to-Device + Device-to-Host) + Kernel execution.
  - `Speedup`: $\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{GPU}}}$.