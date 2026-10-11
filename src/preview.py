"""
Grid preview renderer for Conway's Game of Life.
Renders initial state into an RGB PIL Image for live UI preview.
"""

from typing import Tuple
import numpy as np
from PIL import Image

# Sleek aesthetic theme colors
DEAD_COLOR = (24, 24, 27)      # Dark Charcoal #18181b
ALIVE_COLOR = (52, 211, 153)    # Emerald Mint #34d399
GRID_LINE_COLOR = (39, 39, 42)  # Subtle border #27272a


def render_preview_image(
    grid: np.ndarray,
    target_size: int = 380,
    dead_color: Tuple[int, int, int] = DEAD_COLOR,
    alive_color: Tuple[int, int, int] = ALIVE_COLOR
) -> Image.Image:
    """
    Render 2D binary grid to a PIL Image scaled to target_size x target_size.
    
    Args:
        grid: 2D uint8 numpy array (0 = dead, 1 = alive)
        target_size: Output image width and height in pixels
        dead_color: RGB tuple for dead cells
        alive_color: RGB tuple for alive cells
        
    Returns:
        PIL.Image.Image in RGB mode
    """
    rows, cols = grid.shape
    # Create 3D RGB array
    rgb_arr = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    # Assign colors
    is_alive = (grid == 1)
    rgb_arr[~is_alive] = dead_color
    rgb_arr[is_alive] = alive_color

    # Convert to PIL Image and resize using Nearest Neighbor to maintain crisp cell pixels
    img = Image.fromarray(rgb_arr, mode="RGB")
    scaled_img = img.resize((target_size, target_size), resample=Image.Resampling.NEAREST)
    return scaled_img
