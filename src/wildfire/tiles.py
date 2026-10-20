"""Square tiles of ``T x T`` cells (SPEC 8). A tile is ``(ty, tx)``."""

from __future__ import annotations

import numpy as np

from .state import LAYERS, WorldState


def grid_shape(shape: tuple[int, int], T: int) -> tuple[int, int]:
    """Number of tiles ``(rows, cols)``; edge tiles may be partial."""
    return -(-shape[0] // T), -(-shape[1] // T)


def tile_of(y, x, T: int):
    """Tile index of cell ``(y, x)`` (scalars or arrays)."""
    return np.floor_divide(y, T), np.floor_divide(x, T)


def tile_cells(tile: tuple[int, int], T: int) -> tuple[slice, slice]:
    """Slices selecting the cells of ``tile`` (NumPy clips them at the world edge)."""
    ty, tx = tile
    return slice(ty * T, (ty + 1) * T), slice(tx * T, (tx + 1) * T)


def tiles_in_window(window: tuple[int, int, int, int], T: int, shape) -> list[tuple[int, int]]:
    """In-world tiles intersecting ``window = (y0, x0, h, w)``."""
    y0, x0, h, w = window
    rows, cols = grid_shape(shape, T)
    ya, yb = max(y0 // T, 0), min((y0 + h - 1) // T, rows - 1)
    xa, xb = max(x0 // T, 0), min((x0 + w - 1) // T, cols - 1)
    return [(ty, tx) for ty in range(ya, yb + 1) for tx in range(xa, xb + 1)]


def extract_tile(s: WorldState, tile: tuple[int, int], T: int) -> dict[str, np.ndarray]:
    """All layers of one tile (copies); works on batched states too."""
    ys, xs = tile_cells(tile, T)
    return {k: np.array(getattr(s, k)[..., ys, xs]) for k in LAYERS}


class TileTracker:
    """Per-tile first and last step at which any of its cells was in the agent window."""

    def __init__(self, shape: tuple[int, int], T: int):
        self.T = T
        self.shape = shape
        self.first_seen = np.full(grid_shape(shape, T), -1, np.int64)
        self.last_seen = np.full(grid_shape(shape, T), -1, np.int64)

    def mark(self, window: tuple[int, int, int, int], t: int) -> list[tuple[int, int]]:
        tiles = tiles_in_window(window, self.T, self.shape)
        if tiles:
            ty, tx = np.array(tiles).T
            self.first_seen[ty, tx] = np.where(
                self.first_seen[ty, tx] < 0, t, self.first_seen[ty, tx]
            )
            self.last_seen[ty, tx] = t
        return tiles
