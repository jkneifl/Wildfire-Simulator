"""Pure NumPy top-down renderer (SPEC 5). Does not import pygame.

Colours are computed from the observable channels only (``observe.grid``), so the image
never carries more information than the grid observation (apart from flicker noise).
"""

from __future__ import annotations

import numpy as np

from .. import rng
from ..config import Config
from ..dynamics import light
from ..observe import grid
from ..rng import Purpose
from ..state import GROUND, WATER, WorldState
from . import palette as P


def colors(g: np.ndarray, light_level: float, flicker: np.ndarray | None = None) -> np.ndarray:
    """Float RGB ``(h, w, 3)`` from an observable grid ``(h, w, C)``."""
    terrain, veg, burning, fb, wb, ab = (g[..., i] for i in range(6))
    c = P.TERRAIN[terrain]
    c = np.where((veg > 0)[..., None], P.VEG[veg], c)
    c = np.where(((ab > 0) & (veg == 0) & (terrain == GROUND))[..., None], P.ASH[ab], c)
    tint = (P.WET_STRENGTH * wb * (terrain != WATER))[..., None]
    c = c * (1 - tint) + P.WET_TINT * tint
    c = c * (P.NIGHT_MIN + (1 - P.NIGHT_MIN) * light_level)
    fire = P.FIRE[fb]
    if flicker is not None:
        fire = fire * (0.8 + 0.4 * flicker)[..., None]
    return np.where((burning > 0)[..., None], fire, c)  # fires stay bright at night


def upscale(c: np.ndarray, ppc: int, grid_lines: bool = False) -> np.ndarray:
    img = np.repeat(np.repeat(c, ppc, axis=0), ppc, axis=1)
    if grid_lines and ppc > 2:
        img[::ppc] *= P.GRID_LINE
        img[:, ::ppc] *= P.GRID_LINE
    return np.clip(img, 0, 255).astype(np.uint8)


def flicker(s: WorldState, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    y, x = np.indices((h, w))
    u = rng.uniform(int(s.seed), int(s.stream), int(s.t), Purpose.FLICKER, y + y0, x + x0)
    return u.astype(np.float32)


def render(
    s: WorldState,
    window: tuple[int, int, int, int],
    cfg: Config,
    ppc: int | None = None,
    grid_lines: bool = False,
) -> np.ndarray:
    """uint8 image ``(h * ppc, w * ppc, 3)`` of ``window = (y0, x0, h, w)``."""
    y0, x0, h, w = window
    g = grid(s, y0, x0, h, w, cfg)
    c = colors(g, float(light(s.t, cfg.day_length)), flicker(s, y0, x0, h, w))
    return upscale(c, cfg.pixels_per_cell if ppc is None else ppc, grid_lines)
