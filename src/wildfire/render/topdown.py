"""Pure NumPy top-down renderer (SPEC 5). Does not import pygame.

Colours are computed from the observable channels only (``observe.grid``), so the image
never carries more information than the grid observation. The two per-cell textures
(vegetation/water variant, flame phase) are fixed per cell and depend only on the seed.
"""

from __future__ import annotations

import numpy as np

from .. import rng
from ..config import Config
from ..dynamics import light
from ..observe import grid
from ..rng import Purpose
from ..state import GROUND, ROCK, WATER, WorldState
from . import palette as P

# TERRAIN sub-counter for the variant texture; terrain.py uses 0..5 and 0..36 (noise fields).
VARIANT_COUNTER = 64


def _table(colors) -> np.ndarray:
    return np.array(colors, np.float64)


VEG = _table(P.VEG)
VEG_VARIANT = _table(P.VEG_VARIANT)
ASH = _table(P.ASH)
FIRE, INK, EMBER = _table(P.FIRE), _table(P.INK), _table(P.EMBER)


def _round(c: np.ndarray) -> np.ndarray:
    return np.floor(c + 0.5)


def variant(seed: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Fixed per-cell bit: darker vegetation / lighter water on half of the cells."""
    y, x = np.indices((h, w))
    return rng.uniform(int(seed), 0, VARIANT_COUNTER, Purpose.TERRAIN, y + y0, x + x0) < 0.5


def flame_phase(seed: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Fixed per-cell uniform number ``u_xy`` of the flame shimmer."""
    y, x = np.indices((h, w))
    return rng.uniform(int(seed), 0, 0, Purpose.FLICKER, y + y0, x + x0)


def colors(
    g: np.ndarray, light_level: float, t: float, var: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """RGB ``(h, w, 3)`` uint8 from an observable grid ``(h, w, C)``.

    ``t`` is the (possibly interpolated) step for the flame shimmer; ``var`` and ``phase`` come
    from :func:`variant` and :func:`flame_phase` for the same cells.
    """
    terrain, veg, burning, fb, wb, ab = (g[..., i] for i in range(6))
    c = ASH[ab]  # ground: bare (bucket 0) or ash
    c = np.where((veg > 0)[..., None], np.where(var[..., None], VEG_VARIANT[veg], VEG[veg]), c)
    water = np.where(var[..., None], _table(P.WATER_VARIANT), _table(P.WATER))
    c = np.where((terrain == WATER)[..., None], water, c)
    c = np.where((terrain == ROCK)[..., None], _table(P.ROCK), c)
    # wetness tint from the observable bucket (representative wet = 85 * bucket)
    f = (P.WET_MIX * 85 * wb / 255 * (terrain == GROUND))[..., None]
    c = _round(c + f * (_table(P.WET_TINT) - c))
    c = _round(c * (P.NIGHT_MIN + (1 - P.NIGHT_MIN) * light_level))
    # burning cells stay bright: flame in buckets 2-3, ember in bucket 1
    f = P.FLAME_BASE + P.FLAME_AMP * np.sin(2 * np.pi * (t / P.FLAME_PERIOD + phase))
    flame = _round(FIRE + f[..., None] * (INK - FIRE))
    fire = np.where((fb >= 2)[..., None], flame, EMBER)
    return np.where((burning > 0)[..., None], fire, c).astype(np.uint8)


def gap_px(ppc: int, force: bool = False) -> int:
    """Grid gap in pixels for ``ppc`` pixels per cell."""
    if ppc < P.GAP_MIN_PX:
        return 1 if force and ppc > 1 else 0
    return max(1, int(np.floor(P.GAP_FRAC * ppc + 0.5)))


def upscale(c: np.ndarray, ppc: int, grid_lines: bool = False) -> np.ndarray:
    """Repeat each cell to ``ppc x ppc`` pixels, inset by the grid gap (GROUND shows through)."""
    img = np.repeat(np.repeat(c, ppc, axis=0), ppc, axis=1)
    g = gap_px(ppc, grid_lines)
    if g:
        lo, hi = g // 2, ppc - (g - g // 2)  # inset g/2 per side (the odd pixel goes right/bottom)
        ly = np.arange(img.shape[0]) % ppc
        lx = np.arange(img.shape[1]) % ppc
        gap = ((ly < lo) | (ly >= hi))[:, None] | ((lx < lo) | (lx >= hi))[None, :]
        img[gap] = P.GROUND
    return img


def render(
    s: WorldState,
    window: tuple[int, int, int, int],
    cfg: Config,
    ppc: int | None = None,
    grid_lines: bool | None = None,
    t: float | None = None,
) -> np.ndarray:
    """uint8 image ``(h * ppc, w * ppc, 3)`` of ``window = (y0, x0, h, w)``.

    ``t`` defaults to the integer step (dataset frames); the game passes an interpolated one.
    """
    y0, x0, h, w = window
    g = grid(s, y0, x0, h, w, cfg)
    seed = int(s.seed)
    c = colors(
        g,
        float(light(s.t, cfg.day_length)),
        float(s.t) if t is None else t,
        variant(seed, y0, x0, h, w),
        flame_phase(seed, y0, x0, h, w),
    )
    ppc = cfg.pixels_per_cell if ppc is None else ppc
    return upscale(c, ppc, cfg.render_grid if grid_lines is None else grid_lines)
