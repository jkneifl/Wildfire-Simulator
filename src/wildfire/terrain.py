"""Procedural terrain, initial vegetation and initial state (SPEC 3.4).

All randomness uses ``Purpose.TERRAIN`` with stream 0, so terrain depends only on the
seed. The ``step`` slot of the counter is used as a sub-counter for the noise field.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .rng import Purpose, uniform
from .state import GROUND, ROCK, WATER, WorldState

# Sub-counters for the TERRAIN purpose.
_LAKE, _ROCK, _VEG, _VEG_CELL, _RIVER, _AGENT = 0, 1, 2, 3, 4, 5


def value_noise(
    seed: int, shape: tuple[int, int], scale: float, octaves: int, field: int
) -> np.ndarray:
    """Smooth fractal value noise in [0, 1], shape ``(H, W)``."""
    h, w = shape
    total = np.zeros(shape)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        s = max(scale / 2**o, 1.0)
        gh, gw = int(np.ceil(h / s)) + 2, int(np.ceil(w / s)) + 2
        gy, gx = np.indices((gh, gw))
        lattice = uniform(seed, 0, field * 16 + o, Purpose.TERRAIN, gy, gx)
        fy, fx = np.arange(h) / s, np.arange(w) / s
        y0, x0 = fy.astype(int), fx.astype(int)
        ty, tx = fy - y0, fx - x0
        ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)  # smoothstep
        a = lattice[np.ix_(y0, x0)]
        b = lattice[np.ix_(y0, x0 + 1)]
        c = lattice[np.ix_(y0 + 1, x0)]
        d = lattice[np.ix_(y0 + 1, x0 + 1)]
        tx = tx[None, :]
        ty = ty[:, None]
        total += amp * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)
        norm += amp
        amp *= 0.5
    return total / norm


def _top_k_mask(values: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Mask of the ``k`` largest ``values`` among ``allowed`` cells."""
    mask = np.zeros(values.shape, bool)
    k = int(min(max(k, 0), allowed.sum()))
    if k:
        flat = np.flatnonzero(allowed)
        top = flat[np.argpartition(values.ravel()[flat], -k)[-k:]]
        mask.ravel()[top] = True
    return mask


def _rivers(seed: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    u = uniform(seed, 0, _RIVER, Purpose.TERRAIN, np.zeros(16, int), np.arange(16))
    mask = np.zeros(shape, bool)
    n_rivers = 1 + int(u[0] < 0.5)
    for r in range(n_rivers):
        v = u[1 + 7 * r : 8 + 7 * r]
        vertical = v[0] < 0.5
        length, across = (h, w) if vertical else (w, h)
        a = np.arange(length)
        centre = across * (0.2 + 0.6 * v[1])
        lateral = centre + across * (
            0.08 * np.sin(2 * np.pi * (a / length * (1 + 2 * v[2]) + v[3]))
            + 0.03 * np.sin(2 * np.pi * (a / length * (4 + 4 * v[4]) + v[5]))
        )
        half = 1 + int(v[6] < 0.5)
        for off in range(-half, half + 1):
            c = np.clip(np.round(lateral).astype(int) + off, 0, across - 1)
            if vertical:
                mask[a, c] = True
            else:
                mask[c, a] = True
    return mask


def generate_terrain(cfg: Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(terrain, veg)`` uint8 arrays for ``seed``."""
    shape = (cfg.height, cfg.width)
    n = cfg.height * cfg.width
    scale = max(min(shape) / 6, 4)

    river = _rivers(seed, shape)
    lake_noise = value_noise(seed, shape, scale, 4, _LAKE)
    lakes = _top_k_mask(lake_noise, ~river, round(cfg.water_frac * n) - river.sum())
    water = river | lakes
    rock_noise = value_noise(seed, shape, scale / 2, 4, _ROCK)
    rock = _top_k_mask(rock_noise, ~water, round(cfg.rock_frac * n))

    terrain = np.full(shape, GROUND, np.uint8)
    terrain[water] = WATER
    terrain[rock] = ROCK

    density = np.clip((value_noise(seed, shape, scale, 5, _VEG) - 0.25) / 0.5, 0.0, 1.0)
    y, x = np.indices(shape)
    occupied = uniform(seed, 0, _VEG_CELL, Purpose.TERRAIN, y, x) < density
    stage = np.clip(1 + np.floor(density * cfg.v_max), 1, cfg.v_max).astype(np.uint8)
    veg = np.where(occupied & (terrain == GROUND), stage, 0).astype(np.uint8)
    return terrain, veg


def place_agent(terrain: np.ndarray, seed: int) -> tuple[int, int]:
    """Random ground cell near the map centre; returns ``(x, y)``."""
    h, w = terrain.shape
    y, x = np.indices((h, w))
    dist = np.hypot(y - h / 2, x - w / 2)
    radius = max(min(h, w) / 8, 2)
    ground = terrain == GROUND
    near = ground & (dist <= radius)
    if not near.any():
        near = ground & (dist <= dist[ground].min() + 1)
    score = np.where(near, uniform(seed, 0, _AGENT, Purpose.TERRAIN, y, x), np.inf)
    iy, ix = np.unravel_index(np.argmin(score), score.shape)
    return int(ix), int(iy)


def initial_state(cfg: Config, seed: int, stream: int = 0) -> WorldState:
    terrain, veg = generate_terrain(cfg, seed)
    ax, ay = place_agent(terrain, seed)
    zeros = np.zeros_like(veg)
    i64 = lambda v: np.asarray(v, np.int64)  # noqa: E731
    return WorldState(
        terrain=terrain,
        veg=veg,
        fire=zeros.copy(),
        ash_age=np.zeros(veg.shape, np.uint16),
        wet=zeros.copy(),
        t=i64(cfg.start_step),
        seed=i64(seed),
        stream=i64(stream),
        ax=i64(ax),
        ay=i64(ay),
        facing=i64(2),
        tank=i64(cfg.tank),
    )
