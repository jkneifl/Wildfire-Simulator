"""Discrete agent actions (SPEC 3.5), applied as step 1 of the update."""

from __future__ import annotations

import numpy as np

from .config import Config
from .state import GROUND, WATER, WorldState

NOOP, MOVE_N, MOVE_E, MOVE_S, MOVE_W, IGNITE, SPRAY = range(7)
N_ACTIONS = 7
NAMES = ("noop", "north", "east", "south", "west", "ignite", "water")

# Facing 0 N, 1 E, 2 S, 3 W.
DY = np.array([-1, 0, 1, 0])
DX = np.array([0, 1, 0, -1])


def apply_action(s: WorldState, action: np.ndarray, cfg: Config) -> WorldState:
    """Apply one action per world. ``s`` must be batched; returns a new state.

    ``fire`` and ``wet`` are copied; other layers are shared with ``s`` (never mutated).
    """
    n, h, w = s.veg.shape
    a = np.broadcast_to(np.asarray(action, np.int64), (n,))
    idx = np.arange(n)

    move = (a >= MOVE_N) & (a <= MOVE_W)
    facing = np.where(move, a - MOVE_N, s.facing)
    ty, tx = s.ay + DY[facing], s.ax + DX[facing]
    inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
    cy, cx = np.clip(ty, 0, h - 1), np.clip(tx, 0, w - 1)

    can_move = move & inside & (s.terrain[idx, cy, cx] == GROUND)
    ax = np.where(can_move, tx, s.ax)
    ay = np.where(can_move, ty, s.ay)

    fire = s.fire.copy()
    wet = s.wet.copy()

    veg_t = s.veg[idx, cy, cx]
    ign = (a == IGNITE) & inside & (veg_t > 0) & (fire[idx, cy, cx] == 0)
    ign &= wet[idx, cy, cx] < cfg.wet_block
    fire[idx[ign], cy[ign], cx[ign]] = cfg.burn_base + cfg.burn_per_veg * veg_t[ign]

    spray = (a == SPRAY) & (cfg.infinite_water | (s.tank > 0))
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            by, bx = ty + oy, tx + ox
            m = spray & (by >= 0) & (by < h) & (bx >= 0) & (bx < w)
            fire[idx[m], by[m], bx[m]] = 0
            wet[idx[m], by[m], bx[m]] = 255
    tank = s.tank - (spray & (not cfg.infinite_water))

    # Refill when standing orthogonally next to water.
    near_water = np.zeros(n, bool)
    for d in range(4):
        ny, nx = ay + DY[d], ax + DX[d]
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        near_water |= ok & (s.terrain[idx, np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)] == WATER)
    tank = np.where(near_water | cfg.infinite_water, cfg.tank, tank)

    return WorldState(
        terrain=s.terrain,
        veg=s.veg,
        fire=fire,
        ash_age=s.ash_age,
        wet=wet,
        t=s.t,
        seed=s.seed,
        stream=s.stream,
        ax=ax,
        ay=ay,
        facing=facing,
        tank=tank.astype(np.int64),
    )
