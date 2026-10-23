"""Agent window, observable and privileged views (SPEC 4, R5).

Everything an agent may receive comes from :func:`observe`. Exact timers, the seed
and the stream are available only through :func:`privileged`.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .dynamics import light
from .state import ROCK, WorldState

CHANNELS = ("terrain", "veg", "burning", "fire_bucket", "wet_bucket", "ash_bucket")


def crop(arr: np.ndarray, y0: int, x0: int, h: int, w: int, fill=0) -> np.ndarray:
    """``arr[y0:y0+h, x0:x0+w]``, with cells outside the array set to ``fill``."""
    out = np.full((h, w), fill, arr.dtype)
    H, W = arr.shape
    ya, yb = max(y0, 0), min(y0 + h, H)
    xa, xb = max(x0, 0), min(x0 + w, W)
    if ya < yb and xa < xb:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = arr[ya:yb, xa:xb]
    return out


def fire_bucket(fire: np.ndarray, cfg: Config) -> np.ndarray:
    """0 not burning, 1..3 by remaining burn steps relative to the maximum."""
    b = np.ceil(3 * fire.astype(np.float32) / cfg.fire_max)
    return np.where(fire > 0, np.clip(b, 1, 3), 0).astype(np.uint8)


def wet_bucket(wet: np.ndarray) -> np.ndarray:
    """0 dry, 1 for 1..85, 2 for 86..170, 3 for 171..255."""
    return ((wet.astype(np.uint16) + 84) // 85).astype(np.uint8)


def ash_bucket(ash_age: np.ndarray, cfg: Config) -> np.ndarray:
    """0 no ash, then 1..3 by thirds of REGROW_DELAY (3 = old enough to regrow, or nearly)."""
    a = ash_age.astype(np.int64)
    b = 1 + np.minimum(2, (a - 1) * 3 // max(cfg.regrow_delay, 1))
    return np.where(a > 0, b, 0).astype(np.uint8)


def grid(s: WorldState, y0: int, x0: int, h: int, w: int, cfg: Config) -> np.ndarray:
    """Observable channels ``(h, w, C)`` of a window; outside the world is rock."""
    fire = crop(s.fire, y0, x0, h, w)
    return np.stack(
        [
            crop(s.terrain, y0, x0, h, w, fill=ROCK),
            crop(s.veg, y0, x0, h, w),
            (fire > 0).astype(np.uint8),
            fire_bucket(fire, cfg),
            wet_bucket(crop(s.wet, y0, x0, h, w)),
            ash_bucket(crop(s.ash_age, y0, x0, h, w), cfg),
        ],
        axis=-1,
    )


def transition_log_prob(tp: dict, next_grid: np.ndarray, cfg: Config) -> np.ndarray:
    """Per-cell ``log p`` of an observed next grid ``(..., H, W, 6)`` (``grid`` channels)
    under ``tp = dynamics.transition_probs(...)``: the oracle's one-step likelihood."""
    match = (
        (tp["veg"] == next_grid[..., 1, None])
        & (fire_bucket(tp["fire"], cfg) == next_grid[..., 3, None])
        & (wet_bucket(tp["wet"]) == next_grid[..., 4, None])
        & (ash_bucket(tp["ash_age"], cfg) == next_grid[..., 5, None])
    )
    with np.errstate(divide="ignore"):
        return np.log((tp["prob"] * match).sum(-1))


def agent_window(s: WorldState, cfg: Config) -> tuple[int, int, int, int]:
    """``(y0, x0, h, w)`` of the agent's V x V window."""
    r = cfg.window // 2
    return int(s.ay) - r, int(s.ax) - r, cfg.window, cfg.window


def observe(s: WorldState, cfg: Config, rgb: bool = True) -> dict:
    """Observation of one (unbatched) world."""
    win = agent_window(s, cfg)
    obs = {
        "grid": grid(s, *win, cfg),
        "light": float(light(s.t, cfg.day_length)),
        "agent": {"facing": int(s.facing), "tank": int(s.tank)},
    }
    if rgb:
        from .render.topdown import render

        obs["rgb"] = render(s, win, cfg)
    return obs


def privileged(s: WorldState) -> dict:
    """Full state for ground truth and evaluation only (copies)."""
    return {k: np.array(v) for k, v in s.as_dict().items()}
