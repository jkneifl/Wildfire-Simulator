"""Evaluation utilities (SPEC 11)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import Config
from .dynamics import step
from .state import GROUND, WorldState
from .terrain import initial_state


def global_stats(s: WorldState) -> dict[str, float]:
    """Vegetation density and ash fraction (of ground cells), active fires, mean wetness."""
    ground = s.terrain == GROUND
    n = max(int(ground.sum()), 1)
    return {
        "veg_density": float(((s.veg > 0) & ground).sum() / n),
        "burned_frac": float(((s.ash_age > 0) & ground).sum() / n),
        "active_fires": int((s.fire > 0).sum()),
        "mean_wet": float(s.wet[ground].mean()) if ground.any() else 0.0,
    }


def patch_sizes(mask: np.ndarray) -> np.ndarray:
    """Sizes of 4-connected components of a boolean mask."""
    labels, n = ndimage.label(mask)
    return np.bincount(labels.ravel())[1:] if n else np.zeros(0, np.int64)


def patch_stats(s: WorldState) -> dict[str, np.ndarray]:
    """Size distributions of connected burned (ash) and vegetated components."""
    return {"burned": patch_sizes(s.ash_age > 0), "vegetated": patch_sizes(s.veg > 0)}


def simulate_stats(
    cfg: Config, seed: int, steps: int, every: int = 1, keep_every: int | None = None
) -> dict:
    """Headless no-op run: global statistics every ``every`` steps.

    Returns arrays per statistic, ``t``, the ``final`` state and, if ``keep_every`` is set,
    the list ``states`` of states every ``keep_every`` steps.
    """
    s = initial_state(cfg, seed)
    rows, states = [], []
    for i in range(1, steps + 1):
        s = step(s, 0, cfg)
        if i % every == 0:
            rows.append(global_stats(s))
        if keep_every and i % keep_every == 0:
            states.append(s)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["t"] = np.arange(1, len(rows) + 1) * every + cfg.start_step
    out["final"] = s
    out["states"] = states
    return out


def regime_summary(stats: dict, cfg: Config, burn_in: int = 0) -> dict[str, float]:
    """Numbers for the target regime of SPEC 3.3, from a ``simulate_stats`` run (every=1)."""
    fires = stats["active_fires"][burn_in:]
    veg = stats["veg_density"][burn_in:]
    burned = stats["burned_frac"][burn_in:]
    free = np.concatenate([[0], (fires == 0).astype(int), [0]])
    edges = np.flatnonzero(np.diff(free))
    longest = int((edges[1::2] - edges[::2]).max()) if edges.size else 0
    return {
        "frac_steps_with_fire": float((fires > 0).mean()),
        "longest_fire_free_days": longest / cfg.day_length,
        "mean_active_fires": float(fires.mean()),
        "veg_mean": float(veg.mean()),
        "veg_min": float(veg.min()),
        "veg_max": float(veg.max()),
        "burned_mean": float(burned.mean()),
        "burned_max": float(burned.max()),
    }
