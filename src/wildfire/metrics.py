"""Evaluation utilities (SPEC 11)."""

from __future__ import annotations

from .state import GROUND, WorldState


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
