"""Attribution of new ignitions to their cause (privileged ground truth, for evaluation).

For one step ``prev --action--> new`` every newly burning cell gets exactly one cause:

- ``AGENT``: the cell an ``IGNITE`` action set burning (step 1 of the update);
- ``LIGHTNING``: the cell was struck (rule 4). The strike mask is recomputed from the same
  counter-based random numbers the dynamics use, so this is exact even when the cell also
  had burning neighbours;
- ``SPREAD``: any other new ignition (rule 3).

Neighbours are taken from the burning mask *after* the action, as in the dynamics, so cells
the agent sprayed in this step do not count as burning neighbours.
"""

from __future__ import annotations

import numpy as np

from . import rng
from .actions import apply_action
from .config import Config
from .dynamics import N8
from .rng import Purpose
from .state import SCALARS, WorldState

AGENT, LIGHTNING, SPREAD = 1, 2, 3
CAUSES = {AGENT: "agent", LIGHTNING: "lightning", SPREAD: "spread"}


def _batched(s: WorldState) -> WorldState:
    return WorldState(
        **{k: v.reshape(1) if k in SCALARS else v[None] for k, v in s.as_dict().items()}
    )


def ignitions(prev: WorldState, action: int, new: WorldState, cfg: Config) -> dict:
    """New ignitions of one unbatched step.

    Returns arrays ``y``, ``x``, ``cause`` and ``nbrs``: a bitmask of the 8 neighbours (in
    ``dynamics.N8`` order) that were burning after the action, e.g. to tell spread across a
    tile boundary from spread within a tile at load time.
    """
    post = apply_action(_batched(prev), action, cfg)
    fire_post = post.fire[0]
    burning = fire_post > 0
    agent = (prev.fire == 0) & burning
    dynamic = ~burning & (new.fire > 0)
    lightning = np.zeros_like(dynamic)
    if cfg.p_lightning > 0:
        h, w = prev.shape
        k = rng.key(int(prev.seed), int(prev.stream), int(prev.t), Purpose.LIGHTNING)
        strike = rng.bits(np.uint64(k), rng.grid_hash(h, w)) < rng.threshold(cfg.p_lightning)
        lightning = strike & (prev.veg > 0) & ~burning
    cause = np.zeros(prev.shape, np.uint8)
    cause[dynamic] = SPREAD
    cause[lightning] = LIGHTNING
    cause[agent] = AGENT
    y, x = np.nonzero(cause)
    bp = np.pad(burning, 1)
    nbrs = np.zeros(y.size, np.uint8)
    for bit, (dy, dx, _) in enumerate(N8):
        nbrs |= bp[1 + y + dy, 1 + x + dx].astype(np.uint8) << bit
    return {
        "y": y.astype(np.int32),
        "x": x.astype(np.int32),
        "cause": cause[y, x],
        "nbrs": nbrs,
        "lightning_mask": lightning,
        "dynamic_mask": dynamic,
    }
