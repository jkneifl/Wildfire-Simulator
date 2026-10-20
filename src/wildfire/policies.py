"""Scripted policies (SPEC 10). Deterministic given their own seed.

Randomness uses the counter-based RNG with purpose ``POLICY`` and the policy's own counter.
Policies see the observation and the non-privileged ``info`` (including the agent's own
position ``info["pos"]``).
"""

from __future__ import annotations

import numpy as np

from . import actions as A
from . import rng
from .config import Config
from .rng import Purpose
from .state import GROUND

MOVES = (A.MOVE_N, A.MOVE_E, A.MOVE_S, A.MOVE_W)


class Policy:
    name = "base"

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.counter = 0
        self.log: list = []

    def uniform(self, n: int = 1) -> np.ndarray:
        u = rng.uniform(self.seed, 0, self.counter, Purpose.POLICY, np.zeros(n, int), np.arange(n))
        self.counter += 1
        return u

    def choice(self, n: int) -> int:
        return int(self.uniform()[0] * n)

    def free(self, obs: dict, d: int) -> bool:
        """Whether the cell in direction ``d`` (0 N .. 3 E S W) is walkable ground."""
        r = self.cfg.window // 2
        return obs["grid"][r + A.DY[d], r + A.DX[d], 0] == GROUND

    def act(self, obs: dict, info: dict) -> int:
        raise NotImplementedError


class RandomWalk(Policy):
    """Keeps its heading with probability ``1 - turn``; turns when blocked."""

    name = "random_walk"
    turn = 0.15

    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__(cfg, seed)
        self.heading = self.choice(4)

    def act(self, obs: dict, info: dict) -> int:
        if self.uniform()[0] < self.turn or not self.free(obs, self.heading):
            free = [d for d in range(4) if self.free(obs, d)]
            self.heading = free[self.choice(len(free))] if free else self.choice(4)
        return MOVES[self.heading]


POLICIES: dict[str, type[Policy]] = {p.name: p for p in (RandomWalk,)}


def make_policy(name: str, cfg: Config, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name](cfg, seed)
