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

    def toward(self, obs: dict, dy: int, dx: int) -> int:
        """Greedy move that reduces the offset ``(dy, dx)``; random free move if blocked."""
        dirs = []
        if dy or dx:
            vert = 0 if dy < 0 else 2
            horz = 3 if dx < 0 else 1
            dirs = [vert, horz] if abs(dy) >= abs(dx) else [horz, vert]
            dirs = [d for d in dirs if (d in (0, 2) and dy) or (d in (1, 3) and dx)]
        for d in dirs:
            if self.free(obs, d):
                self.heading = d
                return MOVES[d]
        return RandomWalk.act(self, obs, {})


class Firestarter(RandomWalk):
    """Random walk that ignites the cell in front with probability ``p_ignite``."""

    name = "firestarter"
    p_ignite = 0.02

    def act(self, obs: dict, info: dict) -> int:
        if self.uniform()[0] < self.p_ignite:
            return A.IGNITE
        return super().act(obs, info)


class Firefighter(RandomWalk):
    """Waters the nearest visible fire; refills at the nearest visible water."""

    name = "firefighter"

    def nearest(self, mask: np.ndarray) -> tuple[int, int] | None:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        r = self.cfg.window // 2
        i = np.argmin(np.abs(ys - r) + np.abs(xs - r))
        return int(ys[i] - r), int(xs[i] - r)

    def act(self, obs: dict, info: dict) -> int:
        g = obs["grid"]
        if obs["agent"]["tank"] == 0 and not self.cfg.infinite_water:
            w = self.nearest(g[..., 0] == 1)
            return RandomWalk.act(self, obs, info) if w is None else self.toward(obs, *w)
        f = self.nearest(g[..., 2] > 0)
        if f is None:
            return RandomWalk.act(self, obs, info)
        face = int(obs["agent"]["facing"])
        if max(abs(f[0] - A.DY[face]), abs(f[1] - A.DX[face])) <= 1:
            return A.SPRAY
        return self.toward(obs, *f)


class ExploreReturn(RandomWalk):
    """Observe a region, leave it for a log-uniform absence ``dt``, then return.

    Logs ``{"tile", "leave_step", "return_step", "dt"}`` per revisit in ``self.log``, where
    ``leave_step`` is the last step the tile was in the window and ``return_step`` the first
    step it is back in view.
    """

    name = "explore_return"
    observe_steps = 40
    dt_min, dt_max = 50, 2000

    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__(cfg, seed)
        self.phase = None

    def start(self, info: dict) -> None:
        self.phase, self.home = "observe", info["pos"]
        self.tile = (self.home[1] // self.cfg.tile, self.home[0] // self.cfg.tile)
        self.until = info["step"] + self.observe_steps

    def act(self, obs: dict, info: dict) -> int:
        if self.phase is None:
            self.start(info)
        x, y = info["pos"]
        hx, hy = self.home
        t = info["step"]
        visible = self.tile in info["tiles"]
        if self.phase == "observe":
            if t >= self.until:
                u = self.uniform()[0]
                self.dt = float(np.exp(np.log(self.dt_min) + u * np.log(self.dt_max / self.dt_min)))
                self.phase = "leave"
            elif max(abs(x - hx), abs(y - hy)) > self.cfg.tile // 2:
                return self.toward(obs, hy - y, hx - x)
            return RandomWalk.act(self, obs, info)
        if self.phase == "leave":
            if not visible:
                self.leave_step = int(info["last_seen"][self.tile])
                self.phase = "away"
            else:
                d0 = self.dist(x, y)
                a = self.wander(obs, x, y, lambda nx, ny: self.dist(nx, ny) >= d0)
                return a if a != A.NOOP else self.wander(obs, x, y, lambda nx, ny: True)
        if self.phase == "away":
            if visible:  # safety: back in view, leave again and restart the absence
                self.phase = "leave"
                return RandomWalk.act(self, obs, info)
            if t - self.leave_step >= self.dt:
                self.phase = "return"
                self.deadline = t + 4 * (self.cfg.height + self.cfg.width)
            else:  # wander without bringing the tile into view
                return self.wander(obs, x, y, lambda nx, ny: not self.sees(nx, ny))
        # return
        if visible:
            self.log.append(
                {
                    "tile": list(self.tile),
                    "leave_step": self.leave_step,
                    "return_step": t,
                    "dt": round(self.dt, 1),
                }
            )
            self.start(info)
            return RandomWalk.act(self, obs, info)
        if t > self.deadline:  # stuck behind an obstacle: give up this revisit
            self.start(info)
            return RandomWalk.act(self, obs, info)
        return self.toward(obs, hy - y, hx - x)

    def sees(self, x: int, y: int) -> bool:
        """Whether the window at ``(x, y)`` intersects the target tile."""
        r, T = self.cfg.window // 2, self.cfg.tile
        ty, tx = self.tile
        return (
            abs(x - (tx * T + (T - 1) / 2)) <= r + T / 2
            and abs(y - (ty * T + (T - 1) / 2)) <= r + T / 2
        )

    def dist(self, x: int, y: int) -> float:
        T = self.cfg.tile
        ty, tx = self.tile
        return max(abs(x - (tx + 0.5) * T), abs(y - (ty + 0.5) * T))

    def wander(self, obs: dict, x: int, y: int, ok) -> int:
        """Random walk with heading persistence over free moves to cells where ``ok``."""
        dirs = [d for d in range(4) if self.free(obs, d) and ok(x + A.DX[d], y + A.DY[d])]
        if not dirs:
            return A.NOOP
        if self.heading not in dirs or self.uniform()[0] < self.turn:
            self.heading = dirs[self.choice(len(dirs))]
        return MOVES[self.heading]


POLICIES: dict[str, type[Policy]] = {
    p.name: p for p in (RandomWalk, ExploreReturn, Firestarter, Firefighter)
}


def make_policy(name: str, cfg: Config, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name](cfg, seed)
