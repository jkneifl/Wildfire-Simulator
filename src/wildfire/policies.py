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
    params: tuple[str, ...] = ()  # tunable class attributes (see make_policy)

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.counter = 0
        self.log: list = []

    def param_values(self) -> dict:
        return {k: getattr(self, k) for k in self.params}

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
    params = ("turn",)

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
    params = ("turn", "p_ignite")

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
    p_ignite_before_leave = 0.0  # > 0: set a fire before leaving (explore_return_act)
    p_water = 0.0  # > 0: water visible fire in front while observing
    params = ("turn", "observe_steps", "dt_min", "dt_max", "p_ignite_before_leave", "p_water")

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
                if (
                    self.p_ignite_before_leave > 0
                    and self.uniform()[0] < self.p_ignite_before_leave
                ):
                    return A.IGNITE
            elif (
                self.p_water > 0 and fire_ahead(obs, self.cfg) and self.uniform()[0] < self.p_water
            ):
                return A.SPRAY
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
            abs(x - (tx * T + (T - 1) / 2)) <= r + (T - 1) / 2
            and abs(y - (ty * T + (T - 1) / 2)) <= r + (T - 1) / 2
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


class ExploreReturnAct(ExploreReturn):
    """``ExploreReturn`` that often sets a fire before leaving, so that a directed event
    starts in view and evolves while the place is out of sight."""

    name = "explore_return_act"
    p_ignite_before_leave = 0.5
    p_water = 0.05


def fire_ahead(obs: dict, cfg: Config) -> bool:
    """A burning cell within one cell of the cell in front (what a spray would reach)."""
    r = cfg.window // 2
    f = int(obs["agent"]["facing"])
    cy, cx = r + A.DY[f], r + A.DX[f]
    return bool(obs["grid"][cy - 1 : cy + 2, cx - 1 : cx + 2, 2].any())


class Player(RandomWalk):
    """Intent-based stand-in for a human player.

    Intents: ``explore`` (walk to an unseen tile), ``revisit`` (return to a remembered place,
    preferring places where it set or saw a fire, once its delay has passed), ``ignite`` (set
    a fire in front, then watch nearby), ``fight`` (water the nearest visible fire),
    ``refill`` (walk to visible water when the tank is empty) and ``idle``. An intent is held
    for a sampled dwell time and then re-chosen with context-dependent weights. Uses only
    the observation, ``info`` (own position, own tile history) and its own memory.
    """

    name = "player"
    w_explore, w_revisit, w_ignite, w_fight, w_idle = 1.0, 0.6, 0.25, 2.0, 0.2
    dwell_mean = 60.0  # steps an intent is held (exponential)
    idle_mean = 8.0  # steps of an idle intent
    watch_steps = 30  # steps spent near a fire after setting it
    revisit_median = 400.0  # delay before a remembered place is due (log-normal)
    p_pause = 0.05  # per-step hesitation while moving
    turn = 0.1
    params = (
        "w_explore", "w_revisit", "w_ignite", "w_fight", "w_idle",
        "dwell_mean", "idle_mean", "watch_steps", "revisit_median", "p_pause", "turn",
    )  # fmt: skip
    INTENTS = ("explore", "revisit", "ignite", "fight", "refill", "idle")

    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__(cfg, seed)
        self.intent, self.until, self.target = None, -1, None
        self.places: list[dict] = []  # remembered places {x, y, t, due}
        self.ignited_at = None

    def exp(self, mean: float) -> float:
        return float(-mean * np.log(1 - self.uniform()[0]))

    def remember(self, x: int, y: int, t: int) -> None:
        if any(abs(p["x"] - x) + abs(p["y"] - y) < self.cfg.tile for p in self.places):
            return
        z = np.sqrt(-2 * np.log(1 - self.uniform()[0])) * np.cos(2 * np.pi * self.uniform()[0])
        self.places.append(
            {"x": x, "y": y, "t": t, "due": t + self.revisit_median * np.exp(0.7 * z)}
        )

    def choose(self, obs: dict, info: dict) -> None:
        g, t = obs["grid"], info["step"]
        fire = bool(g[..., 2].any())
        empty = obs["agent"]["tank"] == 0 and not self.cfg.infinite_water
        due = [p for p in self.places if p["due"] <= t]
        w = {
            "explore": self.w_explore,
            "revisit": self.w_revisit if due else 0.0,
            "ignite": self.w_ignite,
            "fight": self.w_fight if fire and not empty else 0.0,
            "refill": 4.0 if empty and (g[..., 0] == 1).any() else 0.0,
            "idle": self.w_idle,
        }
        names = [k for k in self.INTENTS if w[k] > 0]
        p = np.array([w[k] for k in names]) / sum(w[k] for k in names)
        self.intent = names[int(np.searchsorted(np.cumsum(p), self.uniform()[0] * 0.999999))]
        dwell = self.idle_mean if self.intent == "idle" else self.dwell_mean
        self.until = t + 1 + int(self.exp(dwell))
        self.target, self.ignited_at = None, None
        x, y = info["pos"]
        if self.intent == "revisit":
            self.target = due[self.choice(len(due))]
            self.places.remove(self.target)
            self.target = (self.target["x"], self.target["y"])
        elif self.intent == "explore":
            unseen = np.argwhere(info["last_seen"] < 0)
            if len(unseen):
                T = self.cfg.tile
                c = unseen * T + T // 2
                d = np.abs(c[:, 0] - y) + np.abs(c[:, 1] - x)
                near = unseen[np.argsort(d)[:8]]
                ty, tx = near[self.choice(len(near))]
                self.target = (int(tx * T + T // 2), int(ty * T + T // 2))

    def act(self, obs: dict, info: dict) -> int:
        t, (x, y), g = info["step"], info["pos"], obs["grid"]
        if g[..., 2].any():
            self.remember(x, y, t)
        if self.intent is None or t >= self.until:
            self.choose(obs, info)
        if self.intent not in ("idle", "fight") and self.uniform()[0] < self.p_pause:
            return A.NOOP
        if self.intent == "idle":
            return A.NOOP
        if self.intent == "fight":
            if not g[..., 2].any() or (obs["agent"]["tank"] == 0 and not self.cfg.infinite_water):
                self.until = t  # nothing (left) to fight
                return RandomWalk.act(self, obs, info)
            return Firefighter.act(self, obs, info)
        if self.intent == "refill":
            if obs["agent"]["tank"] > 0:
                self.until = t
            return Firefighter.act(self, obs, info)
        if self.intent == "ignite":
            if self.ignited_at is None:
                r = self.cfg.window // 2
                f = int(obs["agent"]["facing"])
                ahead = g[r + A.DY[f], r + A.DX[f]]
                if ahead[1] > 0 and ahead[2] == 0 and ahead[4] < 2:
                    self.ignited_at = t
                    self.until = t + 1 + self.watch_steps
                    self.remember(x, y, t)
                    return A.IGNITE
                return RandomWalk.act(self, obs, info)  # look for something to burn
            return RandomWalk.act(self, obs, info) if self.uniform()[0] < 0.3 else A.NOOP
        if self.target is not None:
            tx, ty = self.target
            if abs(tx - x) + abs(ty - y) <= 2:
                self.until = t  # arrived: choose again next step
                return A.NOOP
            return self.toward(obs, ty - y, tx - x)
        return RandomWalk.act(self, obs, info)


Player.nearest = Firefighter.nearest  # reuse the nearest-visible-cell helper

POLICIES: dict[str, type[Policy]] = {
    p.name: p
    for p in (RandomWalk, ExploreReturn, ExploreReturnAct, Firestarter, Firefighter, Player)
}


def make_policy(name: str, cfg: Config, seed: int = 0, params: dict | None = None) -> Policy:
    """Instantiate a policy; ``params`` override its tunable class attributes."""
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    pol = POLICIES[name](cfg, seed)
    for k, v in (params or {}).items():
        if k not in pol.params:
            raise ValueError(f"{name} has no parameter {k!r}; tunable: {pol.params}")
        setattr(pol, k, type(getattr(pol, k))(v))
    return pol
