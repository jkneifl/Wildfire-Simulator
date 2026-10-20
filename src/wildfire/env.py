"""Environment API (SPEC 7): ``WildfireEnv`` and the batched ``VecWildfire``."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import actions as A
from . import rng
from .config import Config
from .dynamics import step
from .observe import CHANNELS, agent_window, observe, privileged
from .rng import Purpose
from .state import WorldState, stack
from .terrain import initial_state
from .tiles import TileTracker

RewardFn = Callable[[WorldState, WorldState], float]


def fork_streams(s: WorldState, n: int) -> np.ndarray:
    """``n`` stream ids for futures of an unbatched state, distinct per (stream, t, i)."""
    k = [rng.key(int(s.stream), i, int(s.t), Purpose.FORK) & (2**63 - 1) for i in range(n)]
    return np.array(k, np.int64)


def fork(s: WorldState, cfg: Config, n: int, steps: int, streams=None, actions=None) -> WorldState:
    """Run ``n`` futures of an unbatched state that differ only in their stream id.

    ``actions``: None (noop), shape ``(steps,)`` or ``(steps, n)``. Returns the batched
    state after ``steps`` steps.
    """
    streams = fork_streams(s, n) if streams is None else np.asarray(streams, np.int64)
    b = stack([s] * n)
    b.stream = streams.copy()
    for i in range(steps):
        b = step(b, A.NOOP if actions is None else actions[i], cfg)
    return b


def tile_info(tracker: TileTracker, s: WorldState, cfg: Config) -> dict:
    """Mark the current window as seen and return the (non-privileged) info dict."""
    tiles = tracker.mark(agent_window(s, cfg), int(s.t))
    return {
        "step": int(s.t),
        "pos": (int(s.ax), int(s.ay)),
        "tiles": tiles,
        "first_seen": tracker.first_seen.copy(),
        "last_seen": tracker.last_seen.copy(),
    }


def _spaces(cfg: Config, rgb: bool):
    """Per-world Gymnasium spaces (None if gymnasium is not installed)."""
    try:
        from gymnasium import spaces
    except ImportError:  # gymnasium is an optional extra
        return None, None
    V, px = cfg.window, cfg.window * cfg.pixels_per_cell
    obs = {
        "grid": spaces.Box(0, 255, (V, V, len(CHANNELS)), np.uint8),
        "light": spaces.Box(0.0, 1.0, (), np.float64),
        "agent": spaces.Dict({"facing": spaces.Discrete(4), "tank": spaces.Discrete(cfg.tank + 1)}),
    }
    if rgb:
        obs["rgb"] = spaces.Box(0, 255, (px, px, 3), np.uint8)
    return spaces.Discrete(A.N_ACTIONS), spaces.Dict(obs)


class WildfireEnv:
    """Single world. ``obs`` is observable only; privileged state via ``get_state``."""

    def __init__(self, cfg: Config, reward_fn: RewardFn | None = None, rgb: bool = True):
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.rgb = rgb
        self.action_space, self.observation_space = _spaces(cfg, rgb)
        self.state: WorldState | None = None
        self.tiles: TileTracker | None = None

    def reset(self, seed: int = 0, stream: int = 0, options=None):
        self.state = initial_state(self.cfg, seed, stream)
        self.tiles = TileTracker(self.state.shape, self.cfg.tile)
        return self.observe(), self._info()

    def step(self, action: int):
        prev = self.state
        self.state = step(prev, int(action), self.cfg)
        reward = 0.0 if self.reward_fn is None else float(self.reward_fn(prev, self.state))
        return self.observe(), reward, False, False, self._info()

    def observe(self) -> dict:
        return observe(self.state, self.cfg, rgb=self.rgb)

    def privileged(self) -> dict:
        return privileged(self.state)

    def _info(self) -> dict:
        return tile_info(self.tiles, self.state, self.cfg)

    def get_state(self) -> WorldState:
        return self.state.copy()

    def set_state(self, state: WorldState) -> None:
        """Replace the world state. The tile tracker is kept (it is the agent's history)."""
        self.state = state.copy()
        if self.tiles is None or self.tiles.shape != state.shape:
            self.tiles = TileTracker(state.shape, self.cfg.tile)

    def fork(self, n: int = 16, steps: int = 200, streams=None, actions=None) -> WorldState:
        return fork(self.state, self.cfg, n, steps, streams, actions)


class VecWildfire:
    """``n`` worlds stepped together with arrays of shape ``(n, H, W)``."""

    def __init__(self, cfg: Config, n: int, reward_fn: RewardFn | None = None, rgb: bool = False):
        self.cfg, self.n, self.reward_fn, self.rgb = cfg, n, reward_fn, rgb
        self.action_space, self.observation_space = _spaces(cfg, rgb)
        self.state: WorldState | None = None

    def reset(self, seeds=None, streams=None):
        seeds = range(self.n) if seeds is None else seeds
        streams = [0] * self.n if streams is None else streams
        return self.set_state(
            stack([initial_state(self.cfg, s, k) for s, k in zip(seeds, streams)])
        )

    def set_state(self, state: WorldState):
        """Replace all worlds (fresh tile trackers); returns ``(obs, infos)``."""
        assert state.batched and state.veg.shape[0] == self.n
        self.state = state.copy()
        self.tiles = [TileTracker(state.shape, self.cfg.tile) for _ in range(self.n)]
        return self._obs(), self._infos()

    def get_state(self) -> WorldState:
        return self.state.copy()

    def step(self, actions):
        prev = self.state
        self.state = step(prev, np.asarray(actions), self.cfg)
        if self.reward_fn is None:
            rewards = np.zeros(self.n)
        else:
            rewards = np.array([self.reward_fn(prev[i], self.state[i]) for i in range(self.n)])
        done = np.zeros(self.n, bool)
        return self._obs(), rewards, done, done.copy(), self._infos()

    def _obs(self) -> dict:
        obs = [observe(self.state[i], self.cfg, rgb=self.rgb) for i in range(self.n)]
        out = {
            "grid": np.stack([o["grid"] for o in obs]),
            "light": np.array([o["light"] for o in obs]),
            "agent": {
                "facing": np.array([o["agent"]["facing"] for o in obs]),
                "tank": np.array([o["agent"]["tank"] for o in obs]),
            },
        }
        if self.rgb:
            out["rgb"] = np.stack([o["rgb"] for o in obs])
        return out

    def _infos(self) -> list[dict]:
        return [tile_info(self.tiles[i], self.state[i], self.cfg) for i in range(self.n)]
