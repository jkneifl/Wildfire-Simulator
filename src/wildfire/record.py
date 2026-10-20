"""Episode recording, replay and loading (SPEC 9)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .dynamics import step
from .observe import agent_window, grid, observe
from .render.topdown import render
from .state import WorldState, equal
from .terrain import initial_state
from .tiles import TileTracker


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _agent_row(s: WorldState) -> list[int]:
    return [int(s.ax), int(s.ay), int(s.facing), int(s.tank)]


def snapshot_path(ep_dir: Path, i: int) -> Path:
    return Path(ep_dir) / "state" / f"step_{i:06d}.npz"


class Recorder:
    """Records an episode starting from ``state`` (episode step 0).

    ``from_reset`` marks that ``state`` equals ``initial_state(cfg, seed, stream)``; replay
    then rebuilds from the seed, otherwise from the step-0 snapshot.
    """

    def __init__(
        self,
        out_dir: str | Path,
        cfg: Config,
        state: WorldState,
        policy: str = "human",
        from_reset: bool = False,
        rgb: bool = False,
    ):
        self.dir = Path(out_dir)
        (self.dir / "state").mkdir(parents=True, exist_ok=True)
        self.cfg, self.rgb = cfg, rgb
        self.meta = {
            "config": cfg.to_dict(),
            "seed": int(state.seed),
            "stream": int(state.stream),
            "start_step": int(state.t),
            "from_reset": from_reset,
            "policy": policy,
            "git": git_hash(),
            "snapshot_every": cfg.snapshot_every,
        }
        self.actions: list[int] = []
        self.agent: list[list[int]] = []
        self.grids: list[np.ndarray] = []
        self.rgbs: list[np.ndarray] = []
        self.tiles = TileTracker(state.shape, cfg.tile)
        self._observe(state, None)

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    def _observe(self, s: WorldState, obs: dict | None) -> None:
        win = agent_window(s, self.cfg)
        self.agent.append(_agent_row(s))
        self.grids.append(obs["grid"] if obs is not None else grid(s, *win, self.cfg))
        if self.rgb:
            self.rgbs.append(
                obs["rgb"] if obs is not None and "rgb" in obs else render(s, win, self.cfg)
            )
        self.tiles.mark(win, int(s.t))
        if self.n_steps % self.cfg.snapshot_every == 0:
            s.save(snapshot_path(self.dir, self.n_steps))
        self._last = s

    def add(self, action: int, state: WorldState, obs: dict | None = None) -> None:
        """Record ``action`` and the state (and optionally its observation) after it."""
        self.actions.append(int(action))
        self._observe(state, obs)

    def close(self, **extra) -> Path:
        if not snapshot_path(self.dir, self.n_steps).exists():
            self._last.save(snapshot_path(self.dir, self.n_steps))  # final state
        np.save(self.dir / "actions.npy", np.array(self.actions, np.int8))
        np.save(self.dir / "agent.npy", np.array(self.agent, np.int64))
        np.save(self.dir / "obs_grid.npy", np.stack(self.grids))
        if self.rgb:
            np.save(self.dir / "obs_rgb.npy", np.stack(self.rgbs))
        np.save(
            self.dir / "tiles_seen.npy", np.stack([self.tiles.first_seen, self.tiles.last_seen])
        )
        meta = {**self.meta, "n_steps": self.n_steps, **extra}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self.dir


class ReplayMismatch(AssertionError):
    pass


def load_config(ep_dir: str | Path) -> tuple[Config, dict]:
    meta = json.loads((Path(ep_dir) / "meta.json").read_text())
    return Config.from_dict(meta["config"]), meta


def replay(ep_dir: str | Path, check_obs: bool = True) -> dict:
    """Rebuild an episode from ``meta.json`` and ``actions.npy``; check every snapshot.

    Raises :class:`ReplayMismatch` on the first difference.
    """
    ep_dir = Path(ep_dir)
    cfg, meta = load_config(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    snaps = {int(p.stem.split("_")[1]): p for p in (ep_dir / "state").glob("step_*.npz")}
    if meta["from_reset"]:
        s = initial_state(cfg, meta["seed"], meta["stream"])
    else:
        s = WorldState.load(snaps[0])
    grids = np.load(ep_dir / "obs_grid.npy", mmap_mode="r") if check_obs else None
    agent = np.load(ep_dir / "agent.npy")
    checked = 0
    for i in range(len(actions) + 1):
        if i > 0:
            s = step(s, int(actions[i - 1]), cfg)
        if i in snaps:
            if not equal(s, WorldState.load(snaps[i])):
                raise ReplayMismatch(f"{ep_dir}: state differs from snapshot at step {i}")
            checked += 1
        if check_obs:
            if not np.array_equal(observe(s, cfg, rgb=False)["grid"], grids[i]):
                raise ReplayMismatch(f"{ep_dir}: observation differs at step {i}")
            if _agent_row(s) != agent[i].tolist():
                raise ReplayMismatch(f"{ep_dir}: agent differs at step {i}")
    return {"steps": len(actions), "snapshots_checked": checked}


class EpisodeDataset:
    """Episodes under ``root`` (directories containing ``meta.json``), NumPy only."""

    def __init__(self, root: str | Path, mmap: bool = True):
        root = Path(root)
        self.dirs = sorted(p.parent for p in root.glob("*/meta.json"))
        if (root / "meta.json").exists():
            self.dirs = [root]
        self.mmap = "r" if mmap else None

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int) -> dict:
        d = self.dirs[i]
        _, meta = load_config(d)
        ep = {"meta": meta, "dir": d}
        for name in ("actions", "agent", "obs_grid", "obs_rgb", "tiles_seen"):
            if (d / f"{name}.npy").exists():
                ep[name] = np.load(d / f"{name}.npy", mmap_mode=self.mmap)
        return ep

    def snapshot_steps(self, i: int) -> list[int]:
        return sorted(
            int(p.stem.split("_")[1]) for p in (self.dirs[i] / "state").glob("step_*.npz")
        )

    def state(self, i: int, step_index: int) -> WorldState:
        return WorldState.load(snapshot_path(self.dirs[i], step_index))
