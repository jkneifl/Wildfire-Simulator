"""Episode recording, replay and loading (SPEC 9).

Besides the SPEC 9 files, a :class:`Recorder` can write optional *products* for training data
(:class:`Products`), all at full cell resolution so that coarser representations are derived at
load time:

- ``near/chunk_XXXX.npz``: observable crop ``(2r+1, 2r+1, 6)`` around the agent, every step;
- ``world/step_XXXXXX.npz``: full-world observable grid ``(H, W, 6)`` every ``world_every`` steps;
- ``globals.npy``: per step ``t``, day phase, light (observable) and global stats (privileged);
- ``events.npz``: new ignitions with their cause (privileged, see :mod:`wildfire.events`).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .dynamics import light, step
from .events import ignitions
from .metrics import global_stats
from .observe import agent_window, grid, observe
from .render.topdown import render
from .state import WorldState, equal
from .terrain import initial_state
from .tiles import TileTracker

NEAR_CHUNK = 256
GLOBAL_COLUMNS = ("t", "phase", "light", "veg_density", "burned_frac", "active_fires", "mean_wet")
GLOBAL_PROBES = GLOBAL_COLUMNS[3:]  # privileged: probes only, never model inputs
EVENT_FIELDS = ("step", "y", "x", "cause", "nbrs")


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


def world_path(ep_dir: Path, i: int) -> Path:
    return Path(ep_dir) / "world" / f"step_{i:06d}.npz"


def near_crop(s: WorldState, r: int, cfg: Config) -> np.ndarray:
    """Observable crop of radius ``r`` centred on the agent (outside the world = rock)."""
    return grid(s, int(s.ay) - r, int(s.ax) - r, 2 * r + 1, 2 * r + 1, cfg)


def global_row(s: WorldState, cfg: Config) -> list[float]:
    st = global_stats(s)
    t = int(s.t)
    return [t, (t % cfg.day_length) / cfg.day_length, float(light(t, cfg.day_length))] + [
        float(st[k]) for k in GLOBAL_PROBES
    ]


@dataclass(frozen=True)
class Products:
    """Optional training-data products; ``None`` / ``False`` disables one."""

    near_radius: int | None = None
    world_every: int | None = None
    globals: bool = False
    events: bool = False


class Recorder:
    """Records an episode starting from ``state`` (episode step 0).

    ``from_reset`` marks that ``state`` equals ``initial_state(cfg, seed, stream)``; replay
    then rebuilds from the seed, otherwise from the step-0 snapshot. Products are written
    incrementally.
    """

    def __init__(
        self,
        out_dir: str | Path,
        cfg: Config,
        state: WorldState,
        policy: str = "human",
        from_reset: bool = False,
        rgb: bool = False,
        products: Products | None = None,
    ):
        self.dir = Path(out_dir)
        (self.dir / "state").mkdir(parents=True, exist_ok=True)
        products = products or Products()
        self.cfg, self.rgb, self.products = cfg, rgb, products
        if products.near_radius is not None:
            (self.dir / "near").mkdir(exist_ok=True)
        if products.world_every:
            (self.dir / "world").mkdir(exist_ok=True)
        self.meta = {
            "config": cfg.to_dict(),
            "seed": int(state.seed),
            "stream": int(state.stream),
            "start_step": int(state.t),
            "from_reset": from_reset,
            "policy": policy,
            "git": git_hash(),
            "snapshot_every": cfg.snapshot_every,
            "products": asdict(products),
        }
        self.actions: list[int] = []
        self.agent: list[list[int]] = []
        self.grids: list[np.ndarray] = []
        self.rgbs: list[np.ndarray] = []
        self.near: list[np.ndarray] = []
        self.globals: list[list[float]] = []
        self.events: dict[str, list[np.ndarray]] = {k: [] for k in EVENT_FIELDS}
        self.tiles = TileTracker(state.shape, cfg.tile)
        self._observe(state, None)

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    def _observe(self, s: WorldState, obs: dict | None) -> None:
        cfg, p, i = self.cfg, self.products, self.n_steps
        win = agent_window(s, cfg)
        self.agent.append(_agent_row(s))
        self.grids.append(obs["grid"] if obs is not None else grid(s, *win, cfg))
        if self.rgb:
            self.rgbs.append(
                obs["rgb"] if obs is not None and "rgb" in obs else render(s, win, cfg)
            )
        self.tiles.mark(win, int(s.t))
        if i % cfg.snapshot_every == 0:
            s.save(snapshot_path(self.dir, i))
        if p.near_radius is not None:
            self.near.append(near_crop(s, p.near_radius, cfg))
            if len(self.near) == NEAR_CHUNK:
                self._flush_near()
        if p.world_every and i % p.world_every == 0:
            h, w = s.shape
            np.savez_compressed(world_path(self.dir, i), grid=grid(s, 0, 0, h, w, cfg))
        if p.globals:
            self.globals.append(global_row(s, cfg))
        self._last = s

    def _flush_near(self) -> None:
        if self.near:
            k = (len(self.agent) - 1) // NEAR_CHUNK
            path = self.dir / "near" / f"chunk_{k:04d}.npz"
            np.savez_compressed(path, near=np.stack(self.near), start=k * NEAR_CHUNK)
            self.near = []

    def add(self, action: int, state: WorldState, obs: dict | None = None) -> None:
        """Record ``action`` and the state (and optionally its observation) after it."""
        if self.products.events:
            ev = ignitions(self._last, int(action), state, self.cfg)
            self.events["step"].append(np.full(ev["y"].size, self.n_steps + 1, np.int32))
            for k in ("y", "x", "cause", "nbrs"):
                self.events[k].append(ev[k])
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
        self._flush_near()
        if self.products.globals:
            np.save(self.dir / "globals.npy", np.array(self.globals, np.float64))
        if self.products.events:
            ev = {
                k: np.concatenate(v) if v else np.zeros(0, np.int32) for k, v in self.events.items()
            }
            np.savez_compressed(self.dir / "events.npz", **ev)
        meta = {
            **self.meta,
            "n_steps": self.n_steps,
            "global_columns": list(GLOBAL_COLUMNS),
            "global_probes": list(GLOBAL_PROBES),
            **extra,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self.dir


class ReplayMismatch(AssertionError):
    pass


def load_config(ep_dir: str | Path) -> tuple[Config, dict]:
    meta = json.loads((Path(ep_dir) / "meta.json").read_text())
    return Config.from_dict(meta["config"]), meta


def _initial(ep_dir: Path, cfg: Config, meta: dict) -> WorldState:
    if meta["from_reset"]:
        return initial_state(cfg, meta["seed"], meta["stream"])
    return WorldState.load(snapshot_path(ep_dir, 0))


def replay(ep_dir: str | Path, check_obs: bool = True) -> dict:
    """Rebuild an episode from ``meta.json`` and ``actions.npy``; check every stored file.

    Checks state snapshots, observations, the agent trajectory and, if present, the near
    crops, world grids, globals and events, all by array equality. Raises
    :class:`ReplayMismatch` on the first difference.
    """
    ep_dir = Path(ep_dir)
    cfg, meta = load_config(ep_dir)
    p = Products(**meta.get("products", {}))
    actions = np.load(ep_dir / "actions.npy")
    snaps = {int(q.stem.split("_")[1]): q for q in (ep_dir / "state").glob("step_*.npz")}
    s = _initial(ep_dir, cfg, meta)
    grids = np.load(ep_dir / "obs_grid.npy", mmap_mode="r") if check_obs else None
    agent = np.load(ep_dir / "agent.npy")
    ds = EpisodeDataset(ep_dir)
    near = ds.near(0) if p.near_radius is not None else None
    glob = np.load(ep_dir / "globals.npy") if p.globals else None
    events = dict(np.load(ep_dir / "events.npz")) if p.events else None
    seen_events = {k: [] for k in EVENT_FIELDS}
    checked = 0

    def fail(what: str, i: int):
        raise ReplayMismatch(f"{ep_dir}: {what} differs at step {i}")

    n = len(actions) + 1  # recorded steps including step 0
    lengths = {"agent": len(agent), "near crops": None if near is None else len(near),
               "globals": None if glob is None else len(glob)}  # fmt: skip
    for what, got in lengths.items():
        if got is not None and got != n:
            raise ReplayMismatch(f"{ep_dir}: {got} {what} rows for {n} steps")
    if p.world_every:
        expect = list(range(0, n, p.world_every))
        if ds.world_steps(0) != expect:
            raise ReplayMismatch(f"{ep_dir}: world grids at unexpected steps")

    for i in range(n):
        if i > 0:
            prev, s = s, step(s, int(actions[i - 1]), cfg)
            if events is not None:
                ev = ignitions(prev, int(actions[i - 1]), s, cfg)
                seen_events["step"].append(np.full(ev["y"].size, i, np.int32))
                for k in ("y", "x", "cause", "nbrs"):
                    seen_events[k].append(ev[k])
        if i in snaps:
            if not equal(s, WorldState.load(snaps[i])):
                fail("state snapshot", i)
            checked += 1
        if check_obs:
            if not np.array_equal(observe(s, cfg, rgb=False)["grid"], grids[i]):
                fail("observation", i)
            if _agent_row(s) != agent[i].tolist():
                fail("agent", i)
        if near is not None and not np.array_equal(near_crop(s, p.near_radius, cfg), near[i]):
            fail("near crop", i)
        if p.world_every and i % p.world_every == 0:
            h, w = s.shape
            if not np.array_equal(grid(s, 0, 0, h, w, cfg), ds.world(0, i)):
                fail("world grid", i)
        if glob is not None and not np.array_equal(np.array(global_row(s, cfg)), glob[i]):
            fail("globals", i)
    if events is not None:
        for k in EVENT_FIELDS:
            got = np.concatenate(seen_events[k]) if seen_events[k] else np.zeros(0)
            if not np.array_equal(got, events[k]):
                raise ReplayMismatch(f"{ep_dir}: events differ ({k})")
    return {"steps": len(actions), "snapshots_checked": checked}


def rerecord(
    ep_dir: str | Path, out_dir: str | Path, products: Products, rgb: bool = False
) -> Path:
    """Replay an episode (e.g. a human recording) and write it again with ``products``."""
    ep_dir = Path(ep_dir)
    cfg, meta = load_config(ep_dir)
    s = _initial(ep_dir, cfg, meta)
    rec = Recorder(out_dir, cfg, s, meta["policy"], meta["from_reset"], rgb, products)
    for a in np.load(ep_dir / "actions.npy"):
        s = step(s, int(a), cfg)
        rec.add(int(a), s)
    standard = set(rec.meta) | {"n_steps", "global_columns", "global_probes"}
    extra = {k: v for k, v in meta.items() if k not in standard}
    return rec.close(**extra, rerecorded_from=str(ep_dir))


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
        for name in ("actions", "agent", "obs_grid", "obs_rgb", "tiles_seen", "globals"):
            if (d / f"{name}.npy").exists():
                ep[name] = np.load(d / f"{name}.npy", mmap_mode=self.mmap)
        return ep

    def snapshot_steps(self, i: int) -> list[int]:
        return sorted(
            int(p.stem.split("_")[1]) for p in (self.dirs[i] / "state").glob("step_*.npz")
        )

    def state(self, i: int, step_index: int) -> WorldState:
        return WorldState.load(snapshot_path(self.dirs[i], step_index))

    def world_steps(self, i: int) -> list[int]:
        return sorted(
            int(p.stem.split("_")[1]) for p in (self.dirs[i] / "world").glob("step_*.npz")
        )

    def world(self, i: int, step_index: int) -> np.ndarray:
        """Full-world observable grid ``(H, W, 6)`` at an episode step."""
        with np.load(world_path(self.dirs[i], step_index)) as z:
            return z["grid"]

    def near(self, i: int) -> np.ndarray:
        """All near crops of an episode, ``(T+1, 2r+1, 2r+1, 6)`` (loaded into memory)."""
        chunks = sorted((self.dirs[i] / "near").glob("chunk_*.npz"))
        parts = []
        for c in chunks:
            with np.load(c) as z:
                parts.append(z["near"])
        return np.concatenate(parts)

    def events(self, i: int) -> dict[str, np.ndarray]:
        with np.load(self.dirs[i] / "events.npz") as z:
            return {k: z[k] for k in EVENT_FIELDS}
