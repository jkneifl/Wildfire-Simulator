"""Monte Carlo reference sets: many seeded futures of one recorded state (evaluation).

A fork set takes the full ``WorldState`` of a recorded episode at step ``tau`` (rebuilt by
replay from the nearest state snapshot), runs ``n`` futures that differ only in their stream
id with the episode's **recorded actions** from ``tau`` on (the agent's later behaviour is part
of the conditioning; moves and refills depend only on terrain, so the agent trajectory is the
same in every future), and stores the observable state of selected tiles at nested
checkpoints ``tau + dt``. Nested checkpoints along the same futures also give the
revisit-schedule comparison (one jump vs two).

Note: the reference conditions on the *full* world at ``tau``; a model that conditions on
less (a stored tile, its neighbourhood, a clock) should show at least this much spread.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dynamics import step
from .env import fork_streams
from .metrics import energy_distance, obs_stats
from .observe import grid
from .record import EpisodeDataset, _initial, load_config, snapshot_path
from .rng import Purpose, uniform
from .state import WorldState, stack
from .tiles import absences, visibility

DEFAULT_DTS = tuple(2**k for k in range(3, 12))  # 8 .. 2048


def state_at(ep_dir: str | Path, tau: int) -> WorldState:
    """Full state at episode step ``tau``: nearest earlier snapshot, then recorded actions."""
    ep_dir = Path(ep_dir)
    cfg, meta = load_config(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    snaps = EpisodeDataset(ep_dir).snapshot_steps(0)
    start = max([k for k in snaps if k <= tau], default=None)
    s = (
        _initial(ep_dir, cfg, meta)
        if start is None
        else WorldState.load(snapshot_path(ep_dir, start))
    )
    for k in range(start or 0, tau):
        s = step(s, int(actions[k]), cfg)
    return s


def build(
    ep_dir: str | Path,
    tau: int,
    tiles: list[tuple[int, int]],
    n: int = 32,
    dts: tuple[int, ...] = DEFAULT_DTS,
    world_dts: tuple[int, ...] = (),
    tile: int = 16,
    out: str | Path | None = None,
) -> dict:
    """Run the futures and return (and optionally save as ``.npz`` + ``.json``) the fork set."""
    ep_dir = Path(ep_dir)
    cfg, meta = load_config(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    dts = tuple(sorted(dts))
    if tau + dts[-1] > len(actions):
        raise ValueError(
            f"tau + max dt = {tau + dts[-1]} exceeds the {len(actions)} recorded steps"
        )
    s = state_at(ep_dir, tau)
    world_file = ep_dir / "world" / f"step_{tau:06d}.npz"
    if world_file.exists():  # consistency with the recording
        h, w = s.shape
        if not np.array_equal(grid(s, 0, 0, h, w, cfg), np.load(world_file)["grid"]):
            raise AssertionError(f"replayed state at {tau} differs from the recorded world grid")
    streams = fork_streams(s, n)
    b = stack([s] * n)
    b.stream = streams.copy()
    T = tile
    tile_obs = np.zeros((n, len(dts), len(tiles), T, T, 6), np.uint8)
    worlds = {}
    agent = []
    checkpoints = {dt: j for j, dt in enumerate(dts)}
    for k in range(1, dts[-1] + 1):
        b = step(b, int(actions[tau + k - 1]), cfg)
        if not (np.all(b.ax == b.ax[0]) and np.all(b.ay == b.ay[0])):
            raise AssertionError("agent trajectories differ between futures")
        agent.append((int(b.ax[0]), int(b.ay[0])))
        if k in checkpoints or k in world_dts:
            for i in range(n):
                si = b[i]
                if k in checkpoints:
                    for m, (ty, tx) in enumerate(tiles):
                        tile_obs[i, checkpoints[k], m] = grid(si, ty * T, tx * T, T, T, cfg)
                if k in world_dts:
                    h, w = si.shape
                    worlds.setdefault(k, []).append(grid(si, 0, 0, h, w, cfg))
    tile_tau = np.stack([grid(s, ty * T, tx * T, T, T, cfg) for ty, tx in tiles])
    fs = {
        "tile_obs": tile_obs,  # (n, n_dts, n_tiles, T, T, 6)
        "tile_tau": tile_tau,  # (n_tiles, T, T, 6): the frozen baseline
        "dts": np.array(dts),
        "tiles": np.array(tiles, np.int64).reshape(-1, 2),
        "streams": streams,
        "agent": np.array(agent, np.int64),
        **{f"world_dt{k}": np.stack(v) for k, v in worlds.items()},
    }
    info = {
        "episode": str(ep_dir),
        "tau": tau,
        "n": n,
        "tile": T,
        "dts": list(dts),
        "world_dts": sorted(worlds),
        "seed": meta["seed"],
        "stream": meta["stream"],
        "config": cfg.to_dict(),
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix(".npz"), **fs)
        out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return {**fs, "info": info}


def pick_taus(ep_dir: str | Path, n_taus: int, dt_max: int, tile: int = 16, seed: int = 0):
    """``(tau, exiting tiles)`` pairs: steps where tiles leave the agent window, early enough
    that ``tau + dt_max`` steps were recorded; spread over the episode."""
    ds = EpisodeDataset(ep_dir)
    cfg, _ = load_config(ep_dir)
    agent = np.asarray(ds[0]["agent"])
    a = absences(visibility(agent, cfg.window // 2, tile, (cfg.height, cfg.width)))
    a = a[a[:, 2] + dt_max <= len(agent) - 1]
    if not len(a):
        return []
    taus = np.unique(a[:, 2])
    pick = taus[np.linspace(0, len(taus) - 1, min(n_taus, len(taus))).round().astype(int)]
    return [(int(t), [tuple(map(int, r[:2])) for r in a[a[:, 2] == t]]) for t in pick]


def add_random_tiles(tiles, k: int, shape, tile: int, seed: int, tau: int):
    rows, cols = -(-shape[0] // tile), -(-shape[1] // tile)
    u = uniform(seed, 3, tau, Purpose.SAMPLER, np.zeros(2 * k, int), np.arange(2 * k))
    extra = [(int(u[2 * j] * rows), int(u[2 * j + 1] * cols)) for j in range(k)]
    return list(dict.fromkeys(list(tiles) + extra))


def reference_stats(fs: dict, cfg) -> np.ndarray:
    """``obs_stats`` of every future, checkpoint and tile: ``(n, n_dts, n_tiles, K)``."""
    return obs_stats(fs["tile_obs"], cfg)


def split_half_floor(stats: np.ndarray) -> np.ndarray:
    """Energy distance between the two halves of the futures, per checkpoint and tile: the
    noise floor of comparing a sample set of half this size with the reference."""
    n = stats.shape[0] // 2
    out = np.zeros(stats.shape[1:3])
    for j in range(stats.shape[1]):
        for m in range(stats.shape[2]):
            out[j, m] = energy_distance(stats[:n, j, m], stats[n : 2 * n, j, m])
    return out


def frozen_distance(fs: dict, cfg) -> np.ndarray:
    """Energy distance between the reference and the frozen tile (its state at ``tau``)."""
    ref = reference_stats(fs, cfg)
    frozen = obs_stats(fs["tile_tau"], cfg)  # (n_tiles, K)
    return np.array(
        [[energy_distance(ref[:, j, m], frozen[m][None]) for m in range(ref.shape[2])]
         for j in range(ref.shape[1])]
    )  # fmt: skip


def resample_baseline(world_grids, phases, phase, tile: int, n: int, cfg, tol=0.05, seed=0):
    """'Resample without memory': ``n`` random tiles from world grids recorded at a similar
    day phase (``|phase - p| <= tol``, circular). ``world_grids`` is a list of ``(H, W, 6)``
    grids with matching ``phases``. Returns ``(n, T, T, 6)``."""
    phases = np.asarray(phases)
    d = np.abs((phases - phase + 0.5) % 1 - 0.5)
    pool = np.flatnonzero(d <= tol)
    if not pool.size:
        raise ValueError(f"no world grids within phase {phase} +- {tol}")
    H, W = world_grids[0].shape[:2]
    rows, cols = H // tile, W // tile
    u = uniform(seed, 4, 0, Purpose.SAMPLER, np.zeros(3 * n, int), np.arange(3 * n)).reshape(n, 3)
    out = []
    for a, b, c in u:
        g = world_grids[pool[int(a * pool.size)]]
        ty, tx = int(b * rows), int(c * cols)
        out.append(g[ty * tile : (ty + 1) * tile, tx * tile : (tx + 1) * tile])
    return np.stack(out)
