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


# --- tile statistics -------------------------------------------------------------------

TILE_STATS = ("veg_density", "veg_mean", "burned_frac", "fire_frac", "wet_mean")


def tile_stats(s: WorldState, cfg: Config, tiles=None) -> np.ndarray:
    """Per-tile statistics over the ground cells of each tile.

    Returns ``(..., n_tiles, K)`` with ``K = len(TILE_STATS)``: vegetated fraction, mean
    stage / V_MAX, ash fraction, burning fraction, mean wetness / 255. ``tiles`` is a list of
    ``(ty, tx)``; default: all tiles in row-major order. Works on batched states.
    """
    T = cfg.tile
    h, w = s.shape
    ph, pw = -h % T, -w % T
    pad = [(0, 0)] * (s.veg.ndim - 2) + [(0, ph), (0, pw)]
    ground = np.pad(s.terrain == GROUND, pad)
    layers = [
        s.veg > 0,
        s.veg / cfg.v_max,
        s.ash_age > 0,
        s.fire > 0,
        s.wet / 255.0,
    ]
    rows, cols = (h + ph) // T, (w + pw) // T

    def tsum(a):
        a = np.pad(a, pad) * ground
        return a.reshape(*a.shape[:-2], rows, T, cols, T).sum(axis=(-3, -1))

    n = np.maximum(tsum(np.ones(s.veg.shape)), 1)
    out = np.stack([tsum(a) / n for a in layers], axis=-1)  # (..., rows, cols, K)
    out = out.reshape(*out.shape[:-3], rows * cols, len(TILE_STATS))
    if tiles is not None:
        out = out[..., [ty * cols + tx for ty, tx in tiles], :]
    return out


def monte_carlo(s: WorldState, cfg: Config, steps: int, n: int, tiles=None) -> np.ndarray:
    """Fork ``n`` futures for ``steps`` no-op steps; tile statistics ``(n, n_tiles, K)``.

    These are samples of the reference distribution of the tiles after an absence.
    """
    from .env import fork

    return tile_stats(fork(s, cfg, n, steps), cfg, tiles)


def burn_probability(s: WorldState, cfg: Config, n: int = 16, steps: int = 100) -> np.ndarray:
    """Fraction of ``n`` futures in which each cell burns within ``steps`` steps."""
    from .env import fork

    b = fork(s, cfg, n, 0)
    burned = b.fire > 0
    for _ in range(steps):
        b = step(b, 0, cfg)
        burned |= b.fire > 0
    return burned.mean(axis=0)


def retention_curve(series, lags, stat: str = "veg_density") -> np.ndarray:
    """Pearson correlation between a tile statistic at ``tau`` and at ``tau + lag``.

    ``series``: list (episodes) of arrays ``(n_snapshots, n_tiles, K)`` with equally spaced
    snapshots; ``lags`` in snapshot units. Pairs are pooled over tiles, ``tau`` and episodes.
    The correlation includes static between-tile differences (terrain), so it does not
    decay to 0.
    """
    k = TILE_STATS.index(stat)
    out = []
    for lag in lags:
        xs, ys = [], []
        for a in series:
            if lag < len(a):
                xs.append(a[: len(a) - lag, :, k].ravel())
                ys.append(a[lag:, :, k].ravel())
        if not xs:
            out.append(np.nan)
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        out.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan)
    return np.array(out)


def episode_tile_series(ep_dir, tiles=None) -> tuple[np.ndarray, np.ndarray]:
    """Tile statistics at the stored snapshots of a recorded episode: ``(steps, series)``.

    The final snapshot is dropped if it breaks the regular spacing.
    """
    from .record import EpisodeDataset, load_config

    cfg, _ = load_config(ep_dir)
    ds = EpisodeDataset(ep_dir)
    steps = np.array(ds.snapshot_steps(0))
    if len(steps) > 2 and steps[-1] - steps[-2] != steps[1] - steps[0]:
        steps = steps[:-1]
    return steps, np.stack([tile_stats(ds.state(0, int(i)), cfg, tiles) for i in steps])


# --- distances between sample sets -----------------------------------------------------


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x.reshape(len(x), -1)


def energy_distance(x, y) -> float:
    """Energy distance ``2 E|X-Y| - E|X-X'| - E|Y-Y'|`` (V-statistic) between sample sets."""
    from scipy.spatial.distance import cdist

    x, y = _as_2d(x), _as_2d(y)
    return float(2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean())


def mmd(x, y, bandwidth: float | None = None) -> float:
    """Unbiased squared MMD with a Gaussian kernel (median heuristic bandwidth by default)."""
    from scipy.spatial.distance import cdist

    x, y = _as_2d(x), _as_2d(y)
    dxx, dyy, dxy = (
        cdist(x, x, "sqeuclidean"),
        cdist(y, y, "sqeuclidean"),
        cdist(x, y, "sqeuclidean"),
    )
    if bandwidth is None:
        d = np.sqrt(np.concatenate([dxx.ravel(), dyy.ravel(), dxy.ravel()]))
        bandwidth = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    g = 1 / (2 * bandwidth**2)
    kxx, kyy, kxy = np.exp(-g * dxx), np.exp(-g * dyy), np.exp(-g * dxy)
    n, m = len(x), len(y)
    sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) if n > 1 else 0.0
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) if m > 1 else 0.0
    return float(sxx + syy - 2 * kxy.mean())


# --- statistics on observable grids (observe.grid channels) --------------------------------


def obs_stats(g: np.ndarray, cfg: Config) -> np.ndarray:
    """``TILE_STATS`` of observable grids ``(..., h, w, 6)`` over their ground cells.

    Same quantities as :func:`tile_stats`, except that wetness is the mean wet *bucket* / 3,
    since the exact wetness is not observable. Returns ``(..., len(TILE_STATS))``.
    """
    g = np.asarray(g)
    ground = (g[..., 0] == GROUND).astype(np.float64)
    n = np.maximum(ground.sum((-2, -1)), 1)
    layers = (g[..., 1] > 0, g[..., 1] / cfg.v_max, g[..., 5] > 0, g[..., 2] > 0, g[..., 4] / 3)
    return np.stack([(np.asarray(x, np.float64) * ground).sum((-2, -1)) / n for x in layers], -1)


def obs_tiles(world: np.ndarray, T: int) -> np.ndarray:
    """Split a world grid ``(H, W, 6)`` into tiles ``(rows, cols, T, T, 6)``; partial edge
    tiles are padded with rock (excluded from ``obs_stats``, which counts ground cells)."""
    h, w = world.shape[:2]
    ph, pw = -h % T, -w % T
    if ph or pw:
        pad = np.zeros((h + ph, w + pw, 6), world.dtype)
        pad[..., 0] = 2  # rock
        pad[:h, :w] = world
        world = pad
    rows, cols = world.shape[0] // T, world.shape[1] // T
    return world.reshape(rows, T, cols, T, 6).swapaxes(1, 2)


def obs_patch_stats(world: np.ndarray) -> dict[str, np.ndarray]:
    """Component sizes of burned (ash bucket > 0) and vegetated cells of an observable grid."""
    return {"burned": patch_sizes(world[..., 5] > 0), "vegetated": patch_sizes(world[..., 1] > 0)}


# --- behaviour statistics (agent trajectories, for comparing policies with human play) ----

SCALAR_BEHAVIOUR = (
    "frac_move", "frac_idle", "frac_ignite", "frac_water", "move_success", "fire_visible_frac",
)  # fmt: skip
SAMPLE_BEHAVIOUR = ("straight_runs", "idle_runs", "dist_fire_at_water", "revisit_gaps")
COVERAGE_STEPS = (250, 500, 1000, 2000, 4000)


def _runs(mask: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    """Lengths of maximal runs where ``mask`` is true (and ``values`` stays constant)."""
    if not mask.any():
        return np.zeros(0, np.int64)
    key = np.where(mask, 1 if values is None else values, -1)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(key) != 0) + 1, [len(key)]])
    starts, lengths = bounds[:-1], np.diff(bounds)
    return lengths[mask[starts]]


def behaviour_stats(actions, agent, obs_grid, cfg: Config) -> dict:
    """Statistics of one episode's behaviour from ``actions``, ``agent`` and ``obs_grid``.

    Uses only what the agent did and saw. Scalars are listed in ``SCALAR_BEHAVIOUR``, sample
    arrays in ``SAMPLE_BEHAVIOUR``; ``coverage`` is the number of tiles seen after
    ``COVERAGE_STEPS`` steps (clipped to the episode length).
    """
    from .tiles import absences, visibility

    a = np.asarray(actions)
    ag = np.asarray(agent)
    move = (a >= 1) & (a <= 4)
    moved = np.any(ag[1:, :2] != ag[:-1, :2], axis=1)
    r = cfg.window // 2
    burning = np.asarray(obs_grid)[..., 2] > 0  # (T+1, V, V)
    dist = []
    for i in np.flatnonzero(a == 6):
        ys, xs = np.nonzero(burning[i])
        if ys.size:
            dist.append(int(np.min(np.maximum(abs(ys - r), abs(xs - r)))))
    vis = visibility(ag, r, cfg.tile, (cfg.height, cfg.width))
    gaps = absences(vis)
    gaps = gaps[gaps[:, 3] >= 0]
    ever = np.maximum.accumulate(vis.reshape(len(vis), -1), axis=0).sum(1)  # tiles seen so far
    frac = (lambda m: float(m.mean())) if len(a) else (lambda m: 0.0)  # noqa: E731
    return {
        "frac_move": frac(move),
        "frac_idle": frac(a == 0),
        "frac_ignite": frac(a == 5),
        "frac_water": frac(a == 6),
        "move_success": float(moved[move].mean()) if move.any() else 0.0,
        "fire_visible_frac": float(burning.any((1, 2)).mean()),
        "straight_runs": _runs(move, a),
        "idle_runs": _runs(a == 0),
        "dist_fire_at_water": np.array(dist, np.int64),
        "revisit_gaps": gaps[:, 3] - gaps[:, 2],
        "coverage": np.array([ever[min(k, len(ever) - 1)] for k in COVERAGE_STEPS]),
        "n_steps": len(a),
    }


def behaviour_distance(a: list[dict], b: list[dict]) -> dict[str, float]:
    """Per-statistic energy distance between two sets of episodes' ``behaviour_stats``.

    Scalars and coverage are compared as one value per episode; sample statistics pool all
    samples of the episodes (log1p-transformed, so long runs and gaps do not dominate).
    """
    out = {}
    for k in SCALAR_BEHAVIOUR:
        out[k] = energy_distance([e[k] for e in a], [e[k] for e in b])
    for k in SAMPLE_BEHAVIOUR:
        xa = np.concatenate([e[k] for e in a]) if a else np.zeros(0)
        xb = np.concatenate([e[k] for e in b]) if b else np.zeros(0)
        out[k] = energy_distance(np.log1p(xa), np.log1p(xb)) if xa.size and xb.size else np.nan
    out["coverage"] = energy_distance(
        np.log1p([e["coverage"] for e in a]), np.log1p([e["coverage"] for e in b])
    )
    return out
