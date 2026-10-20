"""Retention curve and Monte Carlo tile distributions for a few absences (SPEC M5).

    uv run python scripts/evaluate.py --out docs/evaluation
    uv run python scripts/evaluate.py --episodes data/test   # retention from recorded episodes

Retention: correlation of tile statistics between ``tau`` and ``tau + dt`` over all tiles of
headless runs (snapshots every ``--every`` steps after ``--burn-in`` days), or over the
snapshots of recorded episodes. Monte Carlo: from the state at ``--tau-day`` of seed 0,
``--n`` futures per absence ``dt``; per tile, the distribution of vegetation density after
``dt``, its energy distance to the last-seen value (at ``tau``), and between
absences.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wildfire import Config
from wildfire.metrics import (
    TILE_STATS,
    energy_distance,
    episode_tile_series,
    mmd,
    monte_carlo,
    retention_curve,
    simulate_stats,
    tile_stats,
)
from wildfire.record import EpisodeDataset


def series_job(job):
    cfg_dict, seed, steps, every, burn_in = job
    cfg = Config.from_dict(cfg_dict)
    st = simulate_stats(cfg, seed, steps, every=steps, keep_every=every)
    states = st["states"][burn_in // every :]
    return np.stack([tile_stats(s, cfg) for s in states])


def mc_job(job):
    cfg_dict, state, dt, n, tiles = job
    return dt, monte_carlo(state, Config.from_dict(cfg_dict), dt, n, tiles)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--burn-in", type=int, default=5, help="days")
    p.add_argument("--every", type=int, default=50, help="snapshot spacing (steps)")
    p.add_argument("--episodes", default=None, help="retention from recorded episodes instead")
    p.add_argument("--dts", type=int, nargs="+", default=[50, 200, 800])
    p.add_argument("--n", type=int, default=32, help="Monte Carlo futures per absence")
    p.add_argument("--tau-day", type=int, default=10)
    p.add_argument("--tiles", type=int, default=4, help="number of tiles for Monte Carlo")
    p.add_argument("--out", default="docs/evaluation")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    D = cfg.day_length
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- retention curve ---------------------------------------------------------------
    if args.episodes:
        ds = EpisodeDataset(args.episodes)
        pairs = [episode_tile_series(d) for d in ds.dirs]
        every = int(np.diff(pairs[0][0][:2])[0])
        series = [s for _, s in pairs]
        source = f"{len(series)} episodes in {args.episodes}"
    else:
        jobs = [(cfg.to_dict(), s, args.days * D, args.every, args.burn_in * D) for s in args.seeds]
        with Pool(min(len(jobs), 10)) as pool:
            series = pool.map(series_job, jobs)
        every = args.every
        source = f"headless no-op runs, seeds {args.seeds}, days {args.burn_in}-{args.days}"
    max_lag = min(len(s) for s in series) - 1
    lags = np.arange(1, max_lag + 1)
    curves = {stat: retention_curve(series, lags, stat) for stat in ("veg_density", "burned_frac")}

    # --- Monte Carlo tile distributions ------------------------------------------------
    st = simulate_stats(cfg, args.seeds[0], args.tau_day * D, every=args.tau_day * D)
    tau_state = st["final"]
    all_tiles = tile_stats(tau_state, cfg)
    cols = -(-cfg.width // cfg.tile)
    # tiles with the most vegetation and the most ash at tau (fixed, reproducible choice)
    k_veg, k_ash = TILE_STATS.index("veg_density"), TILE_STATS.index("burned_frac")
    half = max(args.tiles // 2, 1)
    idx = list(np.argsort(-all_tiles[:, k_veg])[:half]) + list(
        np.argsort(-all_tiles[:, k_ash])[: args.tiles - half]
    )
    tiles = [(int(i) // cols, int(i) % cols) for i in idx]
    last_seen = all_tiles[idx]  # last-seen value at tau
    with Pool(min(len(args.dts), 10)) as pool:
        mc = dict(
            pool.map(mc_job, [(cfg.to_dict(), tau_state, dt, args.n, tiles) for dt in args.dts])
        )

    mc_summary = []
    for j, tile in enumerate(tiles):
        stats_at_tau = dict(zip(TILE_STATS, last_seen[j].round(4).tolist()))
        row = {"tile": list(tile), "last_seen": stats_at_tau}
        for dt in args.dts:
            x = mc[dt][:, j, :]
            row[f"dt={dt}"] = {
                "veg_density_mean": float(x[:, k_veg].mean()),
                "veg_density_std": float(x[:, k_veg].std()),
                "energy_to_last_seen": energy_distance(x, last_seen[j][None]),
            }
        for a, b in zip(args.dts, args.dts[1:]):
            row[f"energy dt={a} vs dt={b}"] = energy_distance(mc[a][:, j], mc[b][:, j])
            row[f"mmd2 dt={a} vs dt={b}"] = mmd(mc[a][:, j], mc[b][:, j])
        mc_summary.append(row)

    report = {
        "config": args.config,
        "retention": {
            "source": source,
            "snapshot_spacing_steps": every,
            "n_tiles": int(series[0].shape[1]),
            "n_series": len(series),
            "lag_steps": (lags * every).tolist(),
            **{k: v.round(4).tolist() for k, v in curves.items()},
        },
        "monte_carlo": {
            "seed": args.seeds[0],
            "tau_step": int(tau_state.t),
            "n_futures": args.n,
            "dts": args.dts,
            "tiles": mc_summary,
        },
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in curves.items():
        ax.plot(lags * every, v, marker=".", label=k)
    ax.set_xscale("log")
    ax.set_xlabel("absence Δt (steps)")
    ax.set_ylabel("corr(tile stat at τ, at τ+Δt)")
    ax.set_title(
        f"retention, {series[0].shape[1]} tiles x {len(series)} runs\n{source}", fontsize=8
    )
    ax.axhline(0, color="0.7", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "retention.png", dpi=120)

    fig, axes = plt.subplots(1, len(tiles), figsize=(3.2 * len(tiles), 3), sharey=True)
    for j, (ax, tile) in enumerate(zip(np.atleast_1d(axes), tiles)):
        for dt in args.dts:
            ax.hist(mc[dt][:, j, k_veg], bins=np.linspace(0, 1, 26), alpha=0.5, label=f"Δt={dt}")
        ax.axvline(last_seen[j, k_veg], color="k", ls="--", label="at τ (last seen)")
        ax.set_title(f"tile {tile}", fontsize=9)
        ax.set_xlabel("veg density")
    np.atleast_1d(axes)[0].set_ylabel(f"futures (of {args.n})")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle(f"Monte Carlo from seed {args.seeds[0]}, τ = step {int(tau_state.t)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "monte_carlo.png", dpi=120)
    print(json.dumps(report["monte_carlo"]["tiles"], indent=1))
    print(
        "retention veg_density:",
        dict(zip((lags * every).tolist(), curves["veg_density"].round(3).tolist())),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
