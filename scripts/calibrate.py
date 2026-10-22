"""Calibration runs (SPEC M4): long headless simulations, regime numbers and plots.

    uv run python scripts/calibrate.py --config configs/default.yaml --seeds 0 1 2 --days 30
    uv run python scripts/calibrate.py --set beta=0.2 --set p_lightning=2e-6 --out /tmp/cal

Writes ``timeseries.png``, ``patches.png`` and ``summary.json`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wildfire import Config
from wildfire.metrics import patch_stats, regime_summary, simulate_stats


def run(job):
    cfg_dict, seed, days, burn_in = job
    cfg = Config.from_dict(cfg_dict)
    st = simulate_stats(cfg, seed, days * cfg.day_length, keep_every=cfg.day_length)
    summary = regime_summary(st, cfg, burn_in=burn_in * cfg.day_length)
    patches = [patch_stats(s) for s in st["states"][burn_in:]]
    series = {k: st[k] for k in ("t", "veg_density", "burned_frac", "active_fires")}
    return (
        seed,
        summary,
        series,
        {k: np.concatenate([p[k] for p in patches]) for k in ("burned", "vegetated")},
    )


def ccdf(sizes: np.ndarray):
    x = np.sort(sizes)
    return x, 1.0 - np.arange(len(x)) / len(x)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--burn-in", type=int, default=5, help="days excluded from the summary")
    p.add_argument("--set", action="append", default=[], help="override, e.g. beta=0.2")
    p.add_argument("--out", default="docs/calibration")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    cfg, overrides = Config.from_yaml(args.config).with_overrides(args.set)
    jobs = [(cfg.to_dict(), s, args.days, args.burn_in) for s in args.seeds]
    with Pool(min(len(jobs), 10)) as pool:
        results = pool.map(run, jobs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {seed: s for seed, s, _, _ in results}
    keys = list(next(iter(summaries.values())))
    mean = {k: float(np.mean([s[k] for s in summaries.values()])) for k in keys}
    for seed, s in summaries.items():
        print(f"seed {seed}: " + "  ".join(f"{k}={v:.3g}" for k, v in s.items()))
    print("mean:   " + "  ".join(f"{k}={v:.3g}" for k, v in mean.items()))
    report = {
        "config": args.config,
        "overrides": overrides,
        "world": f"{cfg.height}x{cfg.width}",
        "days": args.days,
        "burn_in_days": args.burn_in,
        "seeds": args.seeds,
        "per_seed": summaries,
        "mean": mean,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    if args.no_plots:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{cfg.height}x{cfg.width}, {args.config} {overrides or ''}"
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for seed, _, ser, _ in results:
        days = ser["t"] / cfg.day_length
        axes[0].plot(days, ser["veg_density"], lw=0.8, label=f"seed {seed}")
        axes[1].plot(days, ser["burned_frac"], lw=0.8)
        axes[2].plot(days, ser["active_fires"], lw=0.6)
    axes[0].set_ylabel("vegetation density")
    axes[1].set_ylabel("burned (ash) fraction")
    axes[2].set_ylabel("active fires (cells)")
    axes[2].set_xlabel("day")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.axvspan(0, args.burn_in, color="0.9", lw=0)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "timeseries.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, c in (("burned", "tab:red"), ("vegetated", "tab:green")):
        sizes = np.concatenate([r[3][k] for r in results])
        if sizes.size:
            ax.loglog(*ccdf(sizes), color=c, label=f"{k} (n={sizes.size})")
    ax.set_xlabel("component size (cells)")
    ax.set_ylabel("P(size >= s)")
    ax.legend()
    ax.set_title(f"patch sizes, daily snapshots after day {args.burn_in}\n{title}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "patches.png", dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
