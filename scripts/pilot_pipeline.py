"""End-to-end pilot of the data pipeline at 512^2 (writes docs/pilot/pipeline.*).

    uv run python scripts/pilot_pipeline.py --out data/pilot --report docs/pilot

1. generate 4 episodes x 500 steps after a 1200-step burn-in (player / explore_return_act,
   random actions with probability --epsilon, bumps with --bump, all products, seeds 9000+);
2. replay one episode with all products (array equality);
3. draw samples from every sampler and time them;
4. retention of tile statistics vs dt from the world grids;
5. one fork set (32 futures, dt up to 256) vs frozen / resample / split-half baselines.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from wildfire import forksets
from wildfire.cli import main as cli_main
from wildfire.datasets import (
    DynamicsSampler,
    Episodes,
    ReentrySampler,
    RefinementSampler,
    TripleSampler,
)
from wildfire.metrics import TILE_STATS, energy_distance, obs_stats, obs_tiles, retention_curve
from wildfire.record import EpisodeDataset, git_hash, replay


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/pilot")
    p.add_argument("--report", default="docs/pilot")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--epsilon", type=float, default=0.05, help="random-action probability")
    p.add_argument("--bump", type=float, default=0.1, help="walk-into-obstacle probability")
    args = p.parse_args(argv)
    out, report_dir = Path(args.out), Path(args.report)
    if out.exists():
        shutil.rmtree(out)
    report_dir.mkdir(parents=True, exist_ok=True)
    rep = {"git": git_hash(), "config": "configs/medium.yaml", "out": str(out)}
    rep["epsilon"], rep["bump"] = args.epsilon, args.bump

    t0 = time.perf_counter()
    cli_main(["generate", "--config", "configs/medium.yaml", "--policy-mix",
              "player:0.5,explore_return_act:0.5", "--products", "all",
              "--episodes", str(args.episodes), "--steps", str(args.steps),
              "--seed", "9000", "--burn-in", "1200", "--epsilon", str(args.epsilon),
              "--bump", str(args.bump), "--out", str(out)])  # fmt: skip
    rep["generate_s"] = time.perf_counter() - t0
    rep["bytes_per_episode"] = int(
        np.mean(
            [sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in out.glob("ep_*")]
        )
    )
    t0 = time.perf_counter()
    rep["replay"] = replay(out / "ep_0000")
    rep["replay_s"] = time.perf_counter() - t0

    eps = Episodes(out)
    cfg = eps.eps[0]["cfg"]
    timings = {}
    for name, sampler in {
        "dynamics(n=32,L=3)": DynamicsSampler(eps, n=32, levels=3),
        "reentry_synthetic": ReentrySampler(eps, dt_range=(1, 256)),
        "reentry_revisits": ReentrySampler(eps, source="revisits"),
        "refinement": RefinementSampler(eps, dt_range=(1, 256)),
        "triples": TripleSampler(eps, dt_range=(2, 256)),
    }.items():
        t0 = time.perf_counter()
        for i in range(100):
            sampler.sample(i)
        timings[name] = (time.perf_counter() - t0) / 100 * 1000
    rep["sample_ms"] = timings
    rep["revisits_available"] = int(
        sum(len(r) for r in ReentrySampler(eps, source="revisits").revisits)
    )

    # retention of tile statistics (world grids every 10 steps)
    ds = EpisodeDataset(out)
    series = []
    for i in range(len(ds)):
        steps = list(range(0, args.steps + 1, 10))
        series.append(
            np.stack(
                [obs_stats(obs_tiles(ds.world(i, t), cfg.tile), cfg).reshape(-1, 5) for t in steps]
            )
        )
    lags = np.arange(1, series[0].shape[0])
    ret = {
        k: retention_curve(series, lags, k).round(4).tolist()
        for k in ("veg_density", "burned_frac")
    }
    rep["retention"] = {"lag_steps": (lags * 10).tolist(), **ret}

    # one fork set
    dts = (8, 16, 32, 64, 128, 256)
    (tau, tiles), *_ = forksets.pick_taus(out / "ep_0000", 1, max(dts))
    tiles = forksets.add_random_tiles(tiles, 4, (cfg.height, cfg.width), cfg.tile, 0, tau)
    t0 = time.perf_counter()
    fs = forksets.build(out / "ep_0000", tau, tiles, n=32, dts=dts, out=out / "forks" / "ep_0000")
    rep["fork_s"] = time.perf_counter() - t0
    ref = forksets.reference_stats(fs, cfg)  # (32, n_dts, n_tiles, K)
    floor = forksets.split_half_floor(ref)
    frozen = forksets.frozen_distance(fs, cfg)
    glob = ds[0]["globals"]
    grids = [ds.world(i, t) for i in range(len(ds)) for t in range(0, args.steps + 1, 5)]
    phases = [ds[i]["globals"][t, 1] for i in range(len(ds)) for t in range(0, args.steps + 1, 5)]
    resample = np.zeros_like(floor)
    for j, dt in enumerate(dts):
        base = forksets.resample_baseline(
            grids, phases, float(glob[tau + dt, 1]), cfg.tile, 32, cfg, tol=0.03
        )
        bs = obs_stats(base, cfg)
        for m in range(len(tiles)):
            resample[j, m] = energy_distance(ref[:, j, m], bs)
    rep["forkset"] = {
        "tau": tau, "tiles": [list(t) for t in tiles], "dts": list(dts), "n": 32,
        "split_half_floor": floor.round(4).tolist(), "frozen": frozen.round(4).tolist(),
        "resample": resample.round(4).tolist(), "stats": list(TILE_STATS),
    }  # fmt: skip
    (report_dir / "pipeline.json").write_text(json.dumps(rep, indent=2, default=float))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, v in ret.items():
        ax[0].plot(lags * 10, v, marker=".", label=k)
    ax[0].set_xlabel("dt (steps)")
    ax[0].set_ylabel("corr(tile stat at tau, at tau+dt)")
    ax[0].set_title(
        f"retention, {series[0].shape[1]} tiles x {len(series)} pilot episodes", fontsize=9
    )
    ax[0].legend()
    for name, v in (
        ("frozen", frozen),
        ("resample (phase-matched)", resample),
        ("split-half floor", floor),
    ):
        ax[1].plot(dts, v.mean(1), marker="o", label=name)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("dt (steps)")
    ax[1].set_ylabel("energy distance to reference (mean over tiles)")
    ax[1].set_title(f"fork set: ep_0000, tau={tau}, 32 futures, {len(tiles)} tiles", fontsize=9)
    ax[1].legend()
    seeds = f"seeds 9000-{9000 + args.episodes - 1}, epsilon {args.epsilon}, bump {args.bump}"
    title = f"pilot, configs/medium.yaml (512^2), 1200-step burn-in, {seeds}, git {rep['git'][:7]}"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(report_dir / "pipeline.png", dpi=120)
    print(
        json.dumps(
            {k: v for k, v in rep.items() if k not in ("retention", "forkset")},
            indent=2,
            default=float,
        )
    )
    print("wrote", report_dir / "pipeline.json")


if __name__ == "__main__":
    main()
