"""Tune the ``player`` policy so its behaviour statistics match recorded human sessions.

    uv run python scripts/tune_player.py --human data/human --candidates 48 --out docs/player

Random search over ``Player`` parameters. Each candidate is rolled out on ``--rollouts``
fresh worlds (seeds 5000+, same config and length as the human sessions) and scored by the
per-statistic energy distance to the human sessions (``metrics.behaviour_distance``), each
divided by its median over all candidates and summed. The human-vs-human split-half
distance is reported as the noise floor of every statistic.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wildfire import Config, WildfireEnv
from wildfire.metrics import behaviour_distance, behaviour_stats
from wildfire.policies import Player, make_policy
from wildfire.record import EpisodeDataset
from wildfire.rng import Purpose, uniform

# (low, high, log-scale) per tuned parameter
SPACE = {
    "w_explore": (0.2, 3.0, True),
    "w_revisit": (0.05, 2.0, True),
    "w_ignite": (0.02, 1.0, True),
    "w_fight": (0.2, 5.0, True),
    "w_idle": (0.02, 1.0, True),
    "dwell_mean": (15.0, 200.0, True),
    "idle_mean": (2.0, 40.0, True),
    "revisit_median": (100.0, 1500.0, True),
    "p_pause": (0.0, 0.3, False),
    "turn": (0.02, 0.4, True),
}


def sample_params(i: int, seed: int) -> dict:
    u = uniform(seed, 2, i, Purpose.POLICY, np.zeros(len(SPACE), int), np.arange(len(SPACE)))
    out = {}
    for (k, (lo, hi, log)), v in zip(SPACE.items(), u):
        out[k] = float(np.exp(np.log(lo) + v * np.log(hi / lo))) if log else lo + v * (hi - lo)
    return out


def rollout(job) -> dict:
    cfg_dict, params, seed, steps = job
    cfg = Config.from_dict(cfg_dict)
    env = WildfireEnv(cfg, rgb=False)
    obs, info = env.reset(seed=seed)
    pol = make_policy("player", cfg, seed=seed, params=params)
    acts, agent, grids = [], [[int(env.state.ax), int(env.state.ay)]], [obs["grid"]]
    for _ in range(steps):
        a = pol.act(obs, info)
        obs, _, _, _, info = env.step(a)
        acts.append(a)
        agent.append([int(env.state.ax), int(env.state.ay)])
        grids.append(obs["grid"])
    return behaviour_stats(np.array(acts), np.array(agent), np.stack(grids), cfg)


def human_stats(root: str) -> tuple[list[dict], Config, int]:
    ds = EpisodeDataset(root)
    if len(ds) < 2:
        raise SystemExit(f"need at least 2 human sessions in {root}, found {len(ds)}")
    stats, steps = [], []
    for i in range(len(ds)):
        ep = ds[i]
        cfg = Config.from_dict(ep["meta"]["config"])
        stats.append(behaviour_stats(ep["actions"], ep["agent"], ep["obs_grid"], cfg))
        steps.append(len(ep["actions"]))
    return stats, cfg, int(np.median(steps))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--human", required=True, help="directory of human episodes")
    p.add_argument("--candidates", type=int, default=48)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="seed of the parameter search")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="docs/player")
    args = p.parse_args(argv)

    human, cfg, steps = human_stats(args.human)
    half = len(human) // 2
    floor = behaviour_distance(human[:half], human[half:])
    cands = [{k: getattr(Player, k) for k in SPACE}] + [
        sample_params(i, args.seed) for i in range(args.candidates - 1)
    ]  # candidate 0: hand-set defaults
    jobs = [(cfg.to_dict(), c, 5000 + r, steps) for c in cands for r in range(args.rollouts)]
    with Pool(args.workers) as pool:
        results = pool.map(rollout, jobs)
    per = [results[i * args.rollouts : (i + 1) * args.rollouts] for i in range(len(cands))]
    dists = [behaviour_distance(r, human) for r in per]
    keys = list(floor)

    def robust_scale(values):
        v = np.array(values, float)
        return float(np.nanmedian(v)) or 1.0 if np.isfinite(v).any() else 1.0

    scale = {k: robust_scale([d[k] for d in dists]) for k in keys}
    scores = [float(np.nansum([d[k] / scale[k] for k in keys])) for d in dists]
    best = int(np.argmin(scores))
    report = {
        "human": args.human,
        "n_human": len(human),
        "steps": steps,
        "config": cfg.to_dict(),
        "candidates": len(cands),
        "rollouts_per_candidate": args.rollouts,
        "best_index": best,
        "best_params": cands[best],
        "best_score": scores[best],
        "default_score": scores[0],
        "per_stat": {
            k: {"best": dists[best][k], "default": dists[0][k], "human_split_half": floor[k]}
            for k in keys
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"best candidate {best}: score {scores[best]:.3f} (defaults {scores[0]:.3f})")
    for k in keys:
        s = report["per_stat"][k]
        flag = "  <- above noise floor" if s["best"] > 2 * max(s["human_split_half"], 1e-9) else ""
        print(f"  {k:20s} best {s['best']:.4f}  human split-half {s['human_split_half']:.4f}{flag}")
    print("wrote", out / "tuning.json")


if __name__ == "__main__":
    main()
