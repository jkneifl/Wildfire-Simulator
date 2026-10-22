"""Command line interface: ``python -m wildfire.cli {run,generate,replay}``."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import Config


def cmd_run(args) -> None:
    from .dynamics import step
    from .metrics import global_stats
    from .terrain import initial_state

    cfg = Config.from_yaml(args.config)
    s = initial_state(cfg, args.seed)
    t0 = time.perf_counter()
    for i in range(1, args.steps + 1):
        s = step(s, 0, cfg)
        if i % args.every == 0 or i == args.steps:
            st = global_stats(s)
            print(f"t={int(s.t):7d}  " + "  ".join(f"{k}={v:.4g}" for k, v in st.items()))
    dt = time.perf_counter() - t0
    size = f"{cfg.height}x{cfg.width}"
    print(f"{args.steps} steps of {size} in {dt:.2f}s: {args.steps / dt:.1f} steps/s")


def generate_episode(job: dict) -> str:
    """Generate one episode from a job dict (see :func:`cmd_generate`)."""
    from .env import WildfireEnv
    from .policies import make_policy
    from .record import Products, Recorder

    cfg = Config.from_dict(job["config"])
    env = WildfireEnv(cfg, rgb=job["rgb"])
    obs, info = env.reset(seed=job["seed"])
    policy = make_policy(job["policy"], cfg, seed=job["seed"], params=job["params"])
    rec = Recorder(
        job["out"], cfg, env.state, job["policy"], True, job["rgb"], Products(**job["products"])
    )
    for _ in range(job["steps"]):
        a = policy.act(obs, info)
        obs, _, _, _, info = env.step(a)
        rec.add(a, env.state, obs)
    rec.close(policy_seed=job["seed"], policy_params=policy.param_values(), policy_log=policy.log)
    return str(job["out"])


def parse_mix(text: str) -> list[tuple[str, float]]:
    """``"a:0.6,b:0.4"`` (or a single name) -> ``[(name, weight), ...]``."""
    out = []
    for part in text.split(","):
        name, _, w = part.partition(":")
        out.append((name.strip(), float(w) if w else 1.0))
    return out


def pick_policy(mix: list[tuple[str, float]], seed: int) -> str:
    """Policy of an episode, from the counter-based RNG (stream 1 keeps it apart from the
    policy's own draws)."""
    from .rng import Purpose, uniform

    w = [x for _, x in mix]
    u = float(uniform(seed, 1, 0, Purpose.POLICY, 0, 0)) * sum(w)
    return mix[min(int(np.searchsorted(np.cumsum(w), u, side="right")), len(mix) - 1)][0]


def products_from_args(args) -> dict:
    if args.products == "all":
        return {"near_radius": 32, "world_every": 1, "globals": True, "events": True}
    return {
        "near_radius": args.near_radius,
        "world_every": args.world_every,
        "globals": args.globals,
        "events": args.events,
    }


def dataset_info(args, extra: dict) -> dict:
    import subprocess

    from .record import git_hash

    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {
        "git": git_hash(),
        "git_dirty": dirty,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        **extra,
    }


def cmd_generate(args) -> None:
    import json
    from collections import Counter
    from multiprocessing import Pool
    from pathlib import Path

    cfg, overrides = Config.from_yaml(args.config).with_overrides(args.set)
    mix = parse_mix(args.policy_mix or args.policy)
    params = json.loads(args.policy_params) if args.policy_params else {}
    products = products_from_args(args)
    out = Path(args.out)
    jobs = []
    for i in range(args.episodes):
        seed = args.seed + i
        name = pick_policy(mix, seed)
        jobs.append(
            {
                "config": cfg.to_dict(), "policy": name, "params": params.get(name, {}),
                "seed": seed, "steps": args.steps, "out": out / f"ep_{i:04d}",
                "rgb": args.rgb, "products": products,
            }
        )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    info = dataset_info(
        args,
        {
            "overrides": overrides,
            "seeds": [args.seed, args.seed + args.episodes - 1],
            "policies": dict(Counter(j["policy"] for j in jobs)),
            "policy_params": params,
            "products": products,
        },
    )
    (out / "dataset.json").write_text(json.dumps(info, indent=2, default=str))
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for d in pool.imap_unordered(generate_episode, jobs):
            print("wrote", d)
    print(f"{args.episodes} episodes x {args.steps} steps in {time.perf_counter() - t0:.1f}s")


def cmd_rerecord(args) -> None:
    from pathlib import Path

    from .record import Products, rerecord

    out = Path(args.out)
    for d in args.episodes:
        target = out / Path(d).name
        rerecord(d, target, Products(**products_from_args(args)))
        print("wrote", target)


def cmd_replay(args) -> None:
    from .record import replay

    for d in args.episodes:
        r = replay(d)
        print(f"{d}: OK, {r['steps']} steps, {r['snapshots_checked']} snapshots match")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="wildfire")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run N headless steps and print statistics")
    r.add_argument("--config", default="configs/default.yaml")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--every", type=int, default=100)
    r.set_defaults(func=cmd_run)

    def add_products(q):
        q.add_argument("--products", choices=["all"], help="all: near r=32, world every step, …")
        q.add_argument("--near-radius", type=int, default=None, help="near crop radius (cells)")
        q.add_argument("--world-every", type=int, default=None, help="full-world grid cadence")
        q.add_argument("--globals", action="store_true", help="write globals.npy")
        q.add_argument("--events", action="store_true", help="write events.npz")

    g = sub.add_parser("generate", help="generate episodes with a scripted policy")
    g.add_argument("--config", default="configs/default.yaml")
    g.add_argument("--set", action="append", default=[], help="config override key=value")
    g.add_argument("--policy", default="explore_return")
    g.add_argument(
        "--policy-mix", default=None, help="per-episode mix, e.g. player:0.8,explore_return:0.2"
    )
    g.add_argument("--policy-params", default=None, help='JSON {"player": {"w_idle": 0.3}}')
    g.add_argument("--episodes", type=int, default=10)
    g.add_argument("--steps", type=int, default=2000)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0, help="seed of episode 0 (episode i: seed + i)")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--rgb", action="store_true", help="also store obs_rgb.npy")
    add_products(g)
    g.set_defaults(func=cmd_generate)

    rp = sub.add_parser("replay", help="rebuild episodes and check every snapshot")
    rp.add_argument("episodes", nargs="+")
    rp.set_defaults(func=cmd_replay)

    rr = sub.add_parser("rerecord", help="replay episodes and write them again with products")
    rr.add_argument("episodes", nargs="+")
    rr.add_argument("--out", required=True)
    add_products(rr)
    rr.set_defaults(func=cmd_rerecord)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
