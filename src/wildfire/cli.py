"""Command line interface: ``python -m wildfire.cli {run,generate,replay}``."""

from __future__ import annotations

import argparse
import time

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


def generate_episode(job: tuple) -> str:
    """Generate one episode; ``job = (cfg_dict, policy, seed, steps, out_dir, rgb)``."""
    from .env import WildfireEnv
    from .policies import make_policy
    from .record import Recorder

    cfg_dict, policy_name, seed, steps, out_dir, rgb = job
    cfg = Config.from_dict(cfg_dict)
    env = WildfireEnv(cfg, rgb=rgb)
    obs, info = env.reset(seed=seed)
    policy = make_policy(policy_name, cfg, seed=seed)
    rec = Recorder(out_dir, cfg, env.state, policy=policy_name, from_reset=True, rgb=rgb)
    for _ in range(steps):
        a = policy.act(obs, info)
        obs, _, _, _, info = env.step(a)
        rec.add(a, env.state, obs)
    rec.close(policy_seed=seed, policy_log=policy.log)
    return str(out_dir)


def cmd_generate(args) -> None:
    from multiprocessing import Pool
    from pathlib import Path

    cfg = Config.from_yaml(args.config)
    out = Path(args.out)
    jobs = [
        (cfg.to_dict(), args.policy, args.seed + i, args.steps, out / f"ep_{i:04d}", args.rgb)
        for i in range(args.episodes)
    ]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for d in pool.imap_unordered(generate_episode, jobs):
            print("wrote", d)
    print(f"{args.episodes} episodes x {args.steps} steps in {time.perf_counter() - t0:.1f}s")


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

    g = sub.add_parser("generate", help="generate episodes with a scripted policy")
    g.add_argument("--config", default="configs/default.yaml")
    g.add_argument("--policy", default="explore_return")
    g.add_argument("--episodes", type=int, default=10)
    g.add_argument("--steps", type=int, default=2000)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0, help="seed of episode 0 (episode i: seed + i)")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--rgb", action="store_true", help="also store obs_rgb.npy")
    g.set_defaults(func=cmd_generate)

    rp = sub.add_parser("replay", help="rebuild episodes and check every snapshot")
    rp.add_argument("episodes", nargs="+")
    rp.set_defaults(func=cmd_replay)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
