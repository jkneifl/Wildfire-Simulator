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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="wildfire")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run N headless steps and print statistics")
    r.add_argument("--config", default="configs/default.yaml")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--every", type=int, default=100)
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
