"""Pilot measurements at 512^2 (results: docs/pilot/pilot_512.json).

uv run python scripts/pilot_measure.py docs/pilot/pilot_512.json
"""

import io
import json
import platform
import subprocess
import sys
import time

import numpy as np

from wildfire import Config
from wildfire.dynamics import step
from wildfire.env import fork_streams
from wildfire.observe import grid
from wildfire.state import stack
from wildfire.terrain import initial_state


def npz_bytes(**arrays) -> int:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getbuffer().nbytes


cfg = Config.from_yaml("configs/medium.yaml")
H, W = cfg.height, cfg.width
out = {
    "config": "configs/medium.yaml",
    "machine": platform.platform() + " " + platform.processor(),
    "git": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip(),
    "seed": 0,
}

# burn in 1500 no-op steps (~6 days) so the world is in its regime, timing the steps
s = initial_state(cfg, 0)
t0 = time.perf_counter()
for _ in range(1500):
    s = step(s, 0, cfg)
out["step_per_s_512"] = 1500 / (time.perf_counter() - t0)

# world snapshot sizes at 8 times over one day
sizes = []
for _ in range(8):
    for _ in range(30):
        s = step(s, 0, cfg)
    sizes.append(npz_bytes(grid=grid(s, 0, 0, H, W, cfg)))
out["world_snapshot_bytes"] = {"min": min(sizes), "mean": float(np.mean(sizes)), "max": max(sizes)}
out["world_snapshot_raw_bytes"] = H * W * 6

# timing of one full-world observable grid
t0 = time.perf_counter()
for _ in range(20):
    grid(s, 0, 0, H, W, cfg)
out["world_grid_ms"] = (time.perf_counter() - t0) / 20 * 1000

# near crops: 256 consecutive steps around a moving point (agent stays; crop around its cell)
for r in (32, 64):
    s2 = s
    crops = []
    for _ in range(256):
        s2 = step(s2, 1 + (len(crops) // 20) % 4, cfg)  # walk in a square
        crops.append(grid(s2, int(s2.ay) - r, int(s2.ax) - r, 2 * r + 1, 2 * r + 1, cfg))
    out[f"near_r{r}_chunk256_bytes"] = npz_bytes(near=np.stack(crops))

# fork cost: 32 futures x 256 steps, batched
b = stack([s] * 32)
b.stream = fork_streams(s, 32)
t0 = time.perf_counter()
for _ in range(256):
    b = step(b, 0, cfg)
dt = time.perf_counter() - t0
out["fork32_world_steps_per_s"] = 32 * 256 / dt
out["fork32x256_s"] = dt
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
