"""The browser port (web/wildfire.js) must reproduce the Python simulator bit for bit."""

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from wildfire import Config
from wildfire.dynamics import step
from wildfire.env import fork_streams
from wildfire.rng import Purpose, uniform
from wildfire.terrain import initial_state

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def digest(s) -> str:
    h = hashlib.sha256()
    for a in (s.terrain, s.veg, s.fire, s.ash_age.astype("<u2"), s.wet):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"{int(s.t)},{int(s.ax)},{int(s.ay)},{int(s.facing)},{int(s.tank)}".encode())
    return h.hexdigest()


def run_node(job: dict) -> dict:
    out = subprocess.run(
        [NODE, "web/parity.js"], input=json.dumps(job), capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize(
    "overrides, seed, steps",
    [
        ({"height": 64, "width": 64, "p_lightning": 1e-3}, 3, 600),
        ({"height": 40, "width": 70, "day_length": 50, "tank": 3}, 11, 400),
        ({}, 0, 300),
    ],
)
def test_js_matches_python(overrides, seed, steps):
    cfg = Config().replace(**overrides)
    u = uniform(seed, 0, 0, Purpose.POLICY, np.zeros(steps, int), np.arange(steps))
    # mostly moves, with ignite and water
    actions = np.where(u < 0.1, 5, np.where(u < 0.2, 6, 1 + (u * 40).astype(int) % 4)).tolist()
    checkpoints = [0, 1, steps // 3, steps // 2, steps]
    s = initial_state(cfg, seed)
    expect = {"0": digest(s)}
    for i, a in enumerate(actions, 1):
        s = step(s, a, cfg)
        if i in checkpoints:
            expect[str(i)] = digest(s)
    job = {
        "config": cfg.to_dict(),
        "seed": seed,
        "stream": 0,
        "actions": actions,
        "checkpoints": checkpoints,
        "forks": 3,
    }
    got = run_node(job)
    assert got.pop("forks") == [str(v) for v in fork_streams(s, 3)]
    assert got == expect
