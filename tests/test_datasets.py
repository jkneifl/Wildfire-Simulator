import numpy as np
import pytest

from wildfire.cli import main as cli_main
from wildfire.datasets import (
    RING_SENTINEL,
    DynamicsSampler,
    Episodes,
    ReentrySampler,
    RefinementSampler,
    TripleSampler,
    _crop,
)
from wildfire.levels import one_hot, restrict
from wildfire.tiles import visibility

PROBE_KEYS = {"names", "globals_tau", "globals_t", "globals_mean", "ignitions", "globals_t0"}


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds")
    cli_main(
        ["generate", "--config", "configs/small.yaml", "--set", "p_lightning=5e-4",
         "--policy-mix", "explore_return_act:0.5,player:0.5", "--products", "all",
         "--episodes", "3", "--steps", "500", "--workers", "3", "--out", str(out)]
    )  # fmt: skip
    return Episodes(out)


def _no_probe_in_input(sample):
    assert not (set(sample["input"]) & PROBE_KEYS)


def test_reentry_synthetic(episodes):
    s = ReentrySampler(episodes, tile=16, ring=8, seed=1)
    for i in range(20):
        x = s.sample(i)
        e = x["meta"]["e"]
        ty, tx = x["meta"]["tile"]
        tau, t = x["meta"]["tau"], x["meta"]["t"]
        assert x["input"]["dt"] == t - tau >= 1
        w_tau, w_t = episodes.world(e, tau), episodes.world(e, t)
        assert np.array_equal(x["input"]["tile_tau"], _crop(w_tau, ty * 16, tx * 16, 16, 16))
        assert np.array_equal(x["target"]["tile_t"], _crop(w_t, ty * 16, tx * 16, 16, 16))
        ring = x["input"]["ring_t"]
        assert (ring[8:24, 8:24] == RING_SENTINEL).all()  # the target never leaks into the ring
        ref = _crop(w_t, ty * 16 - 8, tx * 16 - 8, 32, 32)
        mask = np.ones((32, 32), bool)
        mask[8:24, 8:24] = False
        assert np.array_equal(ring[mask], ref[mask])
        _no_probe_in_input(x)
        assert "ignitions" in x["probe"] and "globals_mean" in x["probe"]
    a, b = s.sample(7), ReentrySampler(episodes, seed=1).sample(7)
    assert a["meta"] == b["meta"] and np.array_equal(a["target"]["tile_t"], b["target"]["tile_t"])


def test_zero_gap_is_identity(episodes):
    s = ReentrySampler(episodes, dt_range=(0, 0), seed=2)
    for i in range(5):
        x = s.sample(i)
        assert x["input"]["dt"] == 0
        assert np.array_equal(x["input"]["tile_tau"], x["target"]["tile_t"])


def test_reentry_from_real_absences(episodes):
    s = ReentrySampler(episodes, source="revisits", seed=3)
    for i in range(10):
        x = s.sample(i)
        e = x["meta"]["e"]
        ep = episodes.eps[e]
        vis = visibility(ep["agent"], ep["cfg"].window // 2, 16, (64, 64))
        ty, tx = x["meta"]["tile"]
        tau, t = x["meta"]["tau"], x["meta"]["t"]
        assert vis[tau, ty, tx] and vis[t, ty, tx] and not vis[tau + 1 : t, ty, tx].any()


def test_dynamics_sampler(episodes):
    s = DynamicsSampler(episodes, n=8, levels=2, dt=2, seed=4)
    x = s.sample(0)
    assert [z.shape for z in x["input"]["levels"]] == [(8, 8, 20)] * 3
    assert [z.shape for z in x["target"]["levels"]] == [(8, 8, 20)] * 3
    assert len(x["input"]["actions"]) == 2 and x["input"]["seen"].shape == (8, 8)
    t = x["meta"]["t"]
    e = x["meta"]["e"]
    oy, ox = x["input"]["origins"][1]
    z = one_hot(_crop(episodes.world(e, t + 2), oy, ox, 16, 16))
    assert np.allclose(x["target"]["levels"][1], restrict(z, 2))
    _no_probe_in_input(x)


def test_refinement_and_triples(episodes):
    r = RefinementSampler(episodes, factor=2, seed=5).sample(0)
    assert np.allclose(restrict(r["input"]["details_tau"]), 0)
    assert np.allclose(restrict(r["target"]["details_t"]), 0)
    assert r["input"]["coarse_t"].shape == (8, 8, 20)
    tr = TripleSampler(episodes, seed=6).sample(3)
    assert tr["input"]["dt"] >= 1 and len(tr["input"]["clocks"]) == 3
    assert tr["meta"]["tau"] + 2 * tr["input"]["dt"] <= 500
