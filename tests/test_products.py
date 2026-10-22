import numpy as np
import pytest

from wildfire import WildfireEnv
from wildfire import actions as A
from wildfire.dynamics import N8, step
from wildfire.events import AGENT, LIGHTNING, SPREAD, ignitions
from wildfire.record import (
    GLOBAL_COLUMNS,
    EpisodeDataset,
    Products,
    Recorder,
    ReplayMismatch,
    near_crop,
    replay,
    rerecord,
)
from wildfire.rng import Purpose, uniform
from wildfire.terrain import initial_state

from .conftest import flat_state

ALL = Products(near_radius=20, world_every=3, globals=True, events=True)


def _actions(n, seed=0):
    u = uniform(seed, 0, 2, Purpose.POLICY, np.zeros(n, int), np.arange(n))
    return np.where(u < 0.15, A.IGNITE, np.where(u < 0.25, A.SPRAY, 1 + (u * 40).astype(int) % 4))


def test_events_cover_every_new_ignition(small):
    cfg = small.replace(p_lightning=2e-3)
    s = initial_state(cfg, 3)
    counts = {AGENT: 0, LIGHTNING: 0, SPREAD: 0}
    for a in _actions(300):
        new = step(s, int(a), cfg)
        ev = ignitions(s, int(a), new, cfg)
        cause = np.zeros(s.shape, np.uint8)
        cause[ev["y"], ev["x"]] = ev["cause"]
        # every new ignition has exactly one cause, and every caused cell burns now
        assert (cause[ev["dynamic_mask"]] > 0).all()
        assert (cause[(s.fire == 0) & (new.fire > 0)] > 0).all()
        assert (new.fire[cause > 0] > 0).all()
        # lightning is exactly the eligible strikes; spread cells had a burning neighbour
        assert not (ev["lightning_mask"] & ~ev["dynamic_mask"]).any()
        spread = ev["cause"] == SPREAD
        assert (ev["nbrs"][spread] > 0).all()
        for c in counts:
            counts[c] += int((ev["cause"] == c).sum())
        s = new
    assert all(v > 0 for v in counts.values()), counts


def test_agent_ignition_and_spray_attribution(small):
    cfg = small.replace(p_lightning=0.0, beta=1.0, m_night=1.0)
    s = flat_state(cfg, veg=4, facing=1)
    y, x = int(s.ay), int(s.ax) + 1
    new = step(s, A.IGNITE, cfg)
    ev = ignitions(s, A.IGNITE, new, cfg)
    causes = dict(zip(zip(ev["y"].tolist(), ev["x"].tolist()), ev["cause"].tolist()))
    assert causes[(y, x)] == AGENT
    assert all(c == SPREAD for k, c in causes.items() if k != (y, x))
    # neighbour bitmask of a spread cell east of the target points west (N8 index 3)
    if (y, x + 1) in causes:
        i = list(zip(ev["y"].tolist(), ev["x"].tolist())).index((y, x + 1))
        assert ev["nbrs"][i] & (1 << N8.index((0, -1, 1.0)))
    # a fire sprayed in this step does not count as a burning neighbour
    s2 = flat_state(cfg, veg=4, facing=1)
    s2.fire[y, x] = 9
    new2 = step(s2, A.SPRAY, cfg)
    assert ignitions(s2, A.SPRAY, new2, cfg)["y"].size == 0


def _record(cfg, path, steps, products=ALL, seed=4):
    env = WildfireEnv(cfg, rgb=False)
    obs, _ = env.reset(seed=seed)
    rec = Recorder(path, cfg, env.state, policy="test", from_reset=True, products=products)
    for a in _actions(steps):
        obs, *_ = env.step(int(a))
        rec.add(int(a), env.state, obs)
    return rec.close()


def test_products_written_and_replayed(small, tmp_path):
    cfg = small.replace(p_lightning=1e-3)
    d = _record(cfg, tmp_path / "ep", 600)
    ds = EpisodeDataset(d)
    near = ds.near(0)
    assert near.shape == (601, 41, 41, 6) and near.dtype == np.uint8
    assert ds.world_steps(0) == list(range(0, 601, 3))
    ep = ds[0]
    assert ep["globals"].shape == (601, len(GLOBAL_COLUMNS))
    ev = ds.events(0)
    assert ev["step"].min() >= 1 and ev["step"].max() <= 600 and len(ev["cause"]) > 0
    # the window observation is the centre of the near crop
    assert np.array_equal(near[:, 4:-4, 4:-4], ep["obs_grid"])
    assert replay(d)["steps"] == 600


def test_replay_detects_tampered_world_grid(small, tmp_path):
    d = _record(small, tmp_path / "ep", 30, Products(world_every=10))
    path = d / "world" / "step_000020.npz"
    g = np.load(path)["grid"]
    g[0, 0, 1] = (g[0, 0, 1] + 1) % 5
    np.savez_compressed(path, grid=g)
    with pytest.raises(ReplayMismatch, match="world grid differs at step 20"):
        replay(d)


def test_rerecord_adds_products_to_a_plain_episode(small, tmp_path):
    plain = _record(small, tmp_path / "plain", 120, Products())
    out = rerecord(plain, tmp_path / "full", ALL)
    assert replay(out)["steps"] == 120  # checks near, world, globals and events too
    ds = EpisodeDataset(out)
    assert np.array_equal(ds.near(0)[100], near_crop(ds.state(0, 100), 20, small))
    assert ds.world(0, 99).shape == (small.height, small.width, 6)
    assert ds[0]["meta"]["rerecorded_from"] == str(plain)
    assert np.array_equal(ds[0]["actions"], EpisodeDataset(plain)[0]["actions"])
