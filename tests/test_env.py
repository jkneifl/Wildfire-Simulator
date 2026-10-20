import json

import numpy as np
import pytest

from wildfire import Config, VecWildfire, WildfireEnv
from wildfire import actions as A
from wildfire.cli import main as cli_main
from wildfire.observe import CHANNELS, observe, privileged
from wildfire.record import EpisodeDataset, Recorder, ReplayMismatch, replay
from wildfire.rng import Purpose, uniform
from wildfire.state import WorldState, equal
from wildfire.tiles import TileTracker, extract_tile, tile_cells, tile_of, tiles_in_window

from .conftest import flat_state


def _actions(n, seed=0):
    u = uniform(seed, 0, 1, Purpose.POLICY, np.zeros(n, int), np.arange(n))
    return (u * A.N_ACTIONS).astype(int)


def test_observation_is_observable_only(small):
    env = WildfireEnv(small)
    obs, info = env.reset(seed=3)
    V = small.window
    assert set(obs) == {"grid", "rgb", "light", "agent"}
    assert obs["grid"].shape == (V, V, len(CHANNELS)) and obs["grid"].dtype == np.uint8
    assert obs["rgb"].shape == (V * small.pixels_per_cell, V * small.pixels_per_cell, 3)
    assert set(obs["agent"]) == {"facing", "tank"}
    # buckets only: values stay within 0..3 for fire, wet and ash channels
    assert obs["grid"][..., 3:].max() <= 3
    # info carries no world layers or rng ids
    assert set(info) == {"step", "pos", "tiles", "first_seen", "last_seen"}
    assert {"seed", "stream", "fire", "ash_age"} <= set(env.privileged())


def test_window_outside_world_is_rock(small):
    s = flat_state(small, agent=(0, 0))
    g = observe(s, small, rgb=False)["grid"]
    r = small.window // 2
    assert (g[:r, :, 0] == 2).all() and (g[:, :r, 0] == 2).all()
    assert (g[r:, r:, 0] == 0).all()


def test_privileged_is_a_copy(small):
    s = flat_state(small)
    p = privileged(s)
    p["veg"][:] = 0
    assert s.veg.max() == 4


def test_state_roundtrip_and_fork(small):
    cfg = small.replace(p_lightning=1e-3)
    env = WildfireEnv(cfg, rgb=False)
    env.reset(seed=1)
    for a in _actions(30):
        env.step(a)
    saved = env.get_state()
    a = env.fork(n=4, steps=60)
    b = env.fork(n=4, steps=60)
    assert equal(env.get_state(), saved)  # forking does not change the env
    assert equal(a, b)  # same streams, identical futures
    assert len(set(a.stream.tolist())) == 4
    assert not np.array_equal(a.veg[0], a.veg[1])  # different streams diverge
    # a future equals stepping the env itself with that stream
    s = saved.copy()
    s.stream = a.stream[2].copy()
    env.set_state(s)
    for _ in range(60):
        env.step(A.NOOP)
    assert equal(env.get_state(), a[2])


def test_vec_matches_single(small):
    vec = VecWildfire(small, 3)
    obs, infos = vec.reset(seeds=[0, 1, 2])
    envs = [WildfireEnv(small, rgb=False) for _ in range(3)]
    for i, e in enumerate(envs):
        e.reset(seed=i)
    acts = _actions(40 * 3).reshape(40, 3)
    for row in acts:
        obs, rew, term, trunc, infos = vec.step(row)
        for i, e in enumerate(envs):
            o, *_, info = e.step(row[i])
            assert np.array_equal(o["grid"], obs["grid"][i])
            assert np.array_equal(info["last_seen"], infos[i]["last_seen"])
    for i, e in enumerate(envs):
        assert equal(vec.state[i], e.state)
    assert rew.shape == (3,) and not term.any()


def test_reward_fn(small):
    def newly_burned(prev, cur):
        return -float(((cur.ash_age == 1) & (prev.ash_age == 0)).sum())

    env = WildfireEnv(small, reward_fn=newly_burned, rgb=False)
    env.reset(seed=0)
    s = env.get_state()
    s.fire[:] = np.where(s.veg > 0, 1, 0).astype(np.uint8)
    env.set_state(s)
    _, r, *_ = env.step(A.NOOP)
    assert r == -float((s.veg > 0).sum())


def test_tiles(small):
    T = 16
    assert tile_of(17, 40, T) == (1, 2)
    ys, xs = tile_cells((1, 2), T)
    assert (ys.start, ys.stop, xs.start, xs.stop) == (16, 32, 32, 48)
    assert tiles_in_window((10, 10, 10, 10), T, (64, 64)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert tiles_in_window((-20, -20, 10, 10), T, (64, 64)) == []
    assert tiles_in_window((-5, 60, 10, 10), T, (64, 64)) == [(0, 3)]
    s = flat_state(small)
    t = extract_tile(s, (3, 3), T)
    assert t["veg"].shape == (16, 16) and set(t) == {"terrain", "veg", "fire", "ash_age", "wet"}
    tr = TileTracker((64, 64), T)
    tr.mark((0, 0, 10, 10), 5)
    tr.mark((0, 0, 20, 10), 9)
    assert tr.first_seen[0, 0] == 5 and tr.last_seen[0, 0] == 9
    assert tr.first_seen[1, 0] == 9 and tr.last_seen[3, 3] == -1


def _record(cfg, path, steps, from_reset=True, warmup=0):
    env = WildfireEnv(cfg, rgb=False)
    obs, _ = env.reset(seed=7)
    for a in _actions(warmup, seed=5):
        obs, *_ = env.step(a)
    rec = Recorder(path, cfg, env.state, policy="test", from_reset=from_reset and warmup == 0)
    for a in _actions(steps):
        obs, *_ = env.step(a)
        rec.add(a, env.state, obs)
    return rec.close()


def test_replay_2000_steps(small, tmp_path):
    cfg = small.replace(p_lightning=1e-4)
    d = _record(cfg, tmp_path / "ep", 2000)
    r = replay(d)
    assert r["steps"] == 2000 and r["snapshots_checked"] == 2000 // cfg.snapshot_every + 1
    meta = json.loads((d / "meta.json").read_text())
    assert meta["seed"] == 7 and meta["from_reset"] and meta["policy"] == "test"
    ep = EpisodeDataset(tmp_path)[0]
    assert ep["actions"].shape == (2000,) and ep["actions"].dtype == np.int8
    assert ep["agent"].shape == (2001, 4)
    assert ep["obs_grid"].shape == (2001, cfg.window, cfg.window, len(CHANNELS))
    assert ep["tiles_seen"].shape == (2, 4, 4)


def test_replay_detects_tampering_and_mid_episode_start(small, tmp_path):
    d = _record(small, tmp_path / "mid", 120, warmup=33)
    assert replay(d)["steps"] == 120
    snap = d / "state" / "step_000050.npz"
    s = WorldState.load(snap)
    s.veg[0, 0] = (s.veg[0, 0] + 1) % 5
    s.save(snap)
    with pytest.raises(ReplayMismatch, match="step 50"):
        replay(d)


def test_generate_cli(tmp_path):
    out = tmp_path / "data"
    cli_main(["generate", "--config", "configs/small.yaml", "--policy", "random_walk",
              "--episodes", "2", "--steps", "60", "--out", str(out), "--workers", "2"])  # fmt: skip
    ds = EpisodeDataset(out)
    assert len(ds) == 2
    for i in range(2):
        assert replay(ds.dirs[i])["steps"] == 60
    assert Config.from_dict(ds[0]["meta"]["config"]) == Config.from_yaml("configs/small.yaml")
