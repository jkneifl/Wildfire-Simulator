import numpy as np
import pytest

from wildfire import WildfireEnv, forksets
from wildfire.cli import main as cli_main
from wildfire.observe import grid
from wildfire.record import EpisodeDataset, load_config


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    out = tmp_path_factory.mktemp("fk")
    cli_main(
        ["generate", "--config", "configs/small.yaml", "--set", "p_lightning=1e-3",
         "--policy", "explore_return_act", "--products", "all",
         "--episodes", "1", "--steps", "300", "--workers", "1", "--out", str(out)]
    )  # fmt: skip
    return out / "ep_0000"


def test_state_at_matches_recording(episode):
    cfg, _ = load_config(episode)
    ds = EpisodeDataset(episode)
    for tau in (0, 37, 50, 123):
        s = forksets.state_at(episode, tau)
        assert np.array_equal(grid(s, 0, 0, 64, 64, cfg), ds.world(0, tau))


def test_build_reproducible_and_matches_single_futures(episode, tmp_path):
    cfg, _ = load_config(episode)
    tiles = [(1, 1), (2, 3)]
    fs = forksets.build(episode, 40, tiles, n=4, dts=(4, 16), world_dts=(16,), out=tmp_path / "f")
    assert fs["tile_obs"].shape == (4, 2, 2, 16, 16, 6)
    assert fs["world_dt16"].shape == (4, 64, 64, 6)
    assert (tmp_path / "f.npz").exists() and (tmp_path / "f.json").exists()
    again = forksets.build(episode, 40, tiles, n=4, dts=(4, 16))
    assert np.array_equal(fs["tile_obs"], again["tile_obs"])
    # future 2 equals stepping one env with that stream and the recorded actions
    s = forksets.state_at(episode, 40)
    s.stream = fs["streams"][2].copy()
    env = WildfireEnv(cfg, rgb=False)
    env.set_state(s)
    actions = np.load(episode / "actions.npy")
    for k in range(16):
        env.step(int(actions[40 + k]))
        if k + 1 == 4:
            assert np.array_equal(grid(env.state, 16, 16, 16, 16, cfg), fs["tile_obs"][2, 0, 0])
    assert np.array_equal(grid(env.state, 32, 48, 16, 16, cfg), fs["tile_obs"][2, 1, 1])
    assert np.array_equal(grid(env.state, 0, 0, 64, 64, cfg), fs["world_dt16"][2])
    assert not np.array_equal(fs["world_dt16"][0], fs["world_dt16"][1])  # streams differ


def test_baselines_and_floors(episode):
    cfg, _ = load_config(episode)
    fs = forksets.build(episode, 20, [(1, 1)], n=8, dts=(8, 64))
    ref = forksets.reference_stats(fs, cfg)
    assert ref.shape == (8, 2, 1, 5)
    floor = forksets.split_half_floor(ref)
    frozen = forksets.frozen_distance(fs, cfg)
    assert floor.shape == frozen.shape == (2, 1) and (frozen >= 0).all()
    ds = EpisodeDataset(episode)
    grids = [ds.world(0, t) for t in range(0, 300, 10)]
    phases = ds[0]["globals"][::10, 1][: len(grids)]
    tiles = forksets.resample_baseline(grids, phases, float(phases[3]), 16, 5, cfg, tol=0.02)
    assert tiles.shape == (5, 16, 16, 6)


def test_pick_taus_and_cli(episode, tmp_path):
    picks = forksets.pick_taus(episode, 2, dt_max=32)
    assert picks and all(t + 32 <= 300 for t, _ in picks) and all(len(tl) for _, tl in picks)
    with pytest.raises(ValueError, match="exceeds"):
        forksets.build(episode, 290, [(0, 0)], n=2, dts=(32,))
    cli_main(["forks", str(episode), "--out", str(tmp_path / "out"), "--n", "2",
              "--dts", "4,8", "--n-taus", "1", "--random-tiles", "1"])  # fmt: skip
    assert len(list((tmp_path / "out").glob("*.npz"))) == 1
