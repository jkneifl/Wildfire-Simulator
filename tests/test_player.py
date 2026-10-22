import numpy as np
import pytest

from wildfire import WildfireEnv
from wildfire import actions as A
from wildfire.cli import main as cli_main
from wildfire.cli import parse_mix, pick_policy
from wildfire.metrics import _runs, behaviour_distance, behaviour_stats
from wildfire.policies import Player, make_policy
from wildfire.record import EpisodeDataset, replay
from wildfire.tiles import absences, visibility


def _rollout(cfg, steps, seed=3, params=None):
    env = WildfireEnv(cfg, rgb=False)
    obs, info = env.reset(seed=seed)
    pol = make_policy("player", cfg, seed=seed, params=params)
    acts, intents = [], set()
    for _ in range(steps):
        a = pol.act(obs, info)
        intents.add(pol.intent)
        obs, *_, info = env.step(a)
        acts.append(a)
    return np.array(acts), intents


def test_player_deterministic_and_all_intents(small):
    a, intents = _rollout(small, 1500)
    b, _ = _rollout(small, 1500)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _rollout(small, 1500, seed=4)[0])
    assert intents >= {"explore", "revisit", "ignite", "fight", "idle"}, intents
    assert (a == A.IGNITE).any() and (a == A.SPRAY).any() and (a == A.NOOP).any()


def test_policy_params_override_and_validation(small):
    p = make_policy("player", small, params={"w_idle": 0.5})
    assert p.w_idle == 0.5 and Player.w_idle == 0.2  # instance override, class default kept
    with pytest.raises(ValueError, match="no parameter"):
        make_policy("player", small, params={"bogus": 1})


def test_runs_and_absences_on_toy_data(small):
    # actions: E E E noop noop N N W
    a = np.array([2, 2, 2, 0, 0, 1, 1, 4])
    assert _runs((a >= 1) & (a <= 4), a).tolist() == [3, 2, 1]
    assert _runs(a == 0).tolist() == [2]
    # visibility on a 64x64 world, tile 16, radius 2: walk from x=8 to x=40 and back
    xs = list(range(8, 41)) + list(range(39, 7, -1))
    agent = np.array([[x, 8, 0, 0] for x in xs])
    vis = visibility(agent, 2, 16, (64, 64))
    assert vis.shape == (len(xs), 4, 4)
    gaps = absences(vis)
    # tile (0, 0) visible while x <= 17; left at x=17 (step 9), back at x=17 (step 55)
    row = gaps[(gaps[:, 0] == 0) & (gaps[:, 1] == 0)][0]
    assert row.tolist() == [0, 0, 9, 55]


def test_behaviour_stats_and_distance(small):
    acts, _ = _rollout(small, 600)
    env = WildfireEnv(small, rgb=False)
    obs, info = env.reset(seed=3)
    pol = make_policy("player", small, seed=3)
    agent, grids = [[int(env.state.ax), int(env.state.ay)]], [obs["grid"]]
    for _ in range(600):
        a = pol.act(obs, info)
        obs, *_, info = env.step(a)
        agent.append([int(env.state.ax), int(env.state.ay)])
        grids.append(obs["grid"])
    st = behaviour_stats(acts, np.array(agent), np.stack(grids), small)
    total = st["frac_move"] + st["frac_idle"] + st["frac_ignite"] + st["frac_water"]
    assert abs(total - 1) < 1e-9 and st["coverage"][0] > 0 and st["straight_runs"].sum() > 0
    d = behaviour_distance([st, st], [st, st])
    assert all(abs(v) < 1e-12 for v in d.values() if v == v)  # nan allowed for empty samples


def test_policy_mix_is_deterministic():
    mix = parse_mix("player:0.75,explore_return:0.25")
    picks = [pick_policy(mix, s) for s in range(400)]
    assert picks == [pick_policy(mix, s) for s in range(400)]
    share = picks.count("player") / len(picks)
    assert 0.65 < share < 0.85
    assert parse_mix("random_walk") == [("random_walk", 1.0)]


def test_generate_with_mix_params_and_products(tmp_path):
    out = tmp_path / "d"
    cli_main(
        ["generate", "--config", "configs/small.yaml", "--set", "p_lightning=1e-4",
         "--policy-mix", "player:0.5,explore_return_act:0.5",
         "--policy-params", '{"player": {"w_idle": 0.4}}', "--products", "all",
         "--episodes", "2", "--steps", "80", "--workers", "2", "--out", str(out)]
    )  # fmt: skip
    ds = EpisodeDataset(out)
    assert len(ds) == 2 and (out / "dataset.json").exists()
    for i in range(2):
        meta = ds[i]["meta"]
        assert meta["config"]["p_lightning"] == 1e-4
        if meta["policy"] == "player":
            assert meta["policy_params"]["w_idle"] == 0.4
        assert replay(ds.dirs[i])["steps"] == 80
