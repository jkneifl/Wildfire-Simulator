import numpy as np
import pytest

from wildfire import Config, WildfireEnv
from wildfire import actions as A
from wildfire.policies import POLICIES, make_policy

from .conftest import flat_state


def _rollout(cfg, name, seed, steps, env_seed=0, state=None):
    env = WildfireEnv(cfg, rgb=False)
    obs, info = env.reset(seed=env_seed)
    if state is not None:
        env.set_state(state)
        obs, info = env.observe(), env._info()
    pol = make_policy(name, cfg, seed=seed)
    acts, seen = [], []
    for _ in range(steps):
        a = pol.act(obs, info)
        acts.append(a)
        obs, _, _, _, info = env.step(a)
        seen.append(set(info["tiles"]))
    return np.array(acts), pol, env, seen


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_deterministic_given_seed(small, name):
    a, *_ = _rollout(small, name, 3, 200)
    b, *_ = _rollout(small, name, 3, 200)
    c, *_ = _rollout(small, name, 4, 200)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_explore_return_revisits():
    cfg = Config.from_yaml("configs/default.yaml")
    acts, pol, env, seen = _rollout(cfg, "explore_return", 0, 5000)
    assert len(pol.log) >= 3
    start = cfg.start_step + 1  # info step after the first action
    for e in pol.log:
        tile = tuple(e["tile"])
        lo, hi = e["leave_step"], e["return_step"]
        assert hi - lo >= e["dt"] and 50 <= e["dt"] <= 2000
        assert tile in seen[lo - start] and tile in seen[hi - start]
        assert all(tile not in seen[t - start] for t in range(lo + 1, hi))


def test_firestarter_ignites(small):
    acts, *_ = _rollout(small.replace(p_lightning=0.0), "firestarter", 0, 500)
    assert 0 < (acts == A.IGNITE).sum() < 30


def test_firefighter_puts_out_fire(small):
    cfg = small.replace(p_lightning=0.0, beta=0.0, infinite_water=True)
    s = flat_state(cfg, veg=2)
    s.fire[25:28, 40:43] = 200  # long-burning patch, visible from the centre
    acts, pol, env, _ = _rollout(cfg, "firefighter", 0, 40, state=s)
    assert (acts == A.SPRAY).any()
    assert env.state.fire.max() == 0
