import numpy as np
import pytest

from wildfire import Config, WildfireEnv
from wildfire import actions as A
from wildfire.policies import POLICIES, bump_action, make_policy, random_action

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


def test_random_action():
    assert all(random_action(0, t, 0.0) is None for t in range(100))
    a = [random_action(0, t, 0.2) for t in range(20000)]
    assert a == [random_action(0, t, 0.2) for t in range(20000)]
    taken = np.array([x for x in a if x is not None])
    assert abs(len(taken) / len(a) - 0.2) < 0.02
    counts = np.bincount(taken, minlength=A.N_ACTIONS)
    assert counts.size == A.N_ACTIONS and np.all(np.abs(counts / len(taken) - 1 / 7) < 0.02)


def test_bump_action(small):
    r = small.window // 2
    grid = np.zeros((small.window, small.window, 6), np.uint8)  # all ground
    obs = {"grid": grid}
    assert all(bump_action(0, t, 1.0, obs, small) is None for t in range(100))  # no obstacle
    grid[r, r + 1, 0] = 1  # water to the east
    assert all(bump_action(0, t, 0.0, obs, small) is None for t in range(100))
    assert all(bump_action(0, t, 1.0, obs, small) == A.MOVE_E for t in range(100))
    taken = [bump_action(0, t, 0.3, obs, small) for t in range(10000)]
    assert abs(sum(a is not None for a in taken) / len(taken) - 0.3) < 0.02
    # independent of random_action's draw at the same seed and step
    both = [random_action(0, t, 0.3) is not None and a is not None for t, a in enumerate(taken)]
    assert abs(np.mean(both) - 0.09) < 0.015
