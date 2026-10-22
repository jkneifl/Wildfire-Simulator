import numpy as np

from wildfire.metrics import (
    TILE_STATS,
    burn_probability,
    energy_distance,
    global_stats,
    mmd,
    monte_carlo,
    patch_sizes,
    regime_summary,
    retention_curve,
    tile_stats,
)
from wildfire.state import WATER, stack
from wildfire.terrain import initial_state

from .conftest import flat_state


def test_global_and_patch_stats(small):
    s = flat_state(small, veg=0)
    s.veg[:10, :10] = 2
    s.veg[20:22, 20:23] = 1
    s.terrain[60:, :] = WATER
    st = global_stats(s)
    assert st["veg_density"] == (100 + 6) / (60 * 64)
    assert sorted(patch_sizes(s.veg > 0).tolist()) == [6, 100]


def test_tile_stats_match_manual(small):
    s = initial_state(small, 2)
    s.fire[3, 3] = 4
    s.wet[17, 1] = 255
    ts = tile_stats(s, small)
    assert ts.shape == (16, len(TILE_STATS))
    for tile in [(0, 0), (1, 0), (3, 2)]:
        ys = slice(tile[0] * 16, tile[0] * 16 + 16)
        xs = slice(tile[1] * 16, tile[1] * 16 + 16)
        g = s.terrain[ys, xs] == 0
        n = g.sum()
        expect = [
            ((s.veg[ys, xs] > 0) & g).sum() / n,
            (s.veg[ys, xs] / small.v_max * g).sum() / n,
            ((s.ash_age[ys, xs] > 0) & g).sum() / n,
            ((s.fire[ys, xs] > 0) & g).sum() / n,
            (s.wet[ys, xs] / 255 * g).sum() / n,
        ]
        got = tile_stats(s, small, [tile])[0]
        assert np.allclose(got, expect) and np.allclose(ts[tile[0] * 4 + tile[1]], expect)
    # batched states give per-world results; non-multiple world sizes are padded
    b = stack([s, s])
    assert np.allclose(tile_stats(b, small)[1], ts)
    odd = small.replace(height=40, width=50)
    assert tile_stats(initial_state(odd, 0), odd).shape == (3 * 4, len(TILE_STATS))


def test_monte_carlo_and_burn_probability(small):
    s = initial_state(small, 1)
    a = monte_carlo(s, small, 30, 4, tiles=[(0, 0), (2, 1)])
    assert a.shape == (4, 2, len(TILE_STATS))
    assert np.array_equal(a, monte_carlo(s, small, 30, 4, tiles=[(0, 0), (2, 1)]))
    lit = np.zeros(s.shape, bool)
    lit[25:35, 25:35] = s.veg[25:35, 25:35] > 0
    s.fire[lit] = 5
    p = burn_probability(s, small, n=4, steps=10)
    assert p.shape == s.shape and p.min() >= 0 and p.max() <= 1
    assert lit.any() and (p[lit] == 1).all() and p[s.terrain != 0].max() == 0


def test_retention_curve_ar1():
    # tiles follow independent AR(1) processes: corr at lag L is phi ** L
    phi, n_t, n_tiles = 0.8, 400, 200
    rng = np.random.default_rng(0)  # test data only, not simulation randomness
    x = np.zeros((n_t, n_tiles))
    for t in range(1, n_t):
        x[t] = phi * x[t - 1] + rng.normal(size=n_tiles)
    series = [np.repeat(x[..., None], len(TILE_STATS), axis=-1)]
    r = retention_curve(series, [1, 2, 5], "veg_density")
    assert np.allclose(r, phi ** np.array([1, 2, 5]), atol=0.03)


def test_distances():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    z = rng.normal(size=(200, 3)) + 1.0
    assert abs(energy_distance(x, y)) < 0.05 < energy_distance(x, z)
    assert abs(mmd(x, y)) < 0.02 < mmd(x, z)
    assert energy_distance(x, x) == 0


def test_regime_summary(small):
    stats = {
        "active_fires": np.array([1, 0, 0, 0, 2, 0, 3]),
        "veg_density": np.linspace(0.2, 0.8, 7),
        "burned_frac": np.zeros(7),
    }
    r = regime_summary(stats, small.replace(day_length=3))
    assert r["frac_steps_with_fire"] == 3 / 7 and r["longest_fire_free_days"] == 1.0


def test_obs_stats_match_state_stats(small):
    from wildfire.dynamics import run
    from wildfire.metrics import obs_patch_stats, obs_stats, obs_tiles
    from wildfire.observe import grid

    cfg = small.replace(p_lightning=2e-3)
    s = run(initial_state(cfg, 5), [0] * 150, cfg)
    g = grid(s, 0, 0, 64, 64, cfg)
    a = obs_stats(obs_tiles(g, 16), cfg).reshape(16, -1)
    b = tile_stats(s, cfg)
    assert np.allclose(a[:, :4], b[:, :4])  # wetness differs only by bucketing
    odd = obs_tiles(g[:50, :40], 16)
    assert odd.shape == (4, 3, 16, 16, 6)
    assert np.array_equal(np.sort(obs_patch_stats(g)["vegetated"]), np.sort(patch_sizes(s.veg > 0)))
