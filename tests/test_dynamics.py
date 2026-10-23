import numpy as np

from wildfire import actions as A
from wildfire.dynamics import light, run, step, transition_probs
from wildfire.env import fork
from wildfire.observe import grid, transition_log_prob
from wildfire.state import GROUND, ROCK, WATER, equal, stack
from wildfire.terrain import initial_state

from .conftest import flat_state


def _random_actions(n, seed=0):
    from wildfire.rng import Purpose, uniform

    u = uniform(seed, 0, 0, Purpose.POLICY, np.zeros(n, int), np.arange(n))
    return (u * A.N_ACTIONS).astype(int)


def test_determinism_and_replay(small):
    acts = _random_actions(300)
    snaps = {}
    s = initial_state(small, 5)
    for i, a in enumerate(acts):
        s = step(s, a, small)
        if i % 50 == 0:
            snaps[i] = s.copy()
    s2 = initial_state(small, 5)
    for i, a in enumerate(acts):
        s2 = step(s2, a, small)
        if i in snaps:
            assert equal(s2, snaps[i])
    assert equal(s, s2)


def test_step_is_pure(small):
    s = initial_state(small, 1)
    s.fire[10, 10] = 5
    before = s.copy()
    step(s, A.SPRAY, small)
    assert equal(s, before)


def test_streams(small):
    cfg = small.replace(p_lightning=1e-3)
    s0 = initial_state(cfg, 2)
    a = run(s0, [0] * 200, cfg)
    s0b = initial_state(cfg, 2)
    assert equal(a, run(s0b, [0] * 200, cfg))
    s1 = initial_state(cfg, 2, stream=1)
    b = run(s1, [0] * 200, cfg)
    assert not np.array_equal(a.veg, b.veg)
    assert np.array_equal(a.terrain, b.terrain)  # terrain does not depend on stream


def test_batched_matches_single(small):
    cfg = small.replace(p_lightning=1e-3)
    singles = [initial_state(cfg, 3, stream=k) for k in range(3)]
    batch = stack(singles)
    acts = np.array([A.IGNITE, A.MOVE_E, A.SPRAY])
    for _ in range(50):
        batch = step(batch, acts, cfg)
        singles = [step(s, a, cfg) for s, a in zip(singles, acts)]
    for k in range(3):
        assert equal(batch[k], singles[k])


def test_water_and_rock_never_burn_or_grow(small):
    cfg = small.replace(p_lightning=0.05, p_seed=0.2, p_grow=0.5)
    s = initial_state(cfg, 4)
    blocked = s.terrain != GROUND
    assert (s.terrain == WATER).any() and (s.terrain == ROCK).any()
    for _ in range(300):
        s = step(s, 0, cfg)
        assert not (s.fire[blocked] > 0).any()
        assert not (s.veg[blocked] > 0).any()


def test_burning_burns_out_to_ash(small):
    cfg = small.replace(beta=0.0, p_lightning=0.0)
    s = flat_state(cfg, veg=2)
    s.fire[5, 5] = 3
    s = step(s, 0, cfg)
    assert s.fire[5, 5] == 2
    s = run(s, [0, 0], cfg)
    assert s.fire[5, 5] == 0 and s.veg[5, 5] == 0 and s.ash_age[5, 5] == 1
    s = step(s, 0, cfg)
    assert s.ash_age[5, 5] == 2


def test_ignition_burn_time(small):
    cfg = small.replace(beta=0.0, p_lightning=0.0)
    s = flat_state(cfg, veg=3, facing=1)  # facing east
    s = step(s, A.IGNITE, cfg)
    # ignited at action time with BURN_BASE + BURN_PER_VEG * veg, then decremented once
    assert s.fire[s.ay, s.ax + 1] == cfg.burn_base + cfg.burn_per_veg * 3 - 1


def test_water_extinguishes_and_wet_cells_do_not_ignite(small):
    cfg = small.replace(beta=1.0, p_lightning=0.0, m_night=1.0)
    s = flat_state(cfg, veg=4, facing=1)
    y, x = int(s.ay), int(s.ax) + 1
    s.fire[y - 1 : y + 2, x - 1 : x + 2] = 9
    s = step(s, A.SPRAY, cfg)
    assert (s.fire[y - 1 : y + 2, x - 1 : x + 2] == 0).all()
    assert (s.wet[y - 1 : y + 2, x - 1 : x + 2] > 250).all()

    # a wet ring around a fire stops it: a wet cell never ignites while wet >= WET_BLOCK
    s = flat_state(cfg, veg=4)
    s.wet[20:45, 20:45] = 255
    s.wet[30:35, 30:35] = 0
    s.fire[32, 32] = 9
    for _ in range(20):
        s = step(s, 0, cfg)
        assert not (s.fire[s.wet >= cfg.wet_block] > 0).any()
    assert not (s.fire[:20] > 0).any()

    # ignite action does nothing on wet cells
    s = flat_state(cfg, veg=4, facing=1)
    s.wet[s.ay, s.ax + 1] = 200
    s = step(s, A.IGNITE, cfg)
    assert s.fire.max() == 0


def _burned_after(cfg, t0, steps, seeds):
    total = 0
    for seed in seeds:
        s = flat_state(cfg, veg=4, t=t0, seed=seed)
        s.fire[32, 32] = cfg.fire_max
        s = run(s, [0] * steps, cfg)
        total += int(((s.fire > 0) | (s.ash_age > 0)).sum())
    return total


def test_spread_slower_at_night(small):
    # Long day so the light level is nearly constant over the test.
    cfg = small.replace(day_length=100_000, p_lightning=0.0, p_grow=0.0, p_seed=0.0, beta=0.2)
    noon, midnight = cfg.day_length // 2, 0
    assert light(noon, cfg.day_length) > 0.999 and light(midnight, cfg.day_length) == 0
    day = _burned_after(cfg, noon, 25, range(8))
    night = _burned_after(cfg, midnight, 25, range(8))
    assert night < 0.7 * day, (day, night)


def test_light_cycle():
    D = 240
    assert light(0, D) == 0 and abs(light(120, D) - 1) < 1e-12
    assert abs(light(60, D) - 0.5) < 1e-12 and light(240, D) == 0


def test_growth_only_in_daylight(small):
    cfg = small.replace(day_length=100_000, p_lightning=0.0, p_grow=0.2, p_seed=0.05)
    s = flat_state(cfg, veg=1, t=0)
    s.veg[:10] = 0
    night = run(s, [0] * 20, cfg)
    assert np.array_equal(night.veg, s.veg)
    day = run(flat_state(cfg, veg=1, t=cfg.day_length // 2), [0] * 20, cfg)
    assert (day.veg > 1).mean() > 0.9


def test_seeding_respects_regrow_delay_and_neighbours(small):
    cfg = small.replace(day_length=100_000, p_lightning=0.0, p_grow=0.0, p_seed=0.02)
    noon = cfg.day_length // 2
    s = flat_state(cfg, veg=0, t=noon)
    s.ash_age[:32] = 1  # fresh ash on the top half
    s.veg[40:, :] = 0
    s = run(s, [0] * 60, cfg)  # ash ages to 61 < REGROW_DELAY
    assert not (s.veg[:32] > 0).any()
    assert (s.veg[32:] > 0).any()
    # sprouted cells are no longer ash
    assert not ((s.veg > 0) & (s.ash_age > 0)).any()

    # more vegetated neighbours, faster seeding: P_SEED * L * (1 + k)
    lone = flat_state(cfg, veg=0, t=noon)
    dense = flat_state(cfg, veg=0, t=noon)
    dense.veg[:, ::2] = 4  # every bare cell has k = 2 vegetated neighbours
    a = (run(lone, [0], cfg).veg > 0).sum()
    b = ((run(dense, [0], cfg).veg > 0) & (dense.veg == 0)).sum()
    # expected: 4096 * 0.02 = 82 vs 2048 * 0.06 = 123
    assert b > a


def test_drying(small):
    cfg = small.replace(day_length=100_000, p_lightning=0.0, dry_rate=2.0)
    s = flat_state(cfg, t=0)  # night: rate 1.0 exactly
    s.wet[:] = 10
    assert (step(s, 0, cfg).wet == 9).all()
    s = flat_state(cfg, t=cfg.day_length // 2)  # noon: rate ~3
    s.wet[:] = 10
    w = step(s, 0, cfg).wet
    assert ((w == 7) | (w == 8)).all() and (w == 7).mean() > 0.95
    s.wet[:] = 1
    assert (step(s, 0, cfg).wet == 0).all()


def test_facing_tank_and_refill(small):
    cfg = small.replace(p_lightning=0.0, tank=3, beta=0.0)
    s = flat_state(cfg, veg=2, facing=0, tank=3)
    y, x = int(s.ay), int(s.ax)
    s = step(s, A.MOVE_W, cfg)  # moves west, faces west
    assert (int(s.ax), int(s.ay), int(s.facing)) == (x - 1, y, 3)
    s = step(s, A.IGNITE, cfg)
    assert s.fire[y, x - 2] > 0 and s.fire.sum() == s.fire[y, x - 2]
    s = step(s, A.SPRAY, cfg)
    assert s.fire.max() == 0 and s.wet[y, x - 2] > 250 and s.wet[y, x] == 0
    assert int(s.tank) == 2
    s = run(s, [A.SPRAY, A.SPRAY], cfg)
    assert int(s.tank) == 0
    s.wet[:] = 0
    s = step(s, A.SPRAY, cfg)  # empty tank: nothing happens
    assert s.wet.max() == 0 and int(s.tank) == 0

    # moves blocked by water and rock; refill next to water
    s.terrain[y, x - 2] = WATER
    s = step(s, A.MOVE_W, cfg)
    assert int(s.ax) == x - 1 and int(s.tank) == cfg.tank
    s.terrain[y - 1, x - 1] = ROCK
    s = step(s, A.MOVE_N, cfg)
    assert (int(s.ax), int(s.ay), int(s.facing)) == (x - 1, y, 0)


def test_infinite_water(small):
    cfg = small.replace(infinite_water=True, tank=1)
    s = run(flat_state(cfg, tank=1), [A.SPRAY] * 5, cfg)
    assert int(s.tank) == 1 and s.wet.max() > 0


def test_world_edge_is_rock(small):
    cfg = small.replace(p_lightning=0.0)
    s = flat_state(cfg, agent=(0, 0), facing=0)
    s = run(s, [A.MOVE_N, A.MOVE_W], cfg)
    assert (int(s.ax), int(s.ay)) == (0, 0)
    s = step(s, A.IGNITE, cfg)  # target outside the world
    assert s.fire.max() == 0


def test_lightning_rate_and_targets(small):
    cfg = small.replace(p_lightning=0.01, beta=0.0, p_grow=0.0, p_seed=0.0)
    s = flat_state(cfg, veg=2)
    s.veg[:, :32] = 0  # bare half never ignites
    strikes = 0
    for _ in range(20):
        s.fire[:] = 0  # count fresh strikes only
        s = step(s, 0, cfg)
        strikes += int((s.fire > 0).sum())
        assert not (s.fire[:, :32] > 0).any()
    n = 20 * 64 * 32
    assert abs(strikes / n - 0.01) < 4 * np.sqrt(0.01 * 0.99 / n)


def test_ash_age_saturates(small):
    cfg = small.replace(p_lightning=0.0, p_seed=0.0)
    s = flat_state(cfg, veg=0)
    s.ash_age[0, 0] = 65534
    s.ash_age[0, 1] = 65535
    s = run(s, [0, 0], cfg)
    assert s.ash_age[0, 0] == 65535 and s.ash_age[0, 1] == 65535 and s.ash_age.dtype == np.uint16


def _busy_state(cfg, t):
    """A state with fires, ash of all ages, wet cells and an agent next to vegetation."""
    s = initial_state(cfg, 3)
    s = run(s, _random_actions(150, seed=3), cfg)
    u = np.random.default_rng(0)  # test fixture only, not simulation randomness
    wet = u.integers(0, 256, s.wet.shape)
    s.wet = np.where(u.random(s.wet.shape) < 0.3, wet, s.wet).astype(np.uint8)
    bare = (s.veg == 0) & (s.fire == 0)
    age = u.integers(0, 2 * cfg.regrow_delay, s.ash_age.shape)
    s.ash_age = np.where(bare, age, 0).astype(np.uint16)
    s.t = np.asarray(t, np.int64)
    return s


def test_transition_probs_match_forks(small):
    cfg = small.replace(p_lightning=0.02, p_grow=0.2, p_seed=0.05, beta=0.35)
    D, M = cfg.day_length, 2000
    # L = 0.3 (fractional drying) with spray, noon with ignite, midnight (no growth) with noop.
    for t, a in [(round(D * 0.185), A.SPRAY), (D // 2, A.IGNITE), (D, A.NOOP)]:
        s = _busy_state(cfg, t)
        tp = transition_probs(s, a, cfg)
        p = tp["prob"]
        assert p.shape == (*s.veg.shape, 8) and (p >= 0).all() and (p <= 1).all()
        assert np.allclose(p.sum(-1), 1.0, atol=1e-12)
        b = fork(s, cfg, n=M, steps=1, actions=[a])
        match = np.ones((M, *p.shape), bool)
        for k in ("veg", "fire", "ash_age", "wet"):
            match &= getattr(b, k)[..., None] == tp[k][None]
        assert (match & (p[None] > 0)).any(-1).all()  # every realised cell is a possible outcome
        same = np.ones((*p.shape, 8), bool)
        for k in ("veg", "fire", "ash_age", "wet"):
            same &= tp[k][..., :, None] == tp[k][..., None, :]
        expected = (same * p[..., None, :]).sum(-1)
        freq = match.mean(0)
        tol = 6 * np.sqrt(expected * (1 - expected) / M) + 1e-12
        assert (np.abs(freq - expected) <= tol).all()
        assert ((expected > 0.01) & (expected < 0.99)).any(-1).sum() > 100  # many random cells
        # Pooled calibration: per outcome and bin of predicted probability, the total count
        # over all cells must match (catches small biases the per-cell check cannot).
        edges = [1e-12, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1 - 1e-12]
        for lo, hi in zip(edges[:-1], edges[1:]):
            for j in range(8):
                sel = (expected[..., j] > lo) & (expected[..., j] <= hi)
                if not sel.any():
                    continue
                e_j = expected[..., j][sel]
                z = (freq[..., j][sel] - e_j).sum() * M / np.sqrt((M * e_j * (1 - e_j)).sum())
                assert abs(z) < 6, (t, a, j, lo, hi, z)
        h, w = s.veg.shape
        for i in range(20):
            lp = transition_log_prob(tp, grid(b[i], 0, 0, h, w, cfg), cfg)
            assert np.isfinite(lp).all() and (lp <= 1e-12).all()
