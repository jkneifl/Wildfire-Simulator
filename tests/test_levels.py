import numpy as np
import pytest

from wildfire.dynamics import run
from wildfire.levels import (
    N_FEATURES,
    clipmap,
    details,
    from_one_hot,
    level_origin,
    one_hot,
    prolong,
    restrict,
)
from wildfire.observe import grid
from wildfire.terrain import initial_state


@pytest.fixture
def world(small):
    cfg = small.replace(p_lightning=1e-3)
    s = run(initial_state(cfg, 2), [0] * 200, cfg)
    return grid(s, 0, 0, cfg.height, cfg.width, cfg)


def test_one_hot_round_trip(world):
    z = one_hot(world)
    assert z.shape == (*world.shape[:2], N_FEATURES)
    assert np.allclose(z.sum(-1), 5)  # one category per channel group
    assert np.array_equal(from_one_hot(z), world)


def test_restrict_prolong_details(world):
    z = one_hot(world).astype(np.float64)
    c = restrict(z)
    assert c.shape == (32, 32, N_FEATURES)
    assert np.allclose(restrict(prolong(c)), c)  # R P = I
    d = details(z)
    assert np.allclose(restrict(d), 0)  # details never change the coarse state
    assert np.allclose(prolong(c) + d, z)
    assert np.allclose(restrict(z, 4), restrict(restrict(z)))  # levels nest


def test_clipmap_levels(world):
    center = (40, 21)
    levels = clipmap(world, center, n=8, levels=3)
    padded = np.zeros((192, 192, 6), np.uint8)  # world surrounded by rock
    padded[..., 0] = 2
    padded[64:128, 64:128] = world
    direct = one_hot(padded)
    for lv in levels:
        s, (oy, ox) = lv["cell"], lv["origin"]
        assert lv["z"].shape == (8, 8, N_FEATURES)
        assert oy % s == 0 and ox % s == 0  # snapped to the level's grid
        # equals the block mean of a direct crop (outside the world = rock)
        crop = direct[64 + oy : 64 + oy + 8 * s, 64 + ox : 64 + ox + 8 * s]
        assert np.allclose(lv["z"], restrict(crop, s) if s > 1 else crop)
    # each level's region contains the finer one
    for a, b in zip(levels, levels[1:]):
        (ay, ax), (by, bx) = a["origin"], b["origin"]
        sa, sb = 8 * a["cell"], 8 * b["cell"]
        assert by <= ay and bx <= ax and ay + sa <= by + sb and ax + sa <= bx + sb


def test_clipmap_from_near_crop_matches_world(world):
    center, r = (30, 30), 20
    near = world[center[0] - r : center[0] + r + 1, center[1] - r : center[1] + r + 1]
    a = clipmap(near, center, 8, 2, source_origin=(10, 10), world_shape=world.shape[:2])
    b = clipmap(world, center, 8, 2)
    for x, y in zip(a, b):
        assert np.array_equal(x["z"], y["z"])
    with pytest.raises(ValueError, match="not covered"):
        clipmap(near, center, 8, 3, source_origin=(10, 10), world_shape=world.shape[:2])
    assert level_origin((30, 30), 8, 2) == (12, 12)
