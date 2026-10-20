import numpy as np

from wildfire import rng
from wildfire.rng import Purpose


def test_pure_and_chunk_independent():
    y, x = np.indices((40, 50))
    full = rng.uniform(3, 1, 7, Purpose.SPREAD, y, x)
    assert np.array_equal(full, rng.uniform(3, 1, 7, Purpose.SPREAD, y, x))
    # a sub-window evaluated on its own gives the same numbers
    sub = rng.uniform(3, 1, 7, Purpose.SPREAD, y[10:20, 5:30], x[10:20, 5:30])
    assert np.array_equal(sub, full[10:20, 5:30])


def test_every_coordinate_matters():
    y, x = np.indices((64, 64))
    base = rng.uniform(1, 0, 0, Purpose.GROWTH, y, x)
    for args in [
        (2, 0, 0, Purpose.GROWTH),
        (1, 1, 0, Purpose.GROWTH),
        (1, 0, 1, Purpose.GROWTH),
        (1, 0, 0, Purpose.SEED),
    ]:
        other = rng.uniform(*args, y, x)
        assert abs(np.corrcoef(base.ravel(), other.ravel())[0, 1]) < 0.05
    assert len(np.unique(base)) == base.size


def test_uniform_moments_and_threshold():
    y, x = np.indices((256, 256))
    u = rng.uniform(0, 0, 0, Purpose.DRY, y, x)
    assert 0 <= u.min() and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.005
    assert abs(u.var() - 1 / 12) < 0.002
    b = rng.bits(np.uint64(rng.key(0, 0, 0, Purpose.DRY)), rng.cell_hash(y, x))
    for p in (0.0, 0.01, 0.3, 1.0):
        assert abs((b < rng.threshold(p)).mean() - p) < 0.005


def test_batched_keys_match_scalar():
    k = rng.keys(np.array([1, 2]), np.array([5, 6]), 3, Purpose.SPREAD)
    assert k[0] == rng.key(1, 5, 3, Purpose.SPREAD)
    assert k[1] == rng.key(2, 6, 3, Purpose.SPREAD)
