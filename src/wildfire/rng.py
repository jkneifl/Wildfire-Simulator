"""Counter-based random numbers (SPEC R2).

Every random number is a pure function of ``(seed, stream, step, purpose, y, x)``:

    bits = fmix64(key(seed, stream, step, purpose) ^ cell_hash(y, x))

``fmix64`` is the MurmurHash3 finalizer, a bijection on 64-bit integers with good
avalanche. ``cell_hash`` is ``fmix64((y << 32) | x)``, so it is unique per cell.
There is no hidden state; results do not depend on evaluation order or chunking.
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache

import numpy as np

MASK = (1 << 64) - 1
_C1 = 0xFF51AFD7ED558CCD
_C2 = 0xC4CEB9FE1A85EC53


class Purpose(IntEnum):
    SPREAD = 1
    LIGHTNING = 2
    GROWTH = 3
    SEED = 4
    DRY = 5
    TERRAIN = 6
    FLICKER = 7
    POLICY = 8  # scripted policies only, never the simulation
    FORK = 9  # derivation of fork stream ids


def _fmix_int(z: int) -> int:
    z &= MASK
    z ^= z >> 33
    z = (z * _C1) & MASK
    z ^= z >> 33
    z = (z * _C2) & MASK
    z ^= z >> 33
    return z


def key(seed: int, stream: int, step: int, purpose: int) -> int:
    """64-bit key for one (seed, stream, step, purpose) counter."""
    z = _fmix_int(int(seed) ^ 0x9E3779B97F4A7C15)
    z = _fmix_int(z ^ (int(stream) & MASK))
    z = _fmix_int(z ^ (int(step) & MASK))
    return _fmix_int(z ^ int(purpose))


def keys(seed, stream, step, purpose) -> np.ndarray:
    """Vector of keys for a batch of worlds (arguments broadcast, 1-d)."""
    seed, stream, step = np.broadcast_arrays(*(np.atleast_1d(a) for a in (seed, stream, step)))
    out = [key(a, b, c, purpose) for a, b, c in zip(seed.ravel(), stream.ravel(), step.ravel())]
    return np.array(out, dtype=np.uint64)


def fmix(z: np.ndarray) -> np.ndarray:
    """Vectorised fmix64 on uint64 arrays (returns a new array)."""
    with np.errstate(over="ignore"):  # wraparound mod 2**64 is intended (scalars warn)
        z = z ^ (z >> np.uint64(33))
        z *= np.uint64(_C1)
        z ^= z >> np.uint64(33)
        z *= np.uint64(_C2)
        z ^= z >> np.uint64(33)
    return z


def cell_hash(y, x) -> np.ndarray:
    y = np.asarray(y).astype(np.uint64)
    x = np.asarray(x).astype(np.uint64)
    return fmix((y << np.uint64(32)) | x)


@lru_cache(maxsize=8)
def grid_hash(height: int, width: int) -> np.ndarray:
    """Read-only ``cell_hash`` for every cell of an ``(H, W)`` grid."""
    y, x = np.indices((height, width))
    h = cell_hash(y, x)
    h.flags.writeable = False
    return h


def bits(k, ch: np.ndarray) -> np.ndarray:
    """Raw 64 random bits for keys ``k`` (broadcast) and cell hashes ``ch``."""
    return fmix(ch ^ np.asarray(k, dtype=np.uint64))


def to_unit(b: np.ndarray) -> np.ndarray:
    """Map 64 random bits to float64 in [0, 1) using the top 53 bits."""
    return (b >> np.uint64(11)).astype(np.float64) * (2.0**-53)


def threshold(p) -> np.ndarray:
    """uint64 threshold so that ``bits < threshold(p)`` has probability ``p``."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    # 2**64 * p, clipped to the largest float64 below 2**64 (p = 1 misses with prob ~1e-16).
    return np.minimum(p * 2.0**64, 2.0**64 - 2048.0).astype(np.uint64)


def uniform(seed: int, stream: int, step: int, purpose: int, y, x) -> np.ndarray:
    """Uniform floats in [0, 1) for cells ``(y, x)``; the reference definition."""
    return to_unit(bits(np.uint64(key(seed, stream, step, purpose)), cell_hash(y, x)))
