"""Level-of-detail representation of observable state (nested regions around a point).

The fine observable grid (``observe.grid`` channels) is one-hot encoded per category, so a
coarse cell is the fraction of each category in its block:

- restriction ``R``: mean over ``2 x 2`` blocks; prolongation ``P``: piecewise constant.
  Hence ``R(P(c)) = c``, and the details ``d = z - P(R(z))`` satisfy ``R(d) = 0``.
- :func:`clipmap`: levels ``l = 0..L`` around a centre, each ``n x n`` cells of size ``2**l``,
  snapped to the level's own grid (as in geometry clipmaps), so a coarse cell always covers
  the same world cells.
"""

from __future__ import annotations

import numpy as np

from .state import ROCK

# (channel index in observe.grid, number of categories)
CATEGORIES = (
    ("terrain", 0, 3),
    ("veg", 1, 5),
    ("fire_bucket", 3, 4),
    ("wet_bucket", 4, 4),
    ("ash_bucket", 5, 4),
)  # noqa: E501
N_FEATURES = sum(k for _, _, k in CATEGORIES)
FEATURES = tuple(f"{name}={v}" for name, _, k in CATEGORIES for v in range(k))


def one_hot(grid: np.ndarray, dtype=np.float32) -> np.ndarray:
    """``(..., 6)`` uint8 observable grid -> ``(..., N_FEATURES)`` category indicators."""
    parts = [np.eye(k, dtype=dtype)[grid[..., ch]] for _, ch, k in CATEGORIES]
    return np.concatenate(parts, axis=-1)


def from_one_hot(z: np.ndarray) -> np.ndarray:
    """Most likely category per channel -> ``(..., 6)`` uint8 grid (burning from fire bucket)."""
    out = np.zeros((*z.shape[:-1], 6), np.uint8)
    i = 0
    for _, ch, k in CATEGORIES:
        out[..., ch] = np.argmax(z[..., i : i + k], axis=-1)
        i += k
    out[..., 2] = out[..., 3] > 0
    return out


def restrict(z: np.ndarray, factor: int = 2) -> np.ndarray:
    """Block mean over ``factor x factor`` cells of the last-but-one two axes ``(..., H, W, F)``."""
    *lead, h, w, f = z.shape
    if h % factor or w % factor:
        raise ValueError(f"shape {(h, w)} not divisible by {factor}")
    return z.reshape(*lead, h // factor, factor, w // factor, factor, f).mean(axis=(-4, -2))


def prolong(c: np.ndarray, factor: int = 2) -> np.ndarray:
    """Piecewise-constant upsampling, the right inverse of :func:`restrict`."""
    return np.repeat(np.repeat(c, factor, axis=-3), factor, axis=-2)


def details(z: np.ndarray, factor: int = 2) -> np.ndarray:
    """``z - P(R(z))``: what coarsening by ``factor`` removes (``restrict(details) == 0``)."""
    return z - prolong(restrict(z, factor), factor)


def level_origin(center: tuple[int, int], n: int, level: int) -> tuple[int, int]:
    """Top-left world cell of level ``level`` (side ``n * 2**level``), snapped to ``2**level``."""
    s = 2**level
    half = n * s // 2
    return tuple(int(np.floor((c - half) / s) * s) for c in center)


def _rock_features() -> np.ndarray:
    g = np.zeros(6, np.uint8)
    g[0] = ROCK
    return one_hot(g)


def clipmap(
    source: np.ndarray,
    center: tuple[int, int],
    n: int,
    levels: int,
    source_origin: tuple[int, int] = (0, 0),
    world_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """Nested levels around ``center = (y, x)`` from an observable grid ``source``.

    ``source`` is a ``(h, w, 6)`` grid whose cell ``(0, 0)`` is world cell ``source_origin``
    (a full-world grid, or a near crop). Cells outside the world are rock. Raises if a level
    needs in-world cells that ``source`` does not cover. Returns one dict per level with
    ``level``, ``origin`` (world cell of the top-left corner), ``cell`` (cell size) and
    ``z``: ``(n, n, N_FEATURES)`` category fractions.
    """
    if n % 2:
        raise ValueError("n must be even")
    H, W = world_shape if world_shape is not None else source.shape[:2]
    sy, sx = source_origin
    out = []
    for level in range(levels + 1):
        s = 2**level
        oy, ox = level_origin(center, n, level)
        side = n * s
        # in-world part of the region, which source must cover
        ya, yb = max(oy, 0), min(oy + side, H)
        xa, xb = max(ox, 0), min(ox + side, W)
        if (
            ya < yb
            and xa < xb
            and (ya < sy or xa < sx or yb > sy + source.shape[0] or xb > sx + source.shape[1])
        ):
            raise ValueError(f"level {level} region {(oy, ox, side)} not covered by the source")
        fine = np.broadcast_to(_rock_features(), (side, side, N_FEATURES)).copy()
        if ya < yb and xa < xb:
            fine[ya - oy : yb - oy, xa - ox : xb - ox] = one_hot(
                source[ya - sy : yb - sy, xa - sx : xb - sx]
            )
        out.append(
            {
                "level": level,
                "origin": (oy, ox),
                "cell": s,
                "z": restrict(fine, s) if s > 1 else fine,
            }
        )
    return out
