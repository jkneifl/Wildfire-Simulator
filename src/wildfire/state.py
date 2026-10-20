"""World state: a dataclass of arrays.

Layers have shape ``(H, W)`` for one world or ``(n, H, W)`` for a batch. Scalars
(``t``, ``seed``, ``stream``) and agent fields are int64 arrays of the batch shape:
0-d for one world, ``(n,)`` for a batch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

GROUND, WATER, ROCK = 0, 1, 2

LAYERS = ("terrain", "veg", "fire", "ash_age", "wet")
SCALARS = ("t", "seed", "stream", "ax", "ay", "facing", "tank")
DTYPES = {
    "terrain": np.uint8,
    "veg": np.uint8,
    "fire": np.uint8,
    "ash_age": np.uint16,
    "wet": np.uint8,
}


@dataclass
class WorldState:
    terrain: np.ndarray
    veg: np.ndarray
    fire: np.ndarray
    ash_age: np.ndarray
    wet: np.ndarray
    t: np.ndarray
    seed: np.ndarray
    stream: np.ndarray
    ax: np.ndarray  # agent x
    ay: np.ndarray  # agent y
    facing: np.ndarray  # 0 N, 1 E, 2 S, 3 W
    tank: np.ndarray

    @property
    def batched(self) -> bool:
        return self.veg.ndim == 3

    @property
    def shape(self) -> tuple[int, int]:
        return self.veg.shape[-2:]

    def copy(self) -> WorldState:
        return WorldState(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

    def as_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __getitem__(self, i: int) -> WorldState:
        """World ``i`` of a batch, as an unbatched copy."""
        return WorldState(**{k: np.array(v[i]) for k, v in self.as_dict().items()})

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.as_dict())

    @classmethod
    def load(cls, path: str | Path) -> WorldState:
        with np.load(path) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls)})


def stack(states: list[WorldState]) -> WorldState:
    """Stack unbatched states into one batched state."""
    return WorldState(
        **{f.name: np.stack([getattr(s, f.name) for s in states]) for f in fields(WorldState)}
    )


def equal(a: WorldState, b: WorldState) -> bool:
    return all(np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(WorldState))
