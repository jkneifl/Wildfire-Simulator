import numpy as np
import pytest

from wildfire import Config
from wildfire.state import GROUND, WorldState


def flat_state(
    cfg: Config,
    veg: int = 4,
    t: int | None = None,
    seed: int = 0,
    stream: int = 0,
    agent=(None, None),
    facing: int = 2,
    tank: int | None = None,
) -> WorldState:
    """All-ground world with uniform vegetation; agent in the centre by default."""
    h, w = cfg.height, cfg.width
    i64 = lambda v: np.asarray(v, np.int64)  # noqa: E731
    ax, ay = agent
    return WorldState(
        terrain=np.full((h, w), GROUND, np.uint8),
        veg=np.full((h, w), veg, np.uint8),
        fire=np.zeros((h, w), np.uint8),
        ash_age=np.zeros((h, w), np.uint16),
        wet=np.zeros((h, w), np.uint8),
        t=i64(cfg.start_step if t is None else t),
        seed=i64(seed),
        stream=i64(stream),
        ax=i64(w // 2 if ax is None else ax),
        ay=i64(h // 2 if ay is None else ay),
        facing=i64(facing),
        tank=i64(cfg.tank if tank is None else tank),
    )


@pytest.fixture
def small() -> Config:
    return Config.from_yaml("configs/small.yaml")
