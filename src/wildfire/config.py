"""Frozen configuration dataclass, loaded from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    # world
    height: int = 256
    width: int = 256
    day_length: int = 240
    start_hour: float = 6.0  # t0 = round(start_hour / 24 * day_length)
    water_frac: float = 0.08
    rock_frac: float = 0.04
    # dynamics (SPEC 3.3)
    v_max: int = 4
    beta: float = 0.35
    m_night: float = 0.5
    wet_block: int = 128
    burn_base: int = 3
    burn_per_veg: int = 2
    p_lightning: float = 2e-7
    p_grow: float = 0.01
    p_seed: float = 0.002
    regrow_delay: int = 120
    dry_rate: float = 2.0
    # agent
    tank: int = 20
    infinite_water: bool = False
    # observation and rendering
    window: int = 33
    pixels_per_cell: int = 4
    tile: int = 16
    # recording
    snapshot_every: int = 50

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def replace(self, **changes) -> Config:
        return dataclasses.replace(self, **changes)

    @property
    def start_step(self) -> int:
        return round(self.start_hour / 24 * self.day_length)

    @property
    def fire_max(self) -> int:
        return self.burn_base + self.burn_per_veg * self.v_max
