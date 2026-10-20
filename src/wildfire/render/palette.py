"""The single palette: dark tile map, teal vegetation, orange fire.

Four base colours; every other colour is a linear sRGB mix of them, rounded to the nearest
integer (halves up). Water and the wetness tint are proposed additions to the style.
"""

import numpy as np


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(a, b, f: float) -> tuple[int, int, int]:
    """``a + f * (b - a)`` per channel, rounded to the nearest integer."""
    return tuple(int(np.floor(x + f * (y - x) + 0.5)) for x, y in zip(a, b))


# base palette
GROUND = hex_rgb("#0F1A17")  # background, grid gaps, fog
FIRE = hex_rgb("#E8833A")  # burning cells, HUD warnings
INK = hex_rgb("#E9EFE9")  # agent, text, outlines
ACCENT = hex_rgb("#4FB3A3")  # HUD highlights; reserved for later overlays

# cells
BARE = mix(GROUND, INK, 0.07)
ROCK = mix(GROUND, INK, 0.30)
WATER = hex_rgb("#1D4A63")  # proposed
WATER_VARIANT = hex_rgb("#22556F")  # proposed, half of the water cells
EMBER = mix(GROUND, FIRE, 0.50)  # last burning bucket
ASH_FRESH = mix(GROUND, INK, 0.15)
ASH = (BARE, ASH_FRESH, mix(ASH_FRESH, BARE, 0.5), BARE)  # by ash bucket 0..3
VEG_F = (0.20, 0.28, 0.37, 0.46)  # by growth stage 1..4
VEG_DARK = 0.72  # darker variant (half of the cells)
VEG = (BARE, *(mix(GROUND, ACCENT, f) for f in VEG_F))
# mix(GROUND, VEG, 0.72) of the rounded colour: reproduces the style table exactly
# (mix(GROUND, ACCENT, 0.72 f) differs by one unit in green at stage 4).
VEG_VARIANT = (BARE, *(mix(GROUND, c, VEG_DARK) for c in VEG[1:]))

# flame: mix(FIRE, INK, FLAME_BASE + FLAME_AMP * sin(2 pi (t / FLAME_PERIOD + u)))
FLAME_BASE, FLAME_AMP, FLAME_PERIOD = 0.10, 0.10, 12

WET_TINT = hex_rgb("#5FA8D3")  # proposed
WET_MIX = 0.30  # mix(c, WET_TINT, WET_MIX * wet / 255)

# grid: gap = max(1, round(GAP_FRAC * cell_px)), none below GAP_MIN_PX unless forced
GAP_FRAC, GAP_MIN_PX = 0.09, 6

FOG = GROUND
NIGHT_MIN = 0.35  # SPEC 5: brightness NIGHT_MIN + (1 - NIGHT_MIN) * L, burning cells excluded
MUTED = mix(GROUND, INK, 0.60)  # secondary HUD text
