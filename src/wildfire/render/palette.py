"""The single palette (RGB, uint8-range floats)."""

import numpy as np

GROUND = (150, 126, 86)
WATER = (40, 92, 170)
ROCK = (112, 112, 118)
# terrain id -> colour
TERRAIN = np.array([GROUND, WATER, ROCK], np.float32)
# veg stage 1..4 (index 0 unused)
VEG = np.array([GROUND, (156, 176, 84), (98, 152, 62), (52, 122, 46), (26, 86, 36)], np.float32)
# fire intensity bucket 1..3 (index 0 unused)
FIRE = np.array([(0, 0, 0), (196, 48, 20), (246, 122, 26), (255, 214, 80)], np.float32)
# ash bucket 1..3, fading towards ground (index 0 unused)
ASH = np.array([GROUND, (52, 48, 46), (88, 80, 70), (124, 108, 80)], np.float32)
WET_TINT = np.array((60, 120, 225), np.float32)
WET_STRENGTH = 0.15  # blend per wetness bucket
FOG = (18, 18, 24)
GRID_LINE = 0.75  # brightness factor of grid-line pixels
NIGHT_MIN = 0.35  # brightness = NIGHT_MIN + (1 - NIGHT_MIN) * L
