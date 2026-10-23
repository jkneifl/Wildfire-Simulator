"""Pure step function (SPEC 3.3). Works on one world ``(H, W)`` or a batch ``(n, H, W)``."""

from __future__ import annotations

import numpy as np

from . import rng
from .actions import apply_action
from .config import Config
from .rng import Purpose
from .state import GROUND, SCALARS, WorldState

_D = 1 / np.sqrt(2)
# (dy, dx, w_n): neighbour offsets and spread weights.
N8 = (
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (-1, -1, _D),
    (-1, 1, _D),
    (1, -1, _D),
    (1, 1, _D),
)
ASH_MAX = np.iinfo(np.uint16).max


def light(t, day_length: int):
    """Light level L_t: 0 at midnight, 1 at noon."""
    phase = (np.asarray(t) % day_length) / day_length
    return 0.5 * (1 - np.cos(2 * np.pi * phase))


def spread_weights(s: WorldState, cfg: Config):
    """Neighbour weights ``w_n`` of the spread rule.

    Hook for wind and slope (out of scope for v1): return per-direction weights,
    scalars or arrays broadcastable to the candidate cells.
    """
    return N8


def _batched(s: WorldState) -> WorldState:
    if s.batched:
        return s
    d = s.as_dict()
    return WorldState(**{k: v[None] if k not in SCALARS else v.reshape(1) for k, v in d.items()})


def _unbatched(s: WorldState) -> WorldState:
    return WorldState(
        **{k: v.reshape(()) if k in SCALARS else v[0] for k, v in s.as_dict().items()}
    )


def step(state: WorldState, action, cfg: Config) -> WorldState:
    """Advance one step. Pure: ``state`` is not modified."""
    single = not state.batched
    s = apply_action(_batched(state), action, cfg)  # 1. agent action
    n, h, w = s.veg.shape
    terrain, veg, fire, ash, wet = s.terrain, s.veg, s.fire, s.ash_age, s.wet
    t = s.t
    L = light(t, cfg.day_length)
    gh = rng.grid_hash(h, w)

    # 2. Burning.
    burning = fire > 0
    fire_new = fire - burning.astype(np.uint8)
    burnt_out = burning & (fire_new == 0)

    # 3. Spread.
    ignited = np.zeros_like(burning)
    if burning.any():
        bp = np.pad(burning, ((0, 0), (1, 1), (1, 1)))
        near = np.zeros_like(burning)
        for dy, dx, _ in N8:
            near |= bp[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        cand = near & ~burning & (veg > 0) & (wet < cfg.wet_block)
        b, y, x = np.nonzero(cand)
        if b.size:
            m = cfg.m_night + (1 - cfg.m_night) * L[b]
            g = np.maximum(0.0, 1 - wet[b, y, x] / cfg.wet_block)
            q = cfg.beta * m * (veg[b, y, x] / cfg.v_max) * g
            keep = np.ones(b.size)
            for dy, dx, wn in spread_weights(s, cfg):
                keep *= np.where(bp[b, 1 + y + dy, 1 + x + dx], 1 - q * wn, 1.0)
            k = rng.keys(s.seed, s.stream, t, Purpose.SPREAD)
            u = rng.to_unit(rng.bits(k[b], gh[y, x]))
            hit = u < 1 - keep
            ignited[b[hit], y[hit], x[hit]] = True

    # 4. Lightning.
    if cfg.p_lightning > 0:
        k = rng.keys(s.seed, s.stream, t, Purpose.LIGHTNING)[:, None, None]
        strike = rng.bits(k, gh) < rng.threshold(cfg.p_lightning)
        ignited |= strike & (veg > 0) & ~burning

    fire_new = np.where(ignited, cfg.burn_base + cfg.burn_per_veg * veg, fire_new).astype(np.uint8)
    veg_new = np.where(burnt_out, np.uint8(0), veg)

    # 7. Ash (computed before growth so that seeding can clear it).
    ash_new = ash + ((ash > 0) & (ash < ASH_MAX)).astype(np.uint16)
    ash_new[burnt_out] = 1

    # 5. Growth and seeding (daylight only). Each cell uses exactly one purpose:
    # GROWTH where veg > 0, SEED where veg == 0, so one hash serves both.
    if np.any(L > 0):
        kg = rng.keys(s.seed, s.stream, t, Purpose.GROWTH)[:, None, None]
        ks = rng.keys(s.seed, s.stream, t, Purpose.SEED)[:, None, None]
        r = rng.bits(np.where(veg > 0, kg, ks), gh)

        grow = (veg >= 1) & (veg < cfg.v_max) & ~burning & ~ignited
        grow &= r < rng.threshold(cfg.p_grow * L)[:, None, None]

        vp = np.pad(veg > 0, ((0, 0), (1, 1), (1, 1)))
        k_orth = (
            vp[:, :-2, 1:-1].astype(np.uint8) + vp[:, 2:, 1:-1] + vp[:, 1:-1, :-2] + vp[:, 1:-1, 2:]
        )
        thr = rng.threshold(cfg.p_seed * L[:, None] * (1 + np.arange(5))[None, :])  # (n, 5)
        bare = (veg == 0) & (terrain == GROUND) & ((ash == 0) | (ash > cfg.regrow_delay))
        sprout = bare & (r < thr[np.arange(n)[:, None, None], k_orth])

        veg_new = np.where(grow, veg + np.uint8(1), veg_new)
        veg_new[sprout] = 1
        ash_new[sprout] = 0  # regrown cells are no longer ash

    # 6. Drying (stochastic rounding of DRY_RATE * (0.5 + L)).
    wet_new = wet
    b, y, x = np.nonzero(wet)
    if b.size:
        rate = cfg.dry_rate * (0.5 + L)
        whole = np.floor(rate)
        k = rng.keys(s.seed, s.stream, t, Purpose.DRY)
        u = rng.to_unit(rng.bits(k[b], gh[y, x]))
        dec = whole[b] + (u < (rate - whole)[b])
        wet_new = wet.copy()
        wet_new[b, y, x] = np.maximum(wet[b, y, x] - dec, 0).astype(np.uint8)

    out = WorldState(
        terrain=terrain,
        veg=veg_new,
        fire=fire_new,
        ash_age=ash_new,
        wet=wet_new,
        t=t + 1,
        seed=s.seed,
        stream=s.stream,
        ax=s.ax,
        ay=s.ay,
        facing=s.facing,
        tank=s.tank,
    )
    return _unbatched(out) if single else out


OUTCOMES = 8  # per cell: ignition, growth or seeding, extra unit of drying (bits 4, 2, 1)


def _unit_prob(x):
    """``P(rng.to_unit(bits) < x)``: ``to_unit`` takes the values ``j * 2**-53``."""
    return np.clip(np.ceil(np.asarray(x, np.float64) * 2.0**53) * 2.0**-53, 0.0, 1.0)


def _bits_prob(thr):
    """``P(bits < thr)`` for a uint64 threshold from ``rng.threshold``."""
    return np.asarray(thr).astype(np.float64) * 2.0**-64


def transition_probs(state: WorldState, action, cfg: Config) -> dict[str, np.ndarray]:
    """Exact one-step distribution of every cell (oracle for evaluation).

    Given the state and the action, cells are independent: every random number is a per-cell,
    per-purpose draw. Each cell has three binary events: ignition (spread or lightning),
    growth or seeding (never after ignition), and the extra unit of stochastic drying.
    Returns the next ``veg``, ``fire``, ``ash_age`` and ``wet`` of each of the 8 outcomes and
    their probability ``prob``, all of shape ``(..., H, W, 8)``. Outcome ``j`` has ignition
    ``j >> 2 & 1``, growth or seeding ``j >> 1 & 1`` and drying ``j & 1``; outcomes with
    probability 0 may repeat others. Mirrors :func:`step`, which stays the reference.
    """
    single = not state.batched
    s = apply_action(_batched(state), action, cfg)
    n, h, w = s.veg.shape
    terrain, veg, fire, ash, wet = s.terrain, s.veg, s.fire, s.ash_age, s.wet
    Lw = light(s.t, cfg.day_length)  # (n,)
    L = Lw[:, None, None]
    burning = fire > 0

    # Ignition: spread (one draw, as in step) or lightning (an independent draw).
    bp = np.pad(burning, ((0, 0), (1, 1), (1, 1)))
    m = cfg.m_night + (1 - cfg.m_night) * L
    g = np.maximum(0.0, 1 - wet / cfg.wet_block)
    q = cfg.beta * m * (veg / cfg.v_max) * g
    keep = np.ones(veg.shape)
    for dy, dx, wn in spread_weights(s, cfg):
        keep *= np.where(bp[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w], 1 - q * wn, 1.0)
    cand = ~burning & (veg > 0) & (wet < cfg.wet_block)
    p_spread = np.where(cand, _unit_prob(1 - keep), 0.0)
    p_light = _bits_prob(rng.threshold(cfg.p_lightning)) if cfg.p_lightning > 0 else 0.0
    p_light = np.where((veg > 0) & ~burning, p_light, 0.0)
    p_ign = 1 - (1 - p_spread) * (1 - p_light)

    # Growth (veg > 0) or seeding (bare ground): one draw, mutually exclusive cells.
    can_grow = (veg >= 1) & (veg < cfg.v_max) & ~burning
    p_grow = np.where(can_grow, _bits_prob(rng.threshold(cfg.p_grow * Lw))[:, None, None], 0.0)
    vp = np.pad(veg > 0, ((0, 0), (1, 1), (1, 1)))
    k_orth = (
        vp[:, :-2, 1:-1].astype(np.uint8) + vp[:, 2:, 1:-1] + vp[:, 1:-1, :-2] + vp[:, 1:-1, 2:]
    )
    thr = rng.threshold(cfg.p_seed * Lw[:, None] * (1 + np.arange(5))[None, :])  # (n, 5)
    bare = (veg == 0) & (terrain == GROUND) & ((ash == 0) | (ash > cfg.regrow_delay))
    p_seed = np.where(bare, _bits_prob(thr[np.arange(n)[:, None, None], k_orth]), 0.0)
    p_gs = p_grow + p_seed

    # Drying: whole units plus one more with the fractional probability.
    rate = cfg.dry_rate * (0.5 + Lw)
    whole = np.floor(rate)
    p_dry = np.where(wet > 0, _unit_prob(rate - whole)[:, None, None], 0.0)

    j = np.arange(OUTCOMES)
    ign, gs, dry = (j >> 2 & 1).astype(bool), (j >> 1 & 1).astype(bool), (j & 1).astype(bool)

    def e(a):
        return np.asarray(a)[..., None]

    pg = e(p_gs) * ~ign  # growth or seeding never follows ignition
    prob = (
        np.where(ign, e(p_ign), 1 - e(p_ign))
        * np.where(gs, pg, 1 - pg)
        * np.where(dry, e(p_dry), 1 - e(p_dry))
    )

    burnt_out = burning & (fire == 1)
    fire_ign = (cfg.burn_base + cfg.burn_per_veg * veg).astype(np.uint8)
    fire_out = np.where(ign, e(fire_ign), e(fire - burning.astype(np.uint8))).astype(np.uint8)
    veg_out = np.where(burnt_out, np.uint8(0), veg)
    veg_out = np.where(gs & e(can_grow), e(veg + np.uint8(1)), e(veg_out))
    veg_out = np.where(gs & e(bare), np.uint8(1), veg_out).astype(np.uint8)
    ash1 = ash + ((ash > 0) & (ash < ASH_MAX)).astype(np.uint16)
    ash1[burnt_out] = 1
    ash_out = np.where(gs & e(bare), np.uint16(0), e(ash1)).astype(np.uint16)
    dec = whole[:, None, None, None] + dry
    wet_out = np.where(e(wet > 0), np.maximum(e(wet) - dec, 0), e(wet)).astype(np.uint8)

    out = {"veg": veg_out, "fire": fire_out, "ash_age": ash_out, "wet": wet_out, "prob": prob}
    return {k: v[0] for k, v in out.items()} if single else out


def run(state: WorldState, actions, cfg: Config) -> WorldState:
    """Apply a sequence of actions."""
    for a in actions:
        state = step(state, a, cfg)
    return state
