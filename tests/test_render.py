import os

import numpy as np

from wildfire import actions as A
from wildfire.render import palette as P
from wildfire.render.topdown import gap_px, render

from .conftest import flat_state


def test_shape_and_reproducible(small):
    s = flat_state(small)
    s.fire[30:34, 30:34] = 5
    a = render(s, (20, 20, 33, 33), small)
    assert a.shape == (33 * small.pixels_per_cell, 33 * small.pixels_per_cell, 3)
    assert a.dtype == np.uint8
    assert np.array_equal(a, render(s, (20, 20, 33, 33), small))


def test_night_dims_but_fire_stays_bright(small):
    cfg = small.replace(day_length=240)
    day = flat_state(cfg, t=120)
    night = flat_state(cfg, t=0)
    for s in (day, night):
        s.fire[10, 10] = 9
    d = render(day, (0, 0, 20, 20), cfg, ppc=1).astype(int)
    n = render(night, (0, 0, 20, 20), cfg, ppc=1).astype(int)
    assert n[0, 0].sum() < 0.5 * d[0, 0].sum()
    assert n[10, 10].sum() > 0.6 * d[10, 10].sum()  # flicker is within 0.8..1.2
    assert n[10, 10, 0] > 150


def test_outside_world_is_rock_and_grid(small):
    cfg = small.replace(day_length=240)
    s = flat_state(cfg, veg=0, t=120)  # noon: no night dimming
    img = render(s, (-5, -5, 10, 10), cfg, ppc=4, grid_lines=True)
    assert tuple(img[1, 1]) == P.ROCK and tuple(img[21, 21]) == P.BARE
    assert tuple(img[3, 1]) == P.GROUND and tuple(img[1, 3]) == P.GROUND  # forced 1 px gap
    assert not (render(s, (0, 0, 4, 4), cfg, ppc=4) == P.GROUND).all(-1).any()  # no gap < 6 px
    for ppc, gap in ((5, 0), (6, 1), (12, 1), (17, 2), (40, 4)):
        assert gap_px(ppc) == gap
    big = render(s, (0, 0, 2, 2), cfg, ppc=17)  # gap 2: one pixel on each side
    gap_rows = (big == P.GROUND).all(-1).all(1)
    assert gap_rows.tolist() == [True] + [False] * 15 + [True, True] + [False] * 15 + [True]


def test_palette_matches_style_table():
    hexes = {
        "BARE": "#1E2926", "ROCK": "#505A56", "WATER": "#1D4A63", "WATER_VARIANT": "#22556F",
        "EMBER": "#7C4F29", "ASH_FRESH": "#303A37",
    }  # fmt: skip
    for name, h in hexes.items():
        assert getattr(P, name) == P.hex_rgb(h), name
    assert P.ASH[1:] == (P.hex_rgb("#303A37"), P.hex_rgb("#27322F"), P.hex_rgb("#1E2926"))
    veg = ["#1C3933", "#21453E", "#27534B", "#2C6057"]
    dark = ["#18302B", "#1C3933", "#20433C", "#244C45"]
    assert list(P.VEG[1:]) == [P.hex_rgb(h) for h in veg]
    assert list(P.VEG_VARIANT[1:]) == [P.hex_rgb(h) for h in dark]
    lo = P.mix(P.FIRE, P.INK, P.FLAME_BASE - P.FLAME_AMP)
    hi = P.mix(P.FIRE, P.INK, P.FLAME_BASE + P.FLAME_AMP)
    assert lo == P.hex_rgb("#E8833A") and hi == P.hex_rgb("#E8995D")


def test_flame_ember_and_variants(small):
    cfg = small.replace(day_length=240)
    s = flat_state(cfg, veg=4, t=0)  # midnight
    s.fire[:, :32] = cfg.fire_max  # bucket 3: flame
    s.fire[:, 32:] = 1  # bucket 1: ember
    img = render(s, (0, 0, 64, 64), cfg, ppc=1).reshape(-1, 3).astype(int)
    flame, ember = img.reshape(64, 64, 3)[:, :32], img.reshape(64, 64, 3)[:, 32:]
    lo, hi = np.array(P.hex_rgb("#E8833A")), np.array(P.hex_rgb("#E8995D"))
    assert ((flame >= lo) & (flame <= hi)).all() and len(
        np.unique(flame.reshape(-1, 3), axis=0)
    ) > 3
    assert (ember == P.EMBER).all()  # burning cells are not dimmed at night
    s.fire[:] = 0
    day = flat_state(cfg, veg=2, t=120)
    cells = render(day, (0, 0, 64, 64), cfg, ppc=1).reshape(-1, 3)
    share = (cells == P.VEG_VARIANT[2]).all(-1).mean()
    assert (
        abs(share - 0.5) < 0.05
        and ((cells == P.VEG[2]).all(-1) | (cells == P.VEG_VARIANT[2]).all(-1)).all()
    )


def test_game_smoke(small, tmp_path, monkeypatch):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from wildfire.app import play

    monkeypatch.setattr(play, "SAVE_PATH", tmp_path / "save.npz")
    g = play.Game(small, seed=1)
    key = lambda k, mod=0: pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod)  # noqa: E731
    x0 = int(g.state.ax)
    g.handle(key(pygame.K_d))
    g.handle(pygame.event.Event(pygame.KEYUP, key=pygame.K_d))
    g.sim_step()
    assert int(g.state.ax) in (x0, x0 + 1) and int(g.state.facing) == 1
    g.handle(key(pygame.K_F5))
    saved = g.state.copy()
    for k in (pygame.K_TAB, pygame.K_g, pygame.K_i, pygame.K_m, pygame.K_EQUALS, pygame.K_p):
        g.handle(key(k))
    g.handle(key(pygame.K_n))  # single step while paused
    assert int(g.state.t) == int(saved.t) + 1
    g.handle(key(pygame.K_SPACE))
    assert g.next_action() == A.SPRAY and g.next_action() == A.NOOP
    g.handle(key(pygame.K_F9))
    assert int(g.state.t) == int(saved.t)
    g.handle(key(pygame.K_r, pygame.KMOD_SHIFT))
    assert g.seed == 1
    monkeypatch.chdir(tmp_path)
    g.handle(key(pygame.K_F2))  # start recording mid-game
    for _ in range(3):
        g.sim_step()
    g.handle(key(pygame.K_F2))
    from wildfire.record import replay

    (ep,) = (tmp_path / "recordings").iterdir()
    assert replay(ep)["steps"] == 3
    g.handle(key(pygame.K_k))  # fork preview
    assert g.god and g.heat.shape == g.state.shape and 0 <= g.heat.min() <= g.heat.max() <= 1
    g.draw()
    g.god = False
    g.overlay = True
    g.draw()
    pygame.quit()
