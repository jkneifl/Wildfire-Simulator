import os

import numpy as np

from wildfire import actions as A
from wildfire.render.topdown import render

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


def test_outside_world_is_rock_and_grid_lines(small):
    s = flat_state(small, veg=0)
    img = render(s, (-5, -5, 10, 10), small, ppc=4, grid_lines=True)
    assert not np.array_equal(img[0, 1], img[20, 21])  # rock vs ground
    assert img[0, 0].sum() < img[1, 1].sum()  # grid line pixel is darker


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
