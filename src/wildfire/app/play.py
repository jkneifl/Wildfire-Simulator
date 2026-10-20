"""pygame-ce frontend (SPEC 6): ``python -m wildfire.app.play``."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pygame

from .. import actions as A
from ..config import Config
from ..dynamics import light, step
from ..metrics import global_stats
from ..observe import agent_window, crop, grid
from ..render import palette as P
from ..render.topdown import colors, flicker
from ..state import WorldState
from ..terrain import initial_state

MAP_W, MAP_H, PANEL_W = 800, 800, 300
AGENT_ZOOM = 12  # px per cell in agent view
SPEEDS = (2, 5, 10, 20, 40)
MINIMAP = 180
SAVE_PATH = Path("saves/quicksave.npz")
MOVE_KEYS = {
    pygame.K_UP: A.MOVE_N, pygame.K_w: A.MOVE_N,
    pygame.K_RIGHT: A.MOVE_E, pygame.K_d: A.MOVE_E,
    pygame.K_DOWN: A.MOVE_S, pygame.K_s: A.MOVE_S,
    pygame.K_LEFT: A.MOVE_W, pygame.K_a: A.MOVE_W,
}  # fmt: skip


def to_surface(img: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))


class Game:
    def __init__(self, cfg: Config, seed: int = 0):
        pygame.init()
        pygame.display.set_caption("Wildfire")
        self.screen = pygame.display.set_mode((MAP_W + PANEL_W, MAP_H))
        self.font = pygame.font.Font(None, 20)
        self.cfg = cfg
        self.speed = SPEEDS.index(10)
        self.paused = False
        self.god = False
        self.overlay = False
        self.minimap = True
        self.inspector = False
        self.running = True
        self.held: list[int] = []
        self.pending: int | None = None
        self.acc = 0.0
        self.step_times: list[float] = []
        self.recorder = None
        self.heat: np.ndarray | None = None  # fork preview, (H, W) burn fraction
        self.message = ""
        self.reset(seed)

    # --- simulation -------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.seed = seed
        self.set_state(initial_state(self.cfg, seed))
        h, w = self.cfg.height, self.cfg.width
        self.zoom = min(MAP_W / w, MAP_H / h)
        self.cam = [w / 2, h / 2]

    def set_state(self, s: WorldState) -> None:
        self.state = s
        self.memory = np.zeros((*s.shape, 6), np.uint8)
        self.seen = np.zeros(s.shape, bool)
        self.heat = None
        self.remember()

    def remember(self) -> None:
        """Store the observed window in the last-seen memory."""
        y0, x0, h, w = agent_window(self.state, self.cfg)
        H, W = self.state.shape
        ya, yb, xa, xb = max(y0, 0), min(y0 + h, H), max(x0, 0), min(x0 + w, W)
        g = grid(self.state, y0, x0, h, w, self.cfg)
        self.memory[ya:yb, xa:xb] = g[ya - y0 : yb - y0, xa - x0 : xb - x0]
        self.seen[ya:yb, xa:xb] = True

    def next_action(self) -> int:
        if self.pending is not None:
            a, self.pending = self.pending, None
            return a
        return MOVE_KEYS[self.held[-1]] if self.held else A.NOOP

    def sim_step(self) -> None:
        a = self.next_action()
        self.state = step(self.state, a, self.cfg)
        self.remember()
        if self.recorder is not None:
            self.recorder.add(a, self.state)
        now = time.perf_counter()
        self.step_times = [t for t in self.step_times if now - t < 1.0] + [now]

    # --- input ------------------------------------------------------------------------
    def handle(self, e: pygame.event.Event) -> None:
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            self.keydown(e)
        elif e.type == pygame.KEYUP and e.key in self.held:
            self.held.remove(e.key)
        elif e.type == pygame.MOUSEWHEEL and self.god:
            self.zoom_at(pygame.mouse.get_pos(), 1.25**e.y)
        elif e.type == pygame.MOUSEMOTION and self.god and (e.buttons[1] or e.buttons[2]):
            self.cam[0] -= e.rel[0] / self.zoom
            self.cam[1] -= e.rel[1] / self.zoom

    def keydown(self, e: pygame.event.Event) -> None:
        k, shift = e.key, e.mod & pygame.KMOD_SHIFT
        if k in MOVE_KEYS:
            self.pending = MOVE_KEYS[k]
            if k in self.held:
                self.held.remove(k)
            self.held.append(k)
        elif k == pygame.K_f:
            self.pending = A.IGNITE
        elif k == pygame.K_SPACE:
            self.pending = A.SPRAY
        elif k == pygame.K_TAB:
            self.god = not self.god
        elif k == pygame.K_g:
            self.overlay = not self.overlay
        elif k == pygame.K_m:
            self.minimap = not self.minimap
        elif k == pygame.K_i:
            self.inspector = not self.inspector
        elif k == pygame.K_p:
            self.paused = not self.paused
        elif k == pygame.K_n and self.paused:
            self.sim_step()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(self.speed + 1, len(SPEEDS) - 1)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(self.speed - 1, 0)
        elif k == pygame.K_r:
            self.stop_recording()
            self.reset(self.seed if shift else self.seed + 1)
        elif k == pygame.K_F5:
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.state.save(SAVE_PATH)
            self.message = f"saved {SAVE_PATH}"
        elif k == pygame.K_F9:
            self.load(SAVE_PATH)
        elif k == pygame.K_F2:
            self.toggle_recording()
        elif k == pygame.K_k:
            self.fork_preview()
        elif k == pygame.K_ESCAPE:
            self.running = False

    def load(self, path: Path) -> None:
        if not path.exists():
            self.message = f"no save at {path}"
            return
        s = WorldState.load(path)
        if s.shape != (self.cfg.height, self.cfg.width):
            self.message = f"save has shape {s.shape}, config {self.cfg.height}x{self.cfg.width}"
            return
        self.stop_recording()
        self.seed = int(s.seed)
        self.set_state(s)
        self.message = f"loaded {path}"

    def toggle_recording(self) -> None:
        self.message = "recording needs wildfire.record (M3)"

    def stop_recording(self) -> None:
        pass

    def fork_preview(self) -> None:
        self.message = "fork preview needs wildfire.metrics.burn_probability (M5)"

    def zoom_at(self, pos, factor: float) -> None:
        h, w = self.state.shape
        fit = min(MAP_W / w, MAP_H / h)
        new = float(np.clip(self.zoom * factor, fit * 0.5, 40.0))
        ox, oy = self.origin()
        cx, cy = ox + pos[0] / self.zoom, oy + pos[1] / self.zoom  # cell under the mouse
        self.cam = [cx - pos[0] / new + MAP_W / (2 * new), cy - pos[1] / new + MAP_H / (2 * new)]
        self.zoom = new

    # --- drawing ----------------------------------------------------------------------
    def view_zoom(self) -> float:
        return self.zoom if self.god else AGENT_ZOOM

    def origin(self) -> tuple[float, float]:
        """Top-left of the map area in cell coordinates."""
        z = self.view_zoom()
        if self.god:
            cx, cy = self.cam
        else:
            cx, cy = int(self.state.ax) + 0.5, int(self.state.ay) + 0.5
        return cx - MAP_W / (2 * z), cy - MAP_H / (2 * z)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.origin()
        z = self.view_zoom()
        return (x - ox) * z, (y - oy) * z

    def draw_map(self) -> None:
        s, cfg = self.state, self.cfg
        z = self.view_zoom()
        ox, oy = self.origin()
        x0, y0 = math.floor(ox), math.floor(oy)
        w, h = math.ceil(MAP_W / z) + 2, math.ceil(MAP_H / z) + 2
        L = float(light(s.t, cfg.day_length))
        c = colors(grid(s, y0, x0, h, w, cfg), L, flicker(s, y0, x0, h, w))
        if not self.god:
            wy, wx, wh, ww = agent_window(s, cfg)
            yy, xx = np.indices((h, w))
            inside = (yy + y0 >= wy) & (yy + y0 < wy + wh) & (xx + x0 >= wx) & (xx + x0 < wx + ww)
            outside = np.broadcast_to(np.array(P.FOG, np.float32), c.shape)
            if self.overlay:
                mem = np.stack([crop(self.memory[..., i], y0, x0, h, w) for i in range(6)], -1)
                m = colors(mem, 1.0)
                gray = m.mean(-1, keepdims=True)
                m = 0.55 * (0.35 * m + 0.65 * gray)  # desaturated and dimmed
                seen = crop(self.seen, y0, x0, h, w)[..., None]
                outside = np.where(seen, m, outside)
            c = np.where(inside[..., None], c, outside)
        if self.god and self.heat is not None:
            heat = crop(self.heat, y0, x0, h, w)[..., None]
            c = c * (1 - 0.8 * heat) + np.array((255, 0, 255), np.float32) * 0.8 * heat
        surf = to_surface(np.clip(c, 0, 255).astype(np.uint8))
        surf = pygame.transform.scale(surf, (round(w * z), round(h * z)))
        self.screen.blit(surf, ((x0 - ox) * z, (y0 - oy) * z))

        # agent window outline (god view) and agent marker
        wy, wx, wh, ww = agent_window(s, cfg)
        if self.god:
            px, py = self.to_screen(wx, wy)
            pygame.draw.rect(self.screen, (255, 255, 255), (px, py, ww * z, wh * z), 1)
        ax, ay = self.to_screen(int(s.ax) + 0.5, int(s.ay) + 0.5)
        r = max(z * 0.45, 3)
        dy, dx = A.DY[int(s.facing)], A.DX[int(s.facing)]
        tip = (ax + dx * r * 1.6, ay + dy * r * 1.6)
        pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), r)
        pygame.draw.circle(self.screen, (20, 20, 20), (ax, ay), r, 1)
        pygame.draw.line(self.screen, (255, 255, 255), (ax, ay), tip, 2)

    def draw_minimap(self) -> None:
        s, cfg = self.state, self.cfg
        H, W = s.shape
        g = grid(s, 0, 0, H, W, cfg)
        c = colors(g, float(light(s.t, cfg.day_length)))
        scale = MINIMAP / max(H, W)
        surf = pygame.transform.scale(
            to_surface(c.astype(np.uint8)), (round(W * scale), round(H * scale))
        )
        x, y = MAP_W - surf.get_width() - 8, MAP_H - surf.get_height() - 8
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, (230, 230, 230), (x - 1, y - 1, *surf.get_size()), 1)
        wy, wx, wh, ww = agent_window(s, cfg)
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (x + wx * scale, y + wy * scale, max(ww * scale, 2), max(wh * scale, 2)),
            1,
        )

    def hovered_cell(self) -> tuple[int, int] | None:
        mx, my = pygame.mouse.get_pos()
        if mx >= MAP_W:
            return None
        ox, oy = self.origin()
        z = self.view_zoom()
        return math.floor(ox + mx / z), math.floor(oy + my / z)

    def hud_lines(self) -> list[str]:
        s, cfg = self.state, self.cfg
        t, D = int(s.t), cfg.day_length
        minutes = round((t % D) / D * 24 * 60)
        st = global_stats(s)
        tank = "inf" if cfg.infinite_water else f"{int(s.tank)}/{cfg.tank}"
        lines = [
            f"Day {t // D + 1}   {minutes // 60:02d}:{minutes % 60:02d}",
            "",  # light bar
            f"step {t}",
            f"speed {SPEEDS[self.speed]} steps/s  (actual {len(self.step_times)})",
            f"active fires {st['active_fires']}",
            f"burned {100 * st['burned_frac']:.1f} %   veg {100 * st['veg_density']:.1f} %",
            f"tank {tank}",
            f"seed {self.seed}  stream {int(s.stream)}",
            f"view {'god' if self.god else 'agent'}   overlay {'on' if self.overlay else 'off'}",
        ]
        if self.paused:
            lines.append("PAUSED  (N: single step)")
        if self.recorder is not None:
            lines.append(f"REC {self.recorder.n_steps} steps")
        if self.heat is not None:
            lines.append("fork preview shown (god view)")
        if self.inspector:
            lines += ["", "-- inspector --", *self.inspect()]
        if self.message:
            lines += ["", self.message]
        lines += [
            "",
            "WASD/arrows move  F ignite  Space water",
            "Tab view  G last-seen  M minimap",
            "I inspector  P pause  N step  +/- speed",
            "R new seed  Shift+R same seed",
            "F5 save  F9 load  F2 record  K forks",
        ]
        return lines

    def inspect(self) -> list[str]:
        cell = self.hovered_cell()
        s = self.state
        if cell is None:
            return ["(hover a map cell)"]
        x, y = cell
        H, W = s.shape
        if not (0 <= x < W and 0 <= y < H):
            return [f"x={x} y={y}: outside (rock)"]
        g = grid(s, y, x, 1, 1, self.cfg)[0, 0]
        names = ("ground", "water", "rock")
        return [
            f"x={x} y={y}  {names[int(s.terrain[y, x])]}",
            f"veg {int(s.veg[y, x])}",
            f"fire {int(s.fire[y, x])}  (bucket {g[3]})",
            f"ash_age {int(s.ash_age[y, x])}  (bucket {g[5]})",
            f"wet {int(s.wet[y, x])}  (bucket {g[4]})",
            f"last seen: {'yes' if self.seen[y, x] else 'never'}",
        ]

    def draw_panel(self) -> None:
        pygame.draw.rect(self.screen, (28, 28, 34), (MAP_W, 0, PANEL_W, MAP_H))
        y = 10
        L = float(light(self.state.t, self.cfg.day_length))
        for i, line in enumerate(self.hud_lines()):
            if i == 1:  # light bar after the clock
                pygame.draw.rect(self.screen, (70, 70, 80), (MAP_W + 10, y + 2, 200, 10))
                pygame.draw.rect(self.screen, (250, 220, 90), (MAP_W + 10, y + 2, 200 * L, 10))
                self.screen.blit(
                    self.font.render(f"light {L:.2f}", True, (220, 220, 220)), (MAP_W + 218, y)
                )
            else:
                self.screen.blit(self.font.render(line, True, (220, 220, 220)), (MAP_W + 10, y))
            y += 20

    def draw(self) -> None:
        self.screen.fill(P.FOG)
        self.draw_map()
        if self.minimap:
            self.draw_minimap()
        self.draw_panel()
        pygame.display.flip()

    # --- loop -------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.paused:
            self.acc = 0.0
            return
        period = 1.0 / SPEEDS[self.speed]
        self.acc = min(self.acc + dt, 5 * period)  # drop steps rather than spiral
        while self.acc >= period:
            self.acc -= period
            self.sim_step()

    def run(self, frames: int | None = None, screenshot: str | None = None) -> None:
        clock = pygame.time.Clock()
        n = 0
        while self.running and (frames is None or n < frames):
            dt = clock.tick(60) / 1000
            for e in pygame.event.get():
                self.handle(e)
            self.update(dt)
            self.draw()
            n += 1
        if screenshot:
            pygame.image.save(self.screen, screenshot)
        self.stop_recording()
        pygame.quit()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Wildfire game")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=None, help="quit after N frames (testing)")
    p.add_argument("--screenshot", default=None, help="save the last frame to this file")
    args = p.parse_args(argv)
    Game(Config.from_yaml(args.config), args.seed).run(args.frames, args.screenshot)


if __name__ == "__main__":
    main()
