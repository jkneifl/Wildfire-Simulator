"""Training samplers over recorded episodes (NumPy only).

Every sample is a dict ``{"input", "target", "probe", "meta"}``:

- ``input`` / ``target``: observable-channel state only (``observe.grid`` channels, uint8, or
  their level-of-detail fractions), plus day phase and light. Ground-truth observable state
  outside the agent's window is used as model state (teacher forcing); ``seen`` masks mark
  which cells the agent actually saw at that step, for observation-limited variants.
- ``probe``: privileged quantities (global statistics, event causes) for analysis only.
  Never feed them to a model.

Samples are drawn on the fly from the ``world/`` grids of episodes recorded with
``Products(world_every=...)``; ``sample(i)`` is a pure function of the sampler's seed and ``i``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .dynamics import N8, light
from .events import AGENT, LIGHTNING, SPREAD
from .levels import clipmap, details, one_hot, restrict
from .record import GLOBAL_COLUMNS, GLOBAL_PROBES, EpisodeDataset, load_config
from .rng import Purpose, uniform
from .state import ROCK
from .tiles import absences, visibility

RING_SENTINEL = 255  # value of the target tile's own cells inside a ring input
_PHASE, _LIGHT = GLOBAL_COLUMNS.index("phase"), GLOBAL_COLUMNS.index("light")
_PROBES = [GLOBAL_COLUMNS.index(k) for k in GLOBAL_PROBES]


class Episodes:
    """Index over episodes with full-world grids; loads and caches per-episode arrays."""

    def __init__(self, root: str | Path):
        self.ds = EpisodeDataset(root)
        self.eps = []
        for i in range(len(self.ds)):
            cfg, meta = load_config(self.ds.dirs[i])
            we = meta.get("products", {}).get("world_every")
            if not we:
                continue
            ep = self.ds[i]
            self.eps.append(
                {
                    "i": i,
                    "cfg": cfg,
                    "world_every": we,
                    "n_steps": meta["n_steps"],
                    "start": meta["start_step"],
                    "agent": np.asarray(ep["agent"]),
                    "actions": np.asarray(ep["actions"]),
                    "globals": np.asarray(ep["globals"]) if "globals" in ep else None,
                }
            )
        if not self.eps:
            raise ValueError(f"no episodes with world/ grids under {root}")

    def __len__(self) -> int:
        return len(self.eps)

    @lru_cache(maxsize=64)  # noqa: B019 - per-instance cache of decompressed grids
    def world(self, e: int, t: int) -> np.ndarray:
        return self.ds.world(self.eps[e]["i"], t)

    def events(self, e: int) -> dict:
        return self.ds.events(self.eps[e]["i"])

    def clock(self, e: int, t: int) -> dict:
        ep = self.eps[e]
        if ep["globals"] is not None:
            row = ep["globals"][t]
            return {"phase": float(row[_PHASE]), "light": float(row[_LIGHT])}
        cfg, tt = ep["cfg"], ep["start"] + t
        return {
            "phase": (tt % cfg.day_length) / cfg.day_length,
            "light": float(light(tt, cfg.day_length)),
        }

    def seen(self, e: int, t: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """Cells of a window that were inside the agent's view at step ``t``."""
        ep = self.eps[e]
        r = ep["cfg"].window // 2
        ax, ay = ep["agent"][t, 0], ep["agent"][t, 1]
        yy, xx = np.arange(y0, y0 + h)[:, None], np.arange(x0, x0 + w)[None, :]
        return (abs(yy - ay) <= r) & (abs(xx - ax) <= r)


def _crop(g: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Crop of a ``(H, W, 6)`` grid; outside the world is rock (as ``observe.crop``)."""
    out = np.zeros((h, w, 6), np.uint8)
    out[..., 0] = ROCK
    H, W = g.shape[:2]
    ya, yb, xa, xb = max(y0, 0), min(y0 + h, H), max(x0, 0), min(x0 + w, W)
    if ya < yb and xa < xb:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = g[ya:yb, xa:xb]
    return out


class _Sampler:
    stream = 0

    def __init__(self, root: str | Path, seed: int = 0):
        self.episodes = root if isinstance(root, Episodes) else Episodes(root)
        self.seed = seed

    def _u(self, i: int, n: int) -> np.ndarray:
        return uniform(self.seed, self.stream, i, Purpose.SAMPLER, np.zeros(n, int), np.arange(n))

    def _steps(self, e: int) -> np.ndarray:
        ep = self.episodes.eps[e]
        return np.arange(0, ep["n_steps"] + 1, ep["world_every"])


class DynamicsSampler(_Sampler):
    """``F``: levels around the agent at ``t``, action and clock -> the same regions at ``t+dt``.

    The target uses the input's level origins; ``target["center"]`` is the agent position at
    ``t + dt`` for re-centring. ``dt`` must be a multiple of the episodes' ``world_every``.
    """

    stream = 1

    def __init__(self, root, n: int = 32, levels: int = 3, dt: int = 1, seed: int = 0):
        super().__init__(root, seed)
        self.n, self.levels, self.dt = n, levels, dt

    def sample(self, i: int) -> dict:
        E = self.episodes
        u = self._u(i, 2)
        e = int(u[0] * len(E))
        ep = E.eps[e]
        if self.dt % ep["world_every"]:
            raise ValueError(f"dt={self.dt} is not a multiple of world_every={ep['world_every']}")
        steps = self._steps(e)
        steps = steps[steps + self.dt <= ep["n_steps"]]
        t = int(steps[int(u[1] * len(steps))])
        ax, ay = ep["agent"][t, :2]
        H, W = ep["cfg"].height, ep["cfg"].width
        lv0 = clipmap(E.world(e, t), (ay, ax), self.n, self.levels)
        lv1 = []
        for lv in lv0:  # same regions at t + dt
            oy, ox = lv["origin"]
            s, side = lv["cell"], self.n * lv["cell"]
            z = one_hot(_crop(E.world(e, t + self.dt), oy, ox, side, side))
            lv1.append(restrict(z, s) if s > 1 else z)
        return {
            "input": {
                "levels": [lv["z"] for lv in lv0],
                "origins": [lv["origin"] for lv in lv0],
                "actions": ep["actions"][t : t + self.dt].copy(),
                **E.clock(e, t),
                "seen": E.seen(e, t, *lv0[0]["origin"], self.n, self.n),
            },
            "target": {
                "levels": lv1,
                "center": (int(ep["agent"][t + self.dt, 1]), int(ep["agent"][t + self.dt, 0])),
            },
            "probe": self._probe(e, t, t + self.dt),
            "meta": {
                "e": e,
                "episode": str(E.ds.dirs[ep["i"]]),
                "t": t,
                "dt": self.dt,
                "world": (H, W),
            },
        }

    def _probe(self, e: int, t0: int, t1: int) -> dict:
        g = self.episodes.eps[e]["globals"]
        if g is None:
            return {}
        return {
            "globals_t0": g[t0, _PROBES].copy(),
            "globals_t1": g[t1, _PROBES].copy(),
            "names": GLOBAL_PROBES,
        }


class TileSampler(_Sampler):
    """Base for tile pairs: re-entries from real absences or synthetic gaps."""

    def __init__(
        self,
        root,
        tile: int = 16,
        ring: int = 8,
        source: str = "synthetic",
        radius: int | None = None,
        dt_range: tuple[int, int] = (1, 2048),
        seed: int = 0,
    ):
        super().__init__(root, seed)
        if source not in ("synthetic", "revisits"):
            raise ValueError("source must be 'synthetic' or 'revisits'")
        self.T, self.ring, self.source, self.dt_range = tile, ring, source, dt_range
        self.radius = radius
        if source == "revisits":
            self.revisits = self._revisits()

    def _revisits(self) -> list[np.ndarray]:
        """Per episode rows ``(ty, tx, exit, reentry)`` at steps with world grids."""
        out = []
        for ep in self.episodes.eps:
            cfg = ep["cfg"]
            r = cfg.window // 2 if self.radius is None else self.radius
            a = absences(visibility(ep["agent"], r, self.T, (cfg.height, cfg.width)))
            we = ep["world_every"]
            ok = (a[:, 3] >= 0) & (a[:, 2] % we == 0) & (a[:, 3] % we == 0)
            out.append(a[ok])
        if not any(len(x) for x in out):
            raise ValueError("no completed absences at steps with world grids")
        return out

    def _pick(self, i: int):
        """(episode, ty, tx, tau, t) for sample ``i``."""
        E = self.episodes
        u = self._u(i, 5)
        if self.source == "revisits":
            pool = [e for e in range(len(E)) if len(self.revisits[e])]
            e = pool[int(u[0] * len(pool))]
            ty, tx, tau, t = self.revisits[e][int(u[1] * len(self.revisits[e]))]
            return e, int(ty), int(tx), int(tau), int(t)
        e = int(u[0] * len(E))
        ep = E.eps[e]
        cfg, we = ep["cfg"], ep["world_every"]
        rows, cols = -(-cfg.height // self.T), -(-cfg.width // self.T)
        ty, tx = int(u[1] * rows), int(u[2] * cols)
        lo, hi = self.dt_range[0], min(self.dt_range[1], ep["n_steps"] // we * we)
        if hi <= 0:
            dt = 0
        else:  # log-uniform on [max(lo, 1), hi], rounded to a multiple of world_every
            a = np.log(max(lo, 1))
            dt = int(round(np.exp(a + u[3] * (np.log(hi) - a)) / we)) * we
            dt = int(np.clip(dt, -(-lo // we) * we, hi))
        steps = self._steps(e)
        steps = steps[steps + dt <= ep["n_steps"]]
        tau = int(steps[int(u[4] * len(steps))])
        return e, ty, tx, tau, tau + dt

    def tile_grid(self, e: int, t: int, ty: int, tx: int) -> np.ndarray:
        return _crop(self.episodes.world(e, t), ty * self.T, tx * self.T, self.T, self.T)

    def ring_grid(self, e: int, t: int, ty: int, tx: int) -> tuple[np.ndarray, tuple[int, int]]:
        """Neighbourhood of width ``ring`` around the tile at ``t``, the tile itself masked."""
        w, T = self.ring, self.T
        y0, x0 = ty * T - w, tx * T - w
        g = _crop(self.episodes.world(e, t), y0, x0, T + 2 * w, T + 2 * w)
        g[w : w + T, w : w + T] = RING_SENTINEL
        return g, (y0, x0)

    def tile_probe(self, e: int, ty: int, tx: int, tau: int, t: int) -> dict:
        ep = self.episodes.eps[e]
        out = {}
        if ep["globals"] is not None:
            g = ep["globals"]
            out["names"] = GLOBAL_PROBES
            out["globals_tau"] = g[tau, _PROBES].copy()
            out["globals_t"] = g[t, _PROBES].copy()
            out["globals_mean"] = g[tau : t + 1, _PROBES].mean(0)
        if (self.episodes.ds.dirs[ep["i"]] / "events.npz").exists():
            ev = self.episodes.events(e)
            T = self.T
            m = (ev["step"] > tau) & (ev["step"] <= t)
            m &= (ev["y"] // T == ty) & (ev["x"] // T == tx)
            cause = ev["cause"][m]
            # spread with a burning neighbour outside the tile: fire crossing into the tile
            y, x, nbrs = ev["y"][m], ev["x"][m], ev["nbrs"][m]
            crossed = np.zeros(y.size, bool)
            for bit, (dy, dx, _) in enumerate(N8):
                burning_nb = ((nbrs >> bit) & 1).astype(bool)
                crossed |= burning_nb & (((y + dy) // T != ty) | ((x + dx) // T != tx))
            out["ignitions"] = {
                "agent": int((cause == AGENT).sum()),
                "lightning": int((cause == LIGHTNING).sum()),
                "spread": int((cause == SPREAD).sum()),
                "crossing": int((crossed & (cause == SPREAD)).sum()),
            }
        return out


class ReentrySampler(TileSampler):
    """``E`` on re-entry: ``m_i(tau)``, ring at ``t``, clocks and ``dt`` -> ``m_i(t)``."""

    stream = 2

    def sample(self, i: int) -> dict:
        E = self.episodes
        e, ty, tx, tau, t = self._pick(i)
        ep = E.eps[e]
        ring, (y0, x0) = self.ring_grid(e, t, ty, tx)
        g = ep["globals"]
        return {
            "input": {
                "tile_tau": self.tile_grid(e, tau, ty, tx),
                "ring_t": ring,
                "dt": t - tau,
                "clock_tau": E.clock(e, tau),
                "clock_t": E.clock(e, t),
                "phase_series": None if g is None else g[tau : t + 1, _PHASE].copy(),
                "seen_ring_t": E.seen(e, t, y0, x0, *ring.shape[:2]),
            },
            "target": {"tile_t": self.tile_grid(e, t, ty, tx)},
            "probe": self.tile_probe(e, ty, tx, tau, t),
            "meta": {
                "e": e,
                "episode": str(E.ds.dirs[ep["i"]]),
                "tile": (ty, tx),
                "tau": tau,
                "t": t,
                "source": self.source,
            },  # fmt: skip
        }


class RefinementSampler(TileSampler):
    """``E`` on refinement: details of the tile at ``tau`` and the coarse tile at ``t`` -> details
    at ``t`` (category fractions; ``factor`` is the coarsening factor)."""

    stream = 3

    def __init__(self, root, factor: int = 2, **kw):
        super().__init__(root, **kw)
        self.factor = factor

    def sample(self, i: int) -> dict:
        E = self.episodes
        e, ty, tx, tau, t = self._pick(i)
        z_tau = one_hot(self.tile_grid(e, tau, ty, tx))
        z_t = one_hot(self.tile_grid(e, t, ty, tx))
        return {
            "input": {
                "details_tau": details(z_tau, self.factor),
                "coarse_t": restrict(z_t, self.factor),
                "dt": t - tau,
                "clock_tau": E.clock(e, tau),
                "clock_t": E.clock(e, t),
            },
            "target": {"details_t": details(z_t, self.factor)},
            "probe": self.tile_probe(e, ty, tx, tau, t),
            "meta": {"e": e, "tile": (ty, tx), "tau": tau, "t": t, "factor": self.factor},
        }


class TripleSampler(TileSampler):
    """Self-consistency: the tile at ``tau``, ``tau+dt``, ``tau+2dt``, rings at the later two."""

    stream = 4

    def sample(self, i: int) -> dict:
        E = self.episodes
        e, ty, tx, tau, t = self._pick(i)
        ep = E.eps[e]
        we = ep["world_every"]
        dt = max((t - tau) // 2 // we * we, we)
        tau = max(min(tau, ep["n_steps"] - 2 * dt) // we * we, 0)  # stay on world-grid steps
        t1, t2 = tau + dt, tau + 2 * dt
        return {
            "input": {
                "tile_tau": self.tile_grid(e, tau, ty, tx),
                "ring_t1": self.ring_grid(e, t1, ty, tx)[0],
                "ring_t2": self.ring_grid(e, t2, ty, tx)[0],
                "dt": dt,
                "clocks": [E.clock(e, tau), E.clock(e, t1), E.clock(e, t2)],
            },
            "target": {
                "tile_t1": self.tile_grid(e, t1, ty, tx),
                "tile_t2": self.tile_grid(e, t2, ty, tx),
            },
            "probe": self.tile_probe(e, ty, tx, tau, t2),
            "meta": {"e": e, "tile": (ty, tx), "tau": tau, "dt": dt},
        }
