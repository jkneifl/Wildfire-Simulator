# Wildfire testbed: specification

## 1. Purpose

An interactive 2D wildfire game that also serves as a ground-truth simulator for experiments with partially observed, stochastic environments. It has two uses:

1. **Demo.** A person walks around a 2D map and can set fires and spray water.
2. **Ground truth for experiments.** The simulator provides the exact full-world state at every step, the agent's partial observations, reproducible stochastic dynamics, and the ability to rerun the future from any saved state with different random seeds.

The simulator updates every cell at every step and is the reference for evaluation.

## 2. Design requirements

Read these first. They constrain every module.

- **R1 Determinism.** A run is fully determined by (config, seed, action sequence). Replaying must reproduce every state exactly.
- **R2 Counter-based randomness.** Every random number is a pure function of `(seed, stream, step, purpose, y, x)`. There is no global RNG state. Results therefore do not depend on update order or on how the grid is split into chunks.
- **R3 Forking.** From any state, the simulator can run `n` futures that differ only in their `stream` id. This provides Monte Carlo samples of possible futures, which the evaluation needs because many outcomes are valid after an absence.
- **R4 Functional core.** `step(state, action, params) -> state` is a pure function on NumPy arrays. The state is a dataclass of arrays and scalars. The core must not import pygame. All per-cell work is vectorized; no Python loops over cells.
- **R5 Observable vs. privileged state.** Separate functions return what the agent can observe and what only the evaluation may use (exact timers, RNG ids). Agents must never receive privileged state.
- **R6 Headless and batched.** The simulator runs without a display. A batched variant steps `n` worlds of shape `(n, H, W)` in one call.
- **R7 Tiles.** The world is partitioned into square tiles. Utilities give tile indices, tile contents and, per tile, the last step at which the agent observed it.

## 3. Simulation rules

### 3.1 Grid and layers

Finite world of `H × W` cells (default 256 × 256; headless runs must handle 2048 × 2048). Cells outside the world behave as rock. Arrays are indexed `[y, x]`, with `x` to the right and `y` downwards; north is `y - 1`.

| Layer | dtype | Values | Observable |
|---|---|---|---|
| `terrain` | uint8 | 0 ground, 1 water, 2 rock; static | yes |
| `veg` | uint8 | 0 bare, 1..`V_MAX` growth stage (default `V_MAX = 4`) | yes |
| `fire` | uint8 | remaining burn steps, 0 = not burning | burning flag and intensity bucket (0..3) only |
| `ash_age` | uint16 | steps since the cell burned out, 0 = no ash; saturates | bucketed (0..3) only |
| `wet` | uint8 | wetness 0..255 | bucketed (0..3) |

Global scalars: step `t`, day length `D`. Agent: position `(x, y)`, facing direction, water tank level.

Water and rock cells never burn and never grow vegetation.

### 3.2 Time and day-night cycle

With phase $\varphi_t$ (0 at midnight) and light level $L_t$ (0 at midnight, 1 at noon):

$$
\begin{align}
\varphi_t &= \frac{t \bmod D}{D}, \\
L_t &= \tfrac{1}{2}\big(1 - \cos 2\pi \varphi_t\big).
\end{align}
$$

Default `D = 240` steps. The day-night cycle affects the dynamics, not only the rendering:

- vegetation grows only in daylight (probabilities scale with $L_t$);
- fire spreads more slowly at night (multiplier $m(L_t)$, see 3.3);
- wetness dries faster during the day.

This makes cyclic events (day-night) interact with directed events (fires, regrowth).

### 3.3 Update order per step

All rules read the state after the agent's action (step 1) and write a new state (synchronous update).

1. **Agent action** (see 3.4).
2. **Burning.** Cells with `fire > 0` decrement `fire`. A cell reaching 0 becomes ash: `veg = 0`, `ash_age = 1`.
3. **Spread.** A cell that is not burning, has `veg > 0` and `wet < WET_BLOCK` ignites with probability

$$
\begin{align}
p_\text{ign} &= 1 - \prod_{n \in \mathcal{N}_8,\ \text{burning}} \Big(1 - \beta\, w_n\, m(L_t)\, \frac{\text{veg}}{V_\text{max}}\, g(\text{wet})\Big), \\
m(L) &= m_\text{night} + (1 - m_\text{night})\, L, \qquad g(\text{wet}) = \max\!\Big(0,\ 1 - \frac{\text{wet}}{\text{WET\_BLOCK}}\Big),
\end{align}
$$

   with $w_n = 1$ for orthogonal and $1/\sqrt{2}$ for diagonal neighbours. An ignited cell gets `fire = BURN_BASE + BURN_PER_VEG * veg`.
4. **Lightning.** A cell with `veg > 0` that is not burning ignites spontaneously with probability `P_LIGHTNING`.
5. **Growth.** Cells with `1 <= veg < V_MAX` increase `veg` by 1 with probability `P_GROW * L_t`. Bare ground cells (`veg = 0`, `terrain = ground`, and `ash_age = 0` or `ash_age > REGROW_DELAY`) become `veg = 1` with probability `P_SEED * L_t * (1 + k)`, where `k` is the number of orthogonal neighbours with `veg > 0`.
6. **Drying.** `wet` decreases by `DRY_RATE * (0.5 + L_t)`, rounded stochastically, clipped at 0.
7. **Ash.** `ash_age` increments where it is positive (saturating).
8. `t += 1`.

Randomness purposes (for R2): `SPREAD`, `LIGHTNING`, `GROWTH`, `SEED`, `DRY`, `TERRAIN`, `FLICKER` (rendering only).

All constants live in the config. Starting defaults (to be calibrated, see M4): `beta = 0.35`, `m_night = 0.5`, `WET_BLOCK = 128`, `BURN_BASE = 3`, `BURN_PER_VEG = 2`, `P_LIGHTNING = 2e-7`, `P_GROW = 0.01`, `P_SEED = 0.002`, `REGROW_DELAY = 120`, `DRY_RATE = 2`.

**Target regime** for the defaults on 256 × 256: some fire is active most of the time, burned areas regrow over a few days, and the forest neither burns out completely nor stays fire-free for long. The reference model is the Drossel–Schwabl forest-fire model.

### 3.4 Initial state

- Terrain from the seed: smooth value noise thresholded into lakes and rock outcrops, plus one or two rivers. Roughly 8 % water, 4 % rock.
- Vegetation from a second noise field, so that density varies over the map (dense forest, sparse areas, clearings).
- No fires at `t = 0`. The start time is configurable (default: 06:00, i.e. `t = D / 4`).
- Agent on a random ground cell near the map centre.

### 3.5 Agent and actions

Discrete actions:

| id | action | effect |
|---|---|---|
| 0 | noop | – |
| 1–4 | move N, E, S, W | move one cell and set facing; blocked by water and rock |
| 5 | ignite | target = cell in facing direction; ignites if `veg > 0`, not burning and `wet < WET_BLOCK` |
| 6 | water | target = cell in facing direction; on the 3 × 3 block around it, set `wet = 255` and `fire = 0`; uses one unit of water |

Water tank: capacity `TANK = 20` (config). It refills completely when the agent stands next to a water cell. If the tank is empty, `water` does nothing. Can be disabled in the config (infinite water).

The agent can walk through burning cells (no health in v1).

## 4. Observations

**Agent window.** A square of `V × V` cells centred on the agent (default `V = 33`). Cells outside the world appear as rock.

`observe(state) -> dict` returns only observable information:

- `grid`: uint8 array `(V, V, C)` with channels `[terrain, veg, burning, fire_bucket, wet_bucket, ash_bucket]`;
- `rgb`: uint8 image of the window from the renderer (section 5), `P` pixels per cell (default 4);
- `light`: float, the light level $L_t$ (it is visible in the image anyway);
- `agent`: facing direction and tank level.

`privileged(state) -> dict` returns the full arrays, exact `fire` and `ash_age`, `t`, `seed`, `stream`. It is used only for ground truth and evaluation.

## 5. Rendering

The renderer is a pure NumPy function `render(state, window, params) -> uint8 image`. It does not import pygame and is used both by the game and by dataset generation.

- Palette defined once in `render/palette.py`: ground, water, rock, four vegetation shades, fire by intensity bucket, ash fading towards ground with `ash_age`, a blue tint for wetness.
- Fire flicker uses the counter-based RNG (`FLICKER` purpose), so frames are reproducible.
- Night: multiply RGB by `0.35 + 0.65 * L_t`, except burning cells, which stay bright. Fires should be clearly visible at night.
- Optional grid lines.

## 6. Interactive game

Frontend in pygame-ce, in `app/play.py`. The simulation runs at a fixed number of steps per second, independent of the render frame rate.

**Views** (toggle with Tab):

- *Agent view*: the camera follows the agent. Only the agent window is shown; everything outside it is fog.
- *God view*: the whole map, zoomable with the mouse wheel.

**Last-seen overlay** (G, agent view only): cells outside the window show their last observed state, desaturated. It shows a person playing what the agent last saw.

**HUD**: day number, clock (HH:MM), light bar, step, steps per second, active fires, burned fraction, tank level, seed, paused flag.

**Controls**

| key | action |
|---|---|
| arrows / WASD | move (repeats while held) |
| F | ignite |
| Space | water |
| Tab | agent view / god view |
| G | last-seen overlay |
| M | minimap |
| I | cell inspector: hover a cell to see all layers, including privileged ones |
| P | pause; N single step while paused |
| + / − | simulation speed (2, 5, 10, 20, 40 steps/s) |
| R / Shift+R | reset with a new seed / the same seed |
| F5 / F9 | save / load state |
| F2 | start or stop recording an episode |
| K | fork preview (M5): run 16 futures for 100 steps from the current state and show the fraction of futures in which each cell burned, as a heat map |

At most one action is applied per simulation step: the last key pressed during that step, with movement repeating while a key is held.

## 7. Environment API

```python
from wildfire import Config, WildfireEnv, VecWildfire

env = WildfireEnv(Config.from_yaml("configs/default.yaml"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)

state = env.get_state()  # full privileged state (copy)
env.set_state(state)
futures = env.fork(n=16, steps=200)  # Monte Carlo futures from the current state
```

- Gymnasium-compatible spaces. `reward = 0` by default; reward functions are pluggable for later use (for example, negative newly burned area for a firefighting task).
- `info` contains the tiles intersecting the window and the per-tile last-seen steps, not privileged state. Privileged state is accessed explicitly through `get_state()` or `privileged()`.
- `VecWildfire(cfg, n)` steps `n` worlds at once with arrays of shape `(n, H, W)`.

## 8. Tiles

Tile size `T = 16` (config). Utilities in `tiles.py`:

- tile index of a cell; cells of a tile; tiles intersecting a window;
- `extract_tile(state, tile)` for all layers;
- per-tile `first_seen` and `last_seen` (step at which any cell of the tile was in the agent window), maintained by the environment.

## 9. Recording, replay, datasets

An episode is a directory:

| file | content |
|---|---|
| `meta.json` | config, seed, stream, start step, policy name, git hash |
| `actions.npy` | int8 `(T,)` |
| `agent.npy` | `(T+1, 4)`: x, y, facing, tank |
| `obs_grid.npy` | uint8 `(T+1, V, V, C)` |
| `obs_rgb.npy` | uint8 `(T+1, V·P, V·P, 3)`, optional |
| `tiles_seen.npy` | per-tile first and last seen step |
| `state/step_XXXXXX.npz` | full privileged state every `K` steps (default 50), compressed |
| `preview.mp4` | optional, god view with agent window outlined |

- `wildfire replay EP_DIR` rebuilds the episode from `meta.json` and `actions.npy` and checks every stored snapshot.
- `wildfire generate --config ... --policy ... --episodes N --steps S --out DIR` generates episodes in parallel with multiprocessing.
- `EpisodeDataset` loads episodes with NumPy only. A PyTorch `Dataset` adapter is an optional extra.

## 10. Scripted policies

In `policies.py`, all deterministic given their own seed:

- `random_walk`
- `explore_return`: observes a region, leaves it for a sampled absence `Δt` (log-uniform, default 50–2000 steps), then returns. Logs `(tile, leave_step, return_step)`. This produces revisit data.
- `firestarter`: random walk that ignites with a small probability.
- `firefighter`: moves to the nearest visible fire and waters it; refills at water.

## 11. Evaluation utilities

In `metrics.py`:

- global statistics: vegetation density, burned (ash) fraction, active fires, mean wetness;
- patch statistics: size distribution of connected burned and vegetated components (`scipy.ndimage.label`);
- tile statistics: the same quantities for one tile;
- `monte_carlo(state, steps, n, tiles=None)`: fork `n` futures and return tile statistics at the final step, i.e. samples of the reference distribution;
- `retention_curve(...)`: correlation between a tile's state at `τ` and at `τ + Δt` as a function of `Δt`, estimated over many tiles and episodes;
- distances between sample sets: energy distance and MMD on tile-statistic vectors.

## 12. Project layout

```
wildfire/
  pyproject.toml
  README.md
  CLAUDE.md
  SPEC.md
  configs/            default.yaml, small.yaml (64 x 64), large.yaml (2048 x 2048)
  src/wildfire/
    config.py         dataclass config, YAML loading
    rng.py            counter-based random numbers
    terrain.py        procedural terrain and initial vegetation
    state.py          WorldState dataclass
    dynamics.py       pure step function
    actions.py
    observe.py        agent window, observable and privileged views
    tiles.py
    render/palette.py
    render/topdown.py
    env.py            WildfireEnv, VecWildfire
    record.py         recording, replay, EpisodeDataset
    policies.py
    metrics.py
    cli.py            generate, replay, calibrate
    app/play.py       pygame-ce frontend
  tests/
  scripts/
```

Stack: Python ≥ 3.11, NumPy, SciPy, PyYAML, pygame-ce, pytest, ruff; `uv` for environments. No GPU required. PyTorch only as an optional extra.

## 13. Milestones

Work in this order. After each milestone: tests pass, README updated, one commit.

**M1 Core simulation.** `config`, `rng`, `terrain`, `state`, `dynamics`, `actions`, and a headless CLI that runs `N` steps and prints statistics.
Tests:
- same config, seed and actions give identical states; replay reproduces snapshots;
- the same stream gives identical futures, different streams diverge;
- water and rock never burn or grow;
- water extinguishes fire and wet cells do not ignite;
- spread is slower at night than at noon (statistical test with fixed seeds);
- ignite and water respect facing direction, tank level and refill.
Performance: ≥ 200 steps/s for 256 × 256 on one CPU core; report the figure for 2048 × 2048.

**M2 Renderer and game.** `render`, `app/play.py`: both views, HUD, all controls except K, last-seen overlay, minimap, cell inspector, save and load. Smooth rendering at 10 steps/s on 256 × 256.

**M3 Environment and data.** `observe`, `tiles`, `env` (including `fork` and `VecWildfire`), `record`, `replay`, `generate`. Replay test over a 2000-step episode.

**M4 Policies and calibration.** All scripted policies. `scripts/calibrate.py` runs long headless simulations and plots vegetation density, burned fraction and active fires over several days, plus patch-size distributions. Tune the defaults to the target regime in 3.3 and record the chosen values and plots in the README.

**M5 Evaluation utilities.** `metrics.py`, `monte_carlo`, `retention_curve`, sample-set distances, and the fork preview (K) in the game. A script that produces a retention curve and Monte Carlo tile distributions for a few absences `Δt`.

## 14. Out of scope for v1

3D, wind and slope (leave a hook in the spread rule), multiple agents, agent health, networking.

## 15. Open decisions

Defaults are set so that work can start. Confirm with Jonas before changing them:

- water tank with refills at lakes: on;
- the agent may walk through fire;
- world 256 × 256, window 33 × 33, tile 16 × 16;
- day length 240 steps;
- simulation rules in 3.3 (any change to the rules needs approval, because it changes the ground truth).
