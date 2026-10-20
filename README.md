# Wildfire testbed

Interactive 2D wildfire game and ground-truth simulator for experiments in partially observed
environments. The full specification is in [SPEC.md](SPEC.md).

## Install and run

```bash
uv sync
uv run pytest
uv run python -m wildfire.cli run --config configs/default.yaml --steps 2000
uv run python -m wildfire.app.play --seed 0     # play
uv run python -m wildfire.cli generate --config configs/default.yaml --policy explore_return \
    --episodes 10 --steps 2000 --out data/test
uv run python -m wildfire.cli replay data/test/ep_0000
```

## Environment API

```python
from wildfire import Config, WildfireEnv, VecWildfire

env = WildfireEnv(Config.from_yaml("configs/default.yaml"))
obs, info = env.reset(seed=0)  # obs: grid (V, V, 6), rgb, light, agent{facing, tank}
obs, reward, terminated, truncated, info = env.step(action)
state = env.get_state()  # privileged WorldState (copy)
env.set_state(state)
futures = env.fork(n=16, steps=200)  # batched WorldState (16, H, W), one stream id per future
vec = VecWildfire(cfg, n=8)  # steps 8 worlds with arrays (8, H, W)
```

`info` holds `step`, the agent's own position `pos`, the tiles intersecting the window and the
per-tile `first_seen` / `last_seen` steps (−1 = never). Gymnasium spaces are defined when the
optional `gym` extra is installed (`uv sync --extra gym`).

Episode directories follow SPEC 9 (`meta.json`, `actions.npy`, `agent.npy`, `obs_grid.npy`,
optional `obs_rgb.npy`, `tiles_seen.npy` with shape `(2, rows, cols)` = first/last seen,
`state/step_XXXXXX.npz` every `snapshot_every` episode steps plus the final step).
`EpisodeDataset(root)` loads them with NumPy (memory-mapped).

## Browser version

`web/` is a browser build of the game: `web/wildfire.js` is a JavaScript port of the
simulation core (RNG, terrain, actions, step, observation buckets, colours) and
`web/index.html` is the frontend (agent and god view, last-seen overlay, minimap, inspector,
pause/step/speed, reset, save/load in browser storage, fork preview in Web Workers, touch
controls). F2 recording is Python-only. The port is checked bit for bit against the Python
package by `tests/test_web_parity.py`, which runs `web/parity.js` under Node. Covered: 64 × 64
with high lightning (600 steps), 40 × 70 with day length 50 and tank 3 (400 steps), and the
default 256 × 256 (300 steps), with random moves, ignite and water, plus fork stream ids. The
test is skipped if `node` is not installed. Any change to the Python rules must be mirrored in
`web/wildfire.js`; the parity test fails otherwise. To run it locally:

```bash
python3 -m http.server 8765 --directory web    # then open http://localhost:8765
```

## Game controls

| key | action |
|---|---|
| arrows / WASD | move (repeats while held) |
| F / Space | ignite / water the cell in front |
| Tab | agent view / god view (mouse wheel zooms, right or middle drag pans) |
| G | last-seen overlay (agent view) |
| M / I | minimap / cell inspector (hover a cell) |
| P / N | pause / single step |
| + / − | simulation speed (2, 5, 10, 20, 40 steps/s) |
| R / Shift+R | reset with seed + 1 / the same seed |
| F5 / F9 | save / load `saves/quicksave.npz` |
| F2 | start or stop recording to `recordings/` |
| K | fork preview: burn probability over 16 futures × 100 steps (god view) |
| Esc | quit |

## Layout

| module | content |
|---|---|
| `config.py` | frozen `Config` dataclass, `Config.from_yaml` |
| `rng.py` | counter-based RNG: `fmix64(key(seed, stream, step, purpose) ^ cell_hash(y, x))` |
| `terrain.py` | value-noise terrain, rivers, initial vegetation, `initial_state` |
| `state.py` | `WorldState` (layers `(H, W)` or `(n, H, W)`), `stack`, `equal`, save/load |
| `actions.py` | action ids, `apply_action` (step 1 of the update) |
| `dynamics.py` | pure `step(state, action, cfg)`, `light(t, D)`, spread-weight hook |
| `observe.py` | window crop, observation buckets, `observe`, `privileged` |
| `render/palette.py`, `render/topdown.py` | palette; pure NumPy `render(state, window, cfg)` |
| `tiles.py` | tile indices, `tiles_in_window`, `extract_tile`, `TileTracker` |
| `env.py` | `WildfireEnv`, `VecWildfire`, `fork` |
| `record.py` | `Recorder`, `replay`, `EpisodeDataset` |
| `policies.py` | scripted policies |
| `metrics.py` | global and patch statistics, `simulate_stats`, `regime_summary` |
| `app/play.py` | pygame-ce frontend |
| `cli.py` | `run` (headless statistics), `generate`, `replay` |

## Performance

`step` on one CPU core (Apple Silicon, NumPy 2, `OMP_NUM_THREADS=1`), default parameters,
no-op actions, measured with `wildfire.cli run`:

| world | steps/s |
|---|---|
| 256 × 256 (2000 steps, seed 0), start defaults | 696 |
| 256 × 256 (2000 steps, seed 0), calibrated defaults | 621 |
| 2048 × 2048 (200 steps, seed 0, 23k–49k burning cells), start defaults | 11.2 |

Environment overhead on 256 × 256 (500 no-op steps, seed 0): `WildfireEnv` 637 steps/s without
and 522 with `rgb`; `VecWildfire(n=16)` 702 world-steps/s; `fork(16, 100)` 2.2 s.

## Scripted policies

`random_walk`, `explore_return` (observe a tile for 40 steps, leave until it is out of view,
stay away for a log-uniform `Δt` in 50–2000 steps without bringing it back into view, return;
`policy_log` in `meta.json` lists `{tile, leave_step, return_step, dt}`), `firestarter`
(random walk, ignites with probability 0.02 per step), `firefighter` (waters the nearest
visible fire, refills at the nearest visible water). All randomness uses purpose `POLICY`
with the policy's seed.

## Calibration (M4)

`scripts/calibrate.py` runs headless no-op simulations on 256 × 256, 30 days each (7200
steps), excludes the first 5 days, and reports per seed: fraction of steps with at least one
burning cell, longest fire-free stretch, vegetation density and ash fraction of ground cells.
It plots the time series and the size distributions (CCDF) of 4-connected burned and vegetated
components in daily snapshots.

```bash
uv run python scripts/calibrate.py --seeds 0 1 2 3 4 --days 30 --out docs/calibration/calibrated
```

Starting defaults miss the target regime: on seeds 0–2, fire is active in only 64 % of steps,
with fire-free stretches up to 2.2 days, and vegetation swings between 11 % and 91 % in
map-wide boom-bust cycles. 15 `(beta, p_lightning)` combinations were tried (beta ∈ {0.05,
0.1, 0.2, 0.35}, p_lightning ∈ {2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5}, 2 seeds, 30 days); all
other parameters are unchanged. More lightning removes the fire-free gaps; a lower `beta`
keeps fires from sweeping the connected forest. `beta = 0.2, p_lightning = 2e-6` also met the
numbers but showed a ~2-day oscillation that is nearly in phase across seeds; `beta = 0.1`
gives less synchronised, more local fires.

| parameter | start | calibrated |
|---|---|---|
| `beta` | 0.35 | **0.1** |
| `p_lightning` | 2e-7 | **1e-6** |

Means over seeds after the 5-day burn-in (256 × 256, 30 days; per-seed values in
`docs/calibration/*/summary.json`):

| config | seeds | steps with fire | longest fire-free | veg density min–mean–max | ash fraction mean / max | active fires (mean cells) |
|---|---|---|---|---|---|---|
| start defaults | 0–2 | 63.9 % | 2.17 days | 0.11 – 0.51 – 0.91 | 0.49 / 0.89 | 462 |
| calibrated | 0–4 | 97.5 % | 0.15 days (max 0.5, seed 3) | 0.36 – 0.60 – 0.81 | 0.40 / 0.64 | 438 |

Seeds 3 and 4 were not used for the choice; they agree with seeds 0–2. Ash persists for
roughly 2–3 days on average before a cell is reseeded (40 % ash cover at about 50 cells
burning out per step), which matches "burned areas regrow over a few days". Patch sizes are
broad and heavy-tailed (1 to ~3·10⁴ cells), without a clean power law; this is a
Drossel–Schwabl-like regime, not a verified critical state.

Plots: [time series](docs/calibration/calibrated/timeseries.png),
[patch sizes](docs/calibration/calibrated/patches.png); start defaults:
[time series](docs/calibration/initial_defaults/timeseries.png),
[patch sizes](docs/calibration/initial_defaults/patches.png).

![calibrated time series](docs/calibration/calibrated/timeseries.png)

## Evaluation utilities (M5)

`metrics.py`: `global_stats`, `patch_stats` (component sizes via `scipy.ndimage.label`),
`tile_stats` (per tile, over ground cells: vegetated fraction, mean stage / V_MAX, ash
fraction, burning fraction, mean wetness / 255; batched), `monte_carlo(state, cfg, steps, n,
tiles)` (tile statistics of `n` forked futures), `burn_probability` (used by the K preview),
`retention_curve(series, lags, stat)`, `episode_tile_series(ep_dir)`, `energy_distance`
and `mmd` (unbiased MMD², Gaussian kernel, median-heuristic bandwidth).

```bash
uv run python scripts/evaluate.py --out docs/evaluation                 # headless runs
uv run python scripts/evaluate.py --episodes data/test --out /tmp/eval  # recorded episodes
```

Output from `scripts/evaluate.py` with its defaults (calibrated `configs/default.yaml`,
256 × 256, 256 tiles of 16 × 16). Retention uses no-op runs with seeds 0–3, days 5–20,
snapshots every 50 steps. For vegetated fraction, the correlation between τ and τ + Δt is
0.885 at Δt = 50, 0.269 at 200 and −0.51 at 550. It then oscillates, with peaks of 0.49 at
1150, 0.32 at 2250–2300 and 0.21 at 3300 steps. In this regime a tile's state is roughly
anti-correlated with itself after about two days, and after about five days it is positively
correlated again. That suggests a local burn–regrow cycle of about 1150 steps (4.8 days). This
is a measured correlation, not a fitted model of the cycle. The ash fraction curve almost
matches the vegetated one, since on ground cells the two are nearly complementary.

Monte Carlo: 32 futures from seed 0 at τ = step 2460 (day 10), Δt ∈ {50, 200, 800}. The 4
tiles are the 2 most vegetated and the 2 most burned at τ. For tile (6, 7), all 32 futures
burn down by Δt = 200 (vegetated fraction 0.98 → mean 0.05) and regrow to 0.76 by
Δt = 800. Burned tiles (0, 0) and (1, 0) regrow from 0.02–0.04 to 0.89–0.90. Energy distance
to the last-seen value grows with Δt for the burned tiles (0.23 → 0.78 → 2.7). Per-tile
numbers are in `docs/evaluation/evaluation.json`. Plots:
[retention](docs/evaluation/retention.png), [Monte Carlo](docs/evaluation/monte_carlo.png).

![retention](docs/evaluation/retention.png)

## Decisions

Choices where the spec is ambiguous. Each keeps determinism and the observable/privileged split.

- **RNG.** MurmurHash3 `fmix64` finalizer; key = chained `fmix64` of seed, stream, step and
  purpose; cell hash = `fmix64((y << 32) | x)` with global coordinates. Probabilities are
  compared as `bits < p * 2**64` (integer) or as floats from the top 53 bits.
  Growth and seeding share one hash per step: each cell uses the `GROWTH` key if `veg > 0`
  and the `SEED` key otherwise. This is identical to evaluating both purposes everywhere and
  selecting per cell.
- **Terrain** uses purpose `TERRAIN` with stream 0, so it depends only on the seed; forks
  share terrain. The `step` slot of the counter indexes the noise field and octave.
  1–2 rivers (sinusoidal meanders, 3–5 cells wide); lakes are the top value-noise cells chosen
  so that water totals `water_frac` (8 %); rock is the top `rock_frac` (4 %) of a second field.
- **Synchronous update.** Rules 2–7 read the state after the action. Burning cells still spread
  on the step they burn out. Growth applies only to cells that are not burning and not igniting
  in this step. Ignited cells (spread or lightning) get `BURN_BASE + BURN_PER_VEG * veg`.
- **Ash cleared on regrowth.** When a bare cell sprouts (`veg = 1`) its `ash_age` is reset
  to 0, so `veg > 0` implies no ash. Otherwise regrown forest would keep counting as ash.
- **Lightning** follows the spec literally (no wetness condition).
- **Actions.** A move sets the facing direction even when blocked. Watering works when the
  target is outside the world (the in-world part of the 3 × 3 block is affected) and costs one
  unit. The tank refills when one of the four orthogonal neighbours is water, checked after
  the action, every step. Initial facing is south.
- **Batches.** The same `step` handles `(H, W)` and `(n, H, W)`; scalars are int64 arrays with
  the batch shape. Terrain is replicated per world.
- **Renderer uses only observable channels.** Colours come from `observe.grid` (terrain, veg,
  burning, fire/wet/ash buckets), so `obs["rgb"]` carries no more state than `obs["grid"]`.
  The two per-cell textures depend only on the seed and the cell: the variant bit (purpose
  `TERRAIN`, sub-counter 64) and the flame phase `u_xy` (purpose `FLICKER`, stream 0, step 0).
  The flame changes over time only through `t`, so the image no longer depends on the stream.
- **Visual style** (`render/palette.py`). There are four base
  colours, and every other colour is a linear sRGB mix of them, rounded half up. Choices
  where the style sheet is open:
  - Wetness uses the observable bucket: the tint is `0.30 * 85 * bucket / 255`, standing for
    wet = 85 × bucket.
  - The darker vegetation variant is `mix(GROUND, VEG[stage], 0.72)` of the rounded colour.
    This reproduces all four table values; `mix(GROUND, ACCENT, 0.72 f)` gives green 77
    instead of 76 (`#244C45`) at stage 4.
  - Water uses the same per-cell variant bit as vegetation.
  - Flames and embers are not wet-tinted and not dimmed at night. The SPEC 5 night factor
    `0.35 + 0.65 L` is still applied to all other cells.
  - The grid gap is inset `g // 2` on the left and top and the rest on the right and bottom,
    and it is drawn in `GROUND` at all times of day.
  - Overlays in the game use mixes of the base colours: last seen is `mix(c, GROUND, 0.55)`,
    the fork preview is `mix(c, INK, 0.75 p)`, and the HUD uses `INK`, `ACCENT` for
    highlights, `FIRE` for warnings and `mix(GROUND, INK, 0.6)` for secondary text.
  - Water and wetness colours are implemented as proposed and still need confirmation.
  - **Brightened after review** ("too dark"): the cell mix fractions are raised about 1.7×.

    | colour | style sheet | now |
    |---|---|---|
    | bare ground | 0.07 | 0.12 |
    | rock | 0.30 | 0.42 |
    | fresh ash | 0.15 | 0.24 |
    | ember | 0.50 | 0.62 |
    | vegetation, stages 1–4 | 0.20 / 0.28 / 0.37 / 0.46 | 0.34 / 0.48 / 0.63 / 0.78 |
    | water / variant | `#1D4A63` / `#22556F` | `#2A6688` / `#307596` |

    The base colours, GROUND gaps and fog, the darker variant (0.72) and the flame are
    unchanged.
  - **Wet tint stronger:** `WET_MIX` 0.30 → 0.65 (bucket 3 mixes 65 % towards `#5FA8D3`),
    so sprayed cells are clearly visible.
  - **Night floor raised (deviates from SPEC 5, on request):** brightness is
    `0.50 + 0.50 L` instead of `0.35 + 0.65 L`. Burning cells are still not dimmed.
- **Buckets.** fire: `ceil(3 * fire / FIRE_MAX)` with `FIRE_MAX = BURN_BASE + BURN_PER_VEG *
  V_MAX`; wet: 1–85, 86–170, 171–255; ash: thirds of `REGROW_DELAY`, bucket 3 from
  `2/3 REGROW_DELAY` on.
- **Game.** The last-seen memory stores the observable grid of every cell when it was last in
  the window. The minimap always shows the live world (it is a demo aid, not an observation).
  If the simulation falls behind, steps are dropped (at most 5 per frame) rather than queued.
- **Fork streams.** Future `i` of a state gets stream `key(stream, i, t, FORK) mod 2**63`,
  so forks of different states or times do not share streams. Futures use no-op actions
  unless `actions` is given.
- **`info["pos"]`.** The agent's own position is given to policies (it is known from the
  agent's own moves); it is not part of `obs`, which stays egocentric as in SPEC 4.
- **Recording** starts from any state. `meta.json` has `from_reset = true` when the episode
  starts at `initial_state(seed, stream)`; replay then rebuilds from the seed and also checks
  the step-0 snapshot, otherwise it starts from the step-0 snapshot. Replay checks every
  snapshot, every observation grid and the agent trajectory. Game recordings go to
  `recordings/ep_<date>_<time>`. `preview.mp4` is not written (no video dependency).
- **Policy randomness** goes through `rng.py` with purpose `POLICY` (not part of the
  simulation's purpose list), keyed by the policy seed and a per-call counter.
- **Episode seeds.** `generate` uses seed `--seed + i` for episode `i`, for both the world
  and the policy.
- **Tile statistics** use only ground cells of a tile, and tiles at the world edge are
  partial. The retention curve pools Pearson correlations over all tiles, τ and runs. Static
  terrain differences between tiles are included and are not removed.
- **Small worlds have extra water.** Rivers are kept whole, and lakes only fill up to
  `water_frac`, so on small maps the rivers alone can exceed it. Seeds 0–29 at 64 × 64 reach
  up to 15.6 % water (8.0 % at 256 × 256). Left unchanged, because changing it would change
  existing worlds.
- **Fork preview (K)** blocks the game for about 2 s on 256 × 256 (16 × 100 no-op steps). The
  heat map (magenta = burns in all futures) stays until K is pressed again or the world is
  reset or loaded.
