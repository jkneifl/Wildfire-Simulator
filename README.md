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
obs, info = env.reset(seed=0)        # obs: grid (V, V, 6), rgb, light, agent{facing, tank}
obs, reward, terminated, truncated, info = env.step(action)
state = env.get_state()              # privileged WorldState (copy)
env.set_state(state)
futures = env.fork(n=16, steps=200)  # batched WorldState (16, H, W), one stream id per future
vec = VecWildfire(cfg, n=8)          # steps 8 worlds with arrays (8, H, W)
```

`info` holds `step`, the agent's own position `pos`, the tiles intersecting the window and the
per-tile `first_seen` / `last_seen` steps (−1 = never). Gymnasium spaces are defined when the
optional `gym` extra is installed (`uv sync --extra gym`).

Episode directories follow SPEC 9 (`meta.json`, `actions.npy`, `agent.npy`, `obs_grid.npy`,
optional `obs_rgb.npy`, `tiles_seen.npy` with shape `(2, rows, cols)` = first/last seen,
`state/step_XXXXXX.npz` every `snapshot_every` episode steps plus the final step).
`EpisodeDataset(root)` loads them with NumPy (memory-mapped).

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
| `app/play.py` | pygame-ce frontend |
| `cli.py` | `run` (headless statistics), `generate`, `replay` |

## Performance

`step` on one CPU core (Apple Silicon, NumPy 2, `OMP_NUM_THREADS=1`), default parameters,
no-op actions, measured with `wildfire.cli run`:

| world | steps/s |
|---|---|
| 256 × 256 (2000 steps, seed 0) | 696 |
| 2048 × 2048 (200 steps, seed 0, 23k–49k burning cells) | 11.2 |

Environment overhead on 256 × 256 (500 no-op steps, seed 0): `WildfireEnv` 637 steps/s without
and 522 with `rgb`; `VecWildfire(n=16)` 702 world-steps/s; `fork(16, 100)` 2.2 s.

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
  burning, fire/wet/ash buckets), so `obs["rgb"]` carries no more state than `obs["grid"]`
  (apart from the `FLICKER` noise). Ash shades and the wetness tint are therefore per bucket.
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
- **Episode seeds.** `generate` uses seed `--seed + i` for episode `i`, for both the world
  and the policy.
