# Wildfire testbed

Interactive 2D wildfire game and ground-truth simulator for experiments in partially observed
environments. The full specification is in [SPEC.md](SPEC.md).

## Install and run

```bash
uv sync
uv run pytest
uv run python -m wildfire.cli run --config configs/default.yaml --steps 2000
```

## Layout

| module | content |
|---|---|
| `config.py` | frozen `Config` dataclass, `Config.from_yaml` |
| `rng.py` | counter-based RNG: `fmix64(key(seed, stream, step, purpose) ^ cell_hash(y, x))` |
| `terrain.py` | value-noise terrain, rivers, initial vegetation, `initial_state` |
| `state.py` | `WorldState` (layers `(H, W)` or `(n, H, W)`), `stack`, `equal`, save/load |
| `actions.py` | action ids, `apply_action` (step 1 of the update) |
| `dynamics.py` | pure `step(state, action, cfg)`, `light(t, D)`, spread-weight hook |
| `cli.py` | `run` (headless statistics) |

## Performance

`step` on one CPU core (Apple Silicon, NumPy 2, `OMP_NUM_THREADS=1`), default parameters,
no-op actions, measured with `wildfire.cli run`:

| world | steps/s |
|---|---|
| 256 × 256 (2000 steps, seed 0) | 696 |
| 2048 × 2048 (200 steps, seed 0, 23k–49k burning cells) | 11.2 |

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
