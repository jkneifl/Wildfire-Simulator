# Wildfire testbed

Interactive 2D wildfire game and ground-truth simulator for experiments in partially observed environments. The full specification is in `SPEC.md`. Read it before starting any task.

## Principles

- **Determinism.** A run is determined by config, seed and actions. Never use `np.random` global state, `random`, or time-based seeds in the simulation or renderer.
- **Counter-based RNG only.** All randomness goes through `rng.py`: a pure function of `(seed, stream, step, purpose, y, x)`.
- **Pure core.** `dynamics.step` is a pure function on arrays. Nothing in `src/wildfire/` except `app/` may import pygame.
- **Vectorize.** No Python loops over cells. Loops over a small fixed number of neighbours are fine.
- **Observable vs. privileged.** Keep the split from SPEC section 4 strict. Anything an agent could receive must come from `observe()`.
- **Ground truth is sacred.** Do not change simulation rules or their defaults (SPEC 3.3, 15) without asking. Parameter values may change only through the calibration milestone, with the result documented.

## Conventions

- Python ≥ 3.11, type hints, dataclasses, `ruff` for linting and formatting, `pytest` for tests.
- Arrays are indexed `[y, x]`; `x` to the right, `y` downwards; north is `y - 1`. Batched arrays are `(n, H, W)`.
- Layers use the dtypes in SPEC 3.1. Avoid silent upcasting in the step function.
- Configuration in YAML under `configs/`, loaded into a frozen dataclass.
- Every simulation rule has at least one test. Statistical tests use fixed seeds and generous tolerances.

## Commands

```bash
uv sync                                       # install
uv run pytest                                 # tests
uv run ruff check . && uv run ruff format .   # lint and format
uv run python -m wildfire.app.play            # play
uv run python -m wildfire.cli generate --config configs/default.yaml --policy explore_return --episodes 10 --steps 2000 --out data/test
uv run python -m wildfire.cli replay data/test/ep_0000
uv run python scripts/calibrate.py
```

## Workflow

- Implement the milestones in SPEC section 13 in order. At the end of each one: tests pass, README updated, one commit.
- When the spec is ambiguous, choose the simplest option that keeps determinism and the observable/privileged split, note it in the README under "Decisions", and mention it in your summary.
- Report performance figures (steps per second) after M1 and whenever the step function changes.
