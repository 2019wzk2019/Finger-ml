# Finger-ml Agent Notes

## Package & commands

- Project name `finger-ml`, Python package `finger_ml` (hyphen vs underscore).
- Managed by **uv** — always use `uv run` and `uv sync`, never bare `pip`.
- CLI entrypoints: `finger-collect`, `finger-preprocess`, `finger-train`, `finger-detect`, `finger-audit`, `finger-eval`, `finger-review` (in `pyproject.toml [project.scripts]`).

## Install

```bash
uv sync                              # runtime only (no torch)
uv sync --group train                # + torch CPU
uv sync --group train-cuda           # + torch CUDA 13.0 (needs NVIDIA + CUDA 12.1+ driver)
```

## Pipeline order

collect → review → preprocess → audit → train → detect → eval. Each step consumes the previous step's output.

## Key gotchas

- **Preprocess once, train many**: `finger-preprocess` extracts `.npz` features into `data/features/`. Reuse them across training experiments; don't re-extract landmarks each time.
- **VIDEO mode, not IMAGE**: `hand_tracking.py` uses `RunningMode.VIDEO` for temporal tracking between frames. Do not switch to `IMAGE` mode for video files — it drops tracking and is slower.
- **VIDEO-mode landmarker is single-use per video**: Timestamps must be monotonically increasing. The landmarker is created fresh per session in `preprocess.py` and closed after; you cannot reuse it across videos.
- **MediaPipe model auto-download**: First run downloads `hand_landmarker.task` to `.models/` (gitignored). Offline-first run will fail without it.
- **GPU delegate fallback**: `--delegate GPU` may fail depending on platform/MediaPipe wheel; code falls back to CPU. CPU is already faster than 60 fps video on modern hardware.
- **7 classes**: `labels.py` defines `GESTURE_ORDER` (6 gesture types, indices 0–5) and `BACKGROUND_LABEL = 6`. `NUM_CLASSES = 7`. `LABEL_NAMES` is the canonical ordered tuple.
- **train_mask vs labels**: Preprocess builds a `train_mask` array — frames near gesture boundaries (transition/rebound) are labeled but masked out of training loss via `IGNORE_INDEX = -100`. Don't just use raw labels for training.
- **Frame indexing**: JSON annotations use 1-indexed `start_frame`/`end_frame`; `preprocess.py` converts to 0-indexed and applies `BOUNDARY_MARGIN = 2` trim on each side.
- **`--hand-side` matters**: Default is `None` (first hand found). For single-hand datasets, pass `Right` or `Left` to avoid picking the wrong hand.

## Gitignored directories

`data/`, `checkpoints/`, `results/`, `.models/` — not in repo, must be created by pipeline steps. Do not assume they exist.

## No test suite

No tests directory or test runner configured. Verify by running pipeline steps end-to-end.
