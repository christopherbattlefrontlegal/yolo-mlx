# Project Instructions: yolo-mlx

Pure-MLX YOLO26 for Apple Silicon. Detection, segmentation, tracking, training. No PyTorch at runtime (convert-time only).

## Tech Stack

- Python 3.10+
- MLX `>=0.30.3,<0.31` (0.31.x hangs on yolo26x-seg training)
- setuptools + pyproject.toml
- CLI entry: `yolo-mlx` → `yolo26mlx.cli:main`

## Install

```bash
pip install -e ".[tracking,segment,convert,dev]"
```

Extras: `tracking` (opencv/lap/scipy), `segment` (pycocotools/matplotlib), `convert` (torch/ultralytics/safetensors), `dev` (pytest/ruff/pre-commit).

## Code Style

- snake_case files and functions; PascalCase classes (Results, Boxes, Masks, TrackerManager).
- Type annotations on new function signatures.
- Frozen dataclasses for value objects (`@dataclass(frozen=True)`).
- Prefer `pathlib.Path` over string concatenation for file paths.
- No `print` for diagnostics in library code; CLI/scripts may use print.

## Project Structure

```
src/yolo26mlx/
  cli.py              CLI dispatch
  cfg/                YAML configs (models, datasets, trackers)
  converters/         .pt -> .npz weight converter
  data/               COCODataset (detection + segmentation)
  engine/             YOLO, Predictor, Trainer, Validator, TrackerManager, Results
  nn/                 Detect, Segment26, Proto26, model builder
  optim/              MuSGD, AdamW
  trackers/           ByteTrack, BoT-SORT, Kalman, matching
  utils/              losses, ops, TAL, metrics, video I/O
scripts/              Benchmarks, evaluators, downloaders
tests/                pytest suite
configs/              Dataset YAMLs for scripts
models/, datasets/, images/, results/   runtime (gitignored)
```

## Build & Run

```bash
make install        # pip install -e ".[dev]"
make install-dev    # same + pre-commit
make lint           # ruff check
make format         # ruff format
make test           # pytest
make check          # lint + test
```

CLI:
```bash
yolo-mlx converters convert models/yolo26n.pt -o models/yolo26n.npz --verify
```

Python API:
```python
from yolo26mlx import YOLO
model = YOLO("models/yolo26n.npz")
model.predict("images/bus.jpg", conf=0.25)
model.track("video.mp4", tracker="bytetrack.yaml", save=True)
```

## Testing

- Run: `make test` or `pytest -q`
- Add tests next to feature: `tests/test_<area>.py`
- AAA pattern; descriptive test names.
- Coverage: `pytest --cov=src/yolo26mlx --cov-report=term-missing`

## Weights & Datasets

- Weights download script: `scripts/download_yolo26_models.sh` (GitHub releases).
- HF mirror: `Ultralytics/YOLO26` — use `hf download` CLI or `huggingface_hub`.
- `huggingface.co` may be reachable while `cas-bridge.xethub.hf.co` (LFS payloads) is blocked on some networks — fall back to browser download into `models/`.
- COCO: `bash scripts/download_coco_val2017.sh datasets/coco`
- MOT17: `bash scripts/download_mot17.sh`
- Conversion is **required** before inference: `.pt -> .npz` via `yolo-mlx converters convert`.

## Known Gotchas

- `mlx>=0.31` breaks yolo26x-seg training. Stay on 0.30.x.
- `Results.__getitem__` in `src/yolo26mlx/engine/results.py` indexes `self.masks.data[idx]` without guarding `self.masks.data is not None` — ByteTrack two-stage association on `-seg` models will crash on any frame where the `Masks` wrapper has `data=None`. Either fix the guard or use detection-only `yolo26{n,s,m,l,x}.npz` for tracking until patched.
- `model.track(..., save=True)` hardcodes the preview output to `results/<basename>_tracked.mp4` at CWD. Move after if you need a different layout.
- Frame index is not exposed on `Results`; track your own counter when iterating `stream=True`.

## Conventions

- Output discipline: write `frames.jsonl` (one record per line, `json.dumps(rec) + "\n"`, single file handle, line-buffered). Never `json.dump(..., "a")` with no newline.
- Embeddings (when wired in): store as `.f16.npy` memmaps, reference by string ID from JSONL. Do **not** inline 512-float vectors into per-frame records.
- Manifests: atomic write via `tmp + os.replace`.
- Runtime dirs (`datasets/`, `images/`, `models/`, `results/`) are created on demand and gitignored.

## Adding a New Runner / Feature

1. Implement in the appropriate subpackage under `src/yolo26mlx/`.
2. Wire into the engine (`engine/model.py`, `engine/predictor.py`) or CLI (`cli.py`) as needed.
3. Add a test in `tests/test_<area>.py`.
4. Run `make check`.

## Git

- Commit message style is not yet established (shallow history); use conventional commits (`feat:`, `fix:`, `docs:`).
- Pre-commit hooks: `make pre-commit-install` then `make pre-commit-run`.
