# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

Never use `unittest` for backend testing. Always use Python's built-in `pytest` module instead.

Use Virtual Environment `source .venv/bin/activate` because `pytest` and everything else needed is installed there already probably.

## Environment & Setup

- Requires Python 3.12 for tracking (BoxMOT incompatibility with 3.13+)
- ONNX model expected at `weights/yolo26s.onnx` (local offline use)
- TrafficEye API key: `export TRAFFICEYE_API_KEY=your_key`
- TrafficEye manual `combinations` are projected into the response by order, but
  manually supplied boxes may not be returned. For batched MMR grids, match
  results by combination/cell order first; use returned boxes only as fallback.
- `analysis/tracks.jsonl` may contain absolute crop paths from the original run
  location. If a run directory is renamed or moved, classification should resolve
  crop filenames against the current run's `crops/` directory.
- Keep `trackers/__init__.py` and pytest `pythonpath = ["."]`; otherwise the
  `.venv` site-packages `trackers` package can shadow the local tracker adapters
  during test collection.

## Project Snapshot

## Tech Stack
