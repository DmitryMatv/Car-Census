# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

Never use `unittest` for backend testing. Always use Python's built-in `pytest` module instead.

Use Virtual Environment `source .venv/bin/activate` because `pytest` and everything else needed is installed there already probably.

## Project Snapshot

Car-Census is a video analysis and annotation tool for conducting real-world, on-the-road car make and model visual surveys. It detects and tracks cars in input footage, and then identifies make, model, generation/year range and variation using external Make and Model Recognition (MMR) API. Collected results are aggregated into structured tables for further data analysis and statistics. Annotated output video can be used to demonstrate the entire process.

## Type Safety

Maintain a strict type-safety direction for the project.

- `mypy` and `pyrefly check` should pass before considering type-related work complete.
- Prefer improving annotations, protocols, and narrow type guards over adding broad ignores.
- If an ignore or checker override is needed for dynamic third-party libraries, keep it as narrow and documented as practical.
- Do not expand Pyrefly or mypy exclusions just to hide ordinary source errors.
- Tests may be excluded from default type checking for now, but production source should keep moving toward stricter typing.

## Environment & Setup

- Default detector is RF-DETR-S through the `rfdetr` package at 512x512.
  Optional local offline checkpoint path: `detector.pretrain_weights`.
- Tracking uses Roboflow's `trackers` package. Do not create a local top-level
  Python package named `trackers`; it collides with the third-party dependency.
- TrafficEye API key: `export TRAFFICEYE_API_KEY=your_key`
- TrafficEye manual `combinations` are projected into the response by order, but
  manually supplied boxes may not be returned. For batched MMR grids, match
  results by combination/cell order first; use returned boxes only as fallback.
- A car can be present in `analysis/frames.jsonl` and still be invisible in
  rendered output. `visible_track_ids_for_render` intersects visible tracks with
  `size_eligible_track_ids`, which requires `max_box_width_px >=
analysis.min_box_width_px`. Check the analysis artifacts before assuming a
  visible missing annotation means detector/tracker failure.
