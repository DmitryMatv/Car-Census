# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

## Ask Before You Assume

Never guess at intent. If a task leaves anything open - which screen, which endpoint, what happens on failure, whether it needs a migration, whether this is user-facing - stop and ask. If an important decision is unresolved, stop and ask. One question up front is cheaper than half-a-day of work in the wrong direction.

- Ask when the request could reasonably mean two different things.
- Ask before changing a public API shape.
- Do not invent product decisions, copy, or acceptance criteria.
- Do not widen scope past what was asked. Note the adjacent thing you spotted; don't fix it unprompted.
- If you had to assume something you couldn't resolve, list it explicitly at the top of your summary.

## Testing

Never use `unittest` for backend testing. Always use Python's built-in `pytest` module instead.

Use Virtual Environment `source .venv/bin/activate` because `pytest` and everything else needed is installed there already probably.

## Project Snapshot

Car-Census is a video analysis and annotation tool for conducting real-world car make and model visual surveys (road traffic surveying). It detects and tracks cars in input footage, and then identifies make, model, generation/year range and variation using external Make and Model Recognition (MMR) API. Collected results are aggregated into structured tables for further data analysis and statistics. Annotated output video can be used to demonstrate the entire process.

## Type Safety

Maintain a strict type-safety direction for the project.

- `mypy` and `pyrefly check` should pass before considering type-related work complete.
- Prefer improving annotations, protocols, and narrow type guards over adding broad ignores.
- If an ignore or checker override is needed for dynamic third-party libraries, keep it as narrow and documented as practical.
- Do not expand Pyrefly or mypy exclusions just to hide ordinary source errors.
- Tests may be excluded from default type checking for now, but production source should keep moving toward stricter typing.

## Environment & Setup

- Default detector is RF-DETR-M through the `rfdetr` package at 576x576.
  Optional local offline checkpoint path: `detector.pretrain_weights`.
- Tracking uses Roboflow's `trackers` package. Do not create a local top-level
  Python package named `trackers`; it collides with the third-party dependency.
- Roboflow `trackers` scales `lost_track_buffer` by `analysis_fps / 30`; the
  configured value is not a direct analysis-frame count. Its BoT-SORT
  implementation uses Kalman motion, IoU, and detection confidence without
  appearance/ReID matching.
- BoT-SORT floors that scaled buffer and keeps tracks only while
  `time_since_update < maximum_frames_without_update`. For example,
  `lost_track_buffer: 15` at 5 analysis FPS becomes 2, so a mature track
  survives only one fully missed analysis frame; an immature track can be
  removed on its first miss.
- The footage comes from a static tripod-mounted camera; the camera never
  moves. Do not assume dashcam/moving-camera scenarios (ego-motion, CMC
  usefulness, "on-the-road" meaning the camera is in a car). CMC stays
  disabled; "on-the-road survey" refers to surveying road traffic.
- TrafficEye API key: `export TRAFFICEYE_API_KEY=your_key`
- TrafficEye manual `combinations` are projected into the response by order, but
  manually supplied boxes may not be returned. For batched MMR grids, match
  results by combination/cell order first; use returned boxes only as fallback.
- TrafficEye make/model/generation/variation identities can combine BEV and
  combustion-equipped factory variants. Use `MIXED` or `UNKNOWN` from the
  powertrain catalog when the complete identity does not support a reliable
  binary classification; do not force it to BEV or combustion.
- OpenCV Hershey fonts do not render country flag emoji. The installed
  `NotoColorEmoji.ttf` exposes a fixed 109-pixel color strike through Pillow;
  render flags at that native size and resize the raster for label scaling.
- A car can be present in `analysis/frames.jsonl` and still be invisible in
  rendered output. `visible_track_ids_for_render` intersects visible tracks with
  `size_eligible_track_ids`, which requires `max_box_width_px >=
analysis.min_box_width_px`. Check the analysis artifacts before assuming a
  visible missing annotation means detector/tracker failure.
- The current test baseline emits an upstream NumPy 2D-cross deprecation
  warning from Supervision `LineZone`. Do not broadly suppress it or treat it as
  a Car-Census counting failure.
- `configs/default.yaml` is the sole source of application defaults. Production
  code and tests must load defaults through `build_effective_config`; do not
  reintroduce zero-argument `AppConfig()` construction or Pydantic field
  defaults for application settings.
- `pyrefly check` currently reports a pre-existing baseline of ~40 findings,
  almost all in `tests/unit/`. Diff against the baseline (e.g. `git stash -u`,
  run pyrefly, pop, compare) before attributing new findings to your change;
  `--output-format json` wraps everything under an `errors` key including
  warnings.
- OpenCV's `cv2.getPerspectiveTransform` silently accepts degenerate or
  collinear calibration points instead of raising. `roi/transform.py`
  (`ViewTransformer`) guards this with a reprojection check; do not bypass it
  when building homographies elsewhere.
