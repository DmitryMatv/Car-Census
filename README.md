# Car-Census

Offline vehicle tracking and make/model census for roadside video.

## What It Does

`car-census` takes a video, analyzes only a configured polygon zone, tracks vehicles with BoT-SORT, selects the best crops per track, sends those crops to TrafficEye/Eyedea for make/model recognition, then renders a clean annotated video with offline-smoothed boxes and exports counts.

The pipeline is staged:

1. `roi edit` to define the polygon ROI
2. `analyze` to detect, track, count polygon-zone tracks, and save crop candidates
3. `classify` to run TrafficEye make/model recognition
4. `render` to create the annotated output video
5. `report` to export counts

If `--camera-id` is omitted for `analyze` or `run`, the full frame is used as the ROI.

## Environment

This project targets local CPU inference with ONNX Runtime by default. Install project dependencies:

```bash
pip install -e .
```

The legacy Ultralytics/PyTorch detector remains available as an optional extra:

```bash
pip install -e ".[pytorch]"
```

BoT-SORT tracking uses BoxMOT. BoxMOT currently supports Python versions below 3.13, so use a Python 3.12 environment for tracking runs:

```bash
pip install -e ".[tracking]"
```

## Model Setup

The default configuration expects a local YOLO26 ONNX detector at:

```text
weights/yolo26n.onnx
```

Use a YOLO26 detection model exported to ONNX. The default path is intentionally a local artifact so the pipeline can run offline after setup.

For quick smoke testing with the optional PyTorch path, set `detector.provider` to `ultralytics_local` and `detector.weights` to a local `.pt` model or an Ultralytics model name.

Device selection is explicit:

```bash
car-census run input_data/test_vid.MP4 --camera-id home-road --device cpu
```

The ONNX Runtime detector always uses `CPUExecutionProvider`; `project.device` only affects the optional Ultralytics/PyTorch provider.

## TrafficEye Setup

Set your API key in the environment:

```bash
export TRAFFICEYE_API_KEY=your_key_here
```

## Quick Start

Create an ROI profile:

```bash
car-census roi edit input_data/test_vid.MP4 --camera-id home-road --device cuda
```

Run the full pipeline:

```bash
car-census run input_data/test_vid.MP4 --camera-id home-road --device cuda
```

Run without make/model API calls:

```bash
car-census run input_data/test_vid.MP4 --skip-classify
```

Artifacts are written to `outputs/<run-id>/`.

## Output Layout

```text
outputs/<run-id>/
  run.json
  analysis/
    frames.jsonl
    render_frames.jsonl
    tracks.jsonl
    count_events.jsonl
  crops/
  mmr/
    labels.json
    cache/
  render/
    annotated.mp4
  reports/
    counts.csv
    counts.json
```

## Notes

- Analysis defaults to the source video FPS. Set `analysis.fps` to a positive value if you want downsampling.
- Render output defaults to the source video FPS. Set `render.output_fps` to a positive value if you want a different export rate.
- Counting uses tracked vehicles inside the configured polygon zone. Older camera profiles with `count_line` are still supported.
- By default, `tracker.ignore_edge_touches: true` ignores detections and tracker outputs whose boxes touch the source-frame edge or selected camera crop edge. Increase `tracker.edge_margin_px` to ignore boxes that are near, but not exactly on, the edge.
- `analysis/frames.jsonl` contains raw tracker output. `analysis/render_frames.jsonl` is generated for annotation only and does not affect counts, crops, or make/model classification.
- `render.smoothing.interpolate` controls whether source-frame annotations are generated between analyzed frames. `render.smoothing.interpolation_method: hermite` is the default and uses monotone cubic Hermite interpolation through tracked keyframes to reduce overshoot. `linear` uses straight-line fills, while `polynomial` uses an arbitrary local polynomial fit and is mainly experimental. `polynomial_order` affects only polynomial interpolation and linear keyframe smoothing. `max_center_offset_ratio` and `max_size_delta_ratio` clamp generated boxes against the linear reference path. The `polynomial` and `hermite` methods preserve actual tracked keyframes exactly.
- For static cameras, leave `tracker.cmc_method` as `null` to disable camera motion compensation. Supported BoxMOT CMC values are `ecc`, `orb`, `sift`, and `sof`.
- Rendering uses ellipse markers and compact make/model labels by default.
