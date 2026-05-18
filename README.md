# Car-Census

Offline vehicle tracking and make/model census for roadside video.

## What It Does

`Car-Census` takes a video, analyzes only a configured polygon zone, tracks vehicles with BoT-SORT, selects the best crops per track, sends those crops to TrafficEye/Eyedea for make/model recognition, then renders a clean annotated video with offline-smoothed boxes and exports a detailed vehicle CSV.

The pipeline is staged:

1. `roi edit` to define the polygon ROI
2. `analyze` to detect, track, count polygon-zone tracks, and save crop candidates
3. `classify` to run TrafficEye make/model recognition
4. `render` to create the annotated output video
5. `report` to export `report.csv`

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

BoT-SORT tracking uses Roboflow's `trackers` package, which is installed with
the main project dependencies.

## Model Setup

The default configuration expects a local YOLO26 ONNX detector at:

```text
weights/yolo26s.onnx
```

Use a YOLO26 detection model exported to ONNX. The default path is intentionally a local artifact so the pipeline can run offline after setup.

For quick smoke testing with the optional PyTorch path, set `detector.provider` to `ultralytics_local` and `detector.weights` to a local `.pt` model or an Ultralytics model name.

Device selection is explicit:

```bash
Car-Census run input_data/test_vid.MP4 --camera-id home-road --device cpu
```

By default, the ONNX Runtime detector uses `CPUExecutionProvider`.
`--accelerator colab-t4`, `--accelerator onnx-cuda`, or a config override can
request CUDA/TensorRT providers. `project.device` still affects the optional
Ultralytics/PyTorch provider.

## Google Colab T4

For Colab GPU runs, install the project and use the GPU ONNX Runtime package:

```bash
source .venv/bin/activate
pip install -e .
pip install onnxruntime-gpu
Car-Census run input_data/test4K.MP4 --camera-id my-camera --accelerator colab-t4
```

The `--accelerator colab-t4` preset keeps the existing ONNX weights path and
requests ONNX Runtime CUDA execution with CPU fallback registered. ONNX is not
required for GPU inference in general, but it is the cleanest GPU path for this
project because the default detector is already ONNX-based. PyTorch CUDA remains
available through `detector.provider: ultralytics_local` and a local `.pt`
model or Ultralytics model name.

If the runtime still has the CPU-only `onnxruntime` package, replace it with
`onnxruntime-gpu`. TensorRT can be requested explicitly with
`--accelerator tensorrt`, but it requires ONNX Runtime TensorRT provider
dependencies in the Colab runtime and does not auto-export `.engine` files.

Colab FFmpeg builds vary. `--accelerator colab-t4` enables `auto-nvenc` for the
final video encode, which uses NVENC when FFmpeg exposes it and falls back to
the current OpenCV writer when it does not.

## TrafficEye Setup

Set your API key in the environment:

```bash
export TRAFFICEYE_API_KEY=your_key_here
```

Classification sends one selected crop per vehicle to TrafficEye. By default,
crops are packed into 4x4 composite images (`mmr.batch_size: 16`) and sent with
manual BOX detections for each grid cell, so batch requests use `MMR` without
`DETECTION`. Set `mmr.batch_size: 1` to restore one API request per crop with
`DETECTION` and `MMR`. OCR and plate detection are intentionally not requested.
Each composite image is saved under `mmr/batch_grids/` with a JSON sidecar that
maps source crop paths to grid cells. The full TrafficEye response is preserved
under `mmr/labels.json[*].raw`; common MMR fields such as make, model,
generation, color, tags, and the selected detection box are also promoted to
typed label fields.

## Quick Start

Create an ROI profile:

```bash
Car-Census roi edit input_data/test_vid.MP4 --camera-id home-road --device cuda
```

Run the full pipeline:

```bash
Car-Census run input_data/test_vid.MP4 --camera-id home-road --device cuda
```

Run without make/model API calls:

```bash
Car-Census run input_data/test_vid.MP4 --skip-classify
```

Artifacts are written to `output/<run-id>/`.

## Output Layout

```text
output/<run-id>/
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
  annotated.mp4
  reports/
    report.csv
```

## Notes

- Input videos are expected to be 30 fps by default. `video.fps` controls source-frame timestamps and render output FPS; `video.fps_tolerance` controls how much OpenCV-reported input FPS drift is accepted before the run fails.
- Analysis can run at a lower cadence for faster tracking. Set `analysis.fps` to the desired tracking rate, such as `10`; rendering still processes every decoded input frame and writes at `video.fps`.
- Counting uses tracked vehicles inside the configured polygon zone. Older camera profiles with `count_line` are still supported.
- By default, `tracker.ignore_edge_touches: true` ignores detections and tracker outputs whose boxes touch the source-frame edge or selected camera crop edge. Increase `tracker.edge_margin_px` to ignore boxes that are near, but not exactly on, the edge.
- `analysis/frames.jsonl` contains raw tracker output. `analysis/render_frames.jsonl` is generated for annotation only and does not affect counts, crops, or make/model classification.
- `render.smoothing.interpolate` controls whether source-frame annotations are generated between analyzed frames. `render.smoothing.interpolation_method: hermite` is the default and uses monotone cubic Hermite interpolation through tracked keyframes to reduce overshoot. `linear` uses straight-line fills, while `polynomial` uses an arbitrary local polynomial fit and is mainly experimental. `polynomial_order` affects only polynomial interpolation and linear keyframe smoothing. `max_center_offset_ratio` and `max_size_delta_ratio` clamp generated boxes against the linear reference path. The `polynomial` and `hermite` methods preserve actual tracked keyframes exactly.
- Tracking uses Roboflow `trackers` BoT-SORT. Camera motion compensation is enabled by default with `tracker.cmc_method: sparseOptFlow`. Supported CMC methods are `sparseOptFlow`, `orb`, `sift`, and `ecc`. Set `tracker.enable_cmc: false` for static-camera runs where CMC hurts stability or performance.
- Rendering shows make, model, generation, and variation when available.
- `reports/report.csv` contains one row per identified vehicle. It preserves the
  detailed MMR fields and affirmative tags as boolean columns with matching
  confidence columns so downstream analytics can aggregate the CSV as needed.
