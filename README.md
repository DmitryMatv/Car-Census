# car-census

Offline vehicle tracking and make/model census for roadside video.

## What It Does

`car-census` takes a video, analyzes only a configured polygon zone, tracks vehicles with ByteTrack, selects the best crops per track, sends those crops to TrafficEye/Eyedea for make/model recognition, then renders a clean annotated video and exports counts.

The pipeline is staged:

1. `roi edit` to define polygon ROI and count line
2. `analyze` to detect, track, count, and save crop candidates
3. `classify` to run TrafficEye make/model recognition
4. `render` to create the annotated output video
5. `report` to export counts

If `--camera-id` is omitted for `analyze` or `run`, the full frame is used as the ROI and a default horizontal count line is placed through the center of the image.

## Environment

This project targets a local NVIDIA GTX 970 workflow. Install PyTorch with the pinned CUDA 12.6 wheels first:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Then install project dependencies:

```bash
pip install -e .
```

## Model Setup

The default configuration expects a local Roboflow-exported detector at:

```text
weights/yolo11n.pt
```

This should be a locally runnable model artifact supported by Ultralytics, such as a `.pt` or `.onnx` export.

For quick smoke testing, you can temporarily change `detector.weights` in `configs/default.yaml` to `yolo11n.pt`.

Device selection is explicit:

```bash
car-census run test_vid.MP4 --camera-id home-road --device cuda
car-census run test_vid.MP4 --camera-id home-road --device auto
car-census run test_vid.MP4 --camera-id home-road --device cpu
```

`auto` uses GPU only when the installed PyTorch wheel supports your card. `cuda` tries to force GPU and will fail with a clear error if the current torch build does not support your GTX 970. That is what you want if you later install a legacy-compatible wheel.

On startup the detector logs whether it resolved to GPU or CPU, along with the GPU capability and the architectures shipped in the installed torch wheel.

## TrafficEye Setup

Set your API key in the environment:

```bash
export TRAFFICEYE_API_KEY=your_key_here
```

## Quick Start

Create an ROI profile:

```bash
car-census roi edit test_vid.MP4 --camera-id home-road --device cuda
```

Run the full pipeline:

```bash
car-census run test_vid.MP4 --camera-id home-road --device cuda
```

Run without make/model API calls:

```bash
car-census run test_vid.MP4 --skip-classify
```

Artifacts are written to `outputs/<run-id>/`.

## Output Layout

```text
outputs/<run-id>/
  run.json
  analysis/
    frames.jsonl
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
- Counting is line-crossing based, not raw detection based.
- Rendering uses ellipse markers and compact make/model labels by default.
