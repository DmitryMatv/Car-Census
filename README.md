# Car-Census

Car counting and make/model/year identification from video footage.

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

This project uses local RF-DETR-M inference. Install project dependencies:

```bash
pip install -e .
```

BoT-SORT tracking uses Roboflow's `trackers` package, which is installed with
the main project dependencies.

Color country flags in annotated labels require `NotoColorEmoji.ttf` to be
installed and discoverable by Pillow. On Debian/Ubuntu systems, install the
`fonts-noto-color-emoji` package.

## Model Setup

The default detector is RF-DETR-M through Roboflow's `rfdetr` package. RF-DETR-M
uses a 576x576 detector input by default. The first run may download and cache
the pretrained COCO checkpoint through the package.

For offline reproducibility, set `detector.pretrain_weights` to a local
RF-DETR-M checkpoint path.

## Google Colab T4

For Colab GPU runs, install the project and use the Colab accelerator preset:

```bash
pip install -e .
Car-Census run input_data/test4K.MP4 --camera-id my-camera --accelerator colab-t4
```

To run analysis, classification, and report export in Colab without rendering an
annotated video, add `--skip-render`:

```bash
Car-Census run input_data/test4K.MP4 --camera-id my-camera --accelerator colab-t4 --skip-render
```

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

## Brand-Origin Flags

`mmr/data/MakeCountry.csv` maps each TrafficEye make to its historical
origin-country flag. Accepted make/model labels in annotated videos start with
that color flag. The flag is rendered separately from OpenCV text and scales
with the label according to the tracked vehicle's box width. Set
`render.label_flag_gap_px` to control the scaled horizontal gap between the flag
and make/model text.

The live render overlay uses the same catalog to show the top counted makes with
origin flags, descending counts, and proportional bars. It uses the same
accepted-label, visibility, and counting rules as the legacy live counter;
Roboflow detector and tracker behavior is unchanged.

## Powertrain Catalog

`mmr/data/MakeModelGenVar.csv` maps complete TrafficEye make, model,
generation, and variation identities to a `powertrain_class`:

- `BEV`: every factory-produced vehicle represented by the identity is battery
  electric.
- `COMBUSTION`: every represented vehicle contains a combustion engine,
  including ICE, HEV, PHEV, and range-extender vehicles.
- `MIXED`: the identity can represent both BEV and combustion-equipped factory
  variants.
- `UNKNOWN`: the identity is insufficient or includes a powertrain outside this
  taxonomy, such as a fuel-cell vehicle.

`mmr.powertrain_catalog.lookup_powertrain_class` uses an exact,
case-sensitive match across all four TrafficEye identity fields. It treats a
missing variation as an empty variation, but does not use partial or
make/model-only fallbacks. An unmatched result returns `None`, which is distinct
from a matched `UNKNOWN` identity.

Accepted `BEV` labels render all text lines in bright blue
(`render.label_bev_text_color`, default `#00BFFF`), while accepted `MIXED`
labels render in bright green (`render.label_mixed_text_color`, default
`#39FF14`). `COMBUSTION`, `UNKNOWN`, unmatched, incomplete, and unclassified
results retain `render.label_text_color` (white by default), while rejected
labels remain excluded from rendering. Country flags, label backgrounds, boxes,
and the counter keep their existing colors. Powertrain coloring requires the
same exact complete identity match as the catalog lookup.

## Quick Start

Create an ROI profile:

```bash
Car-Census roi edit input_data/test_vid.MP4 --camera-id home-road
```

Run the full pipeline:

```bash
Car-Census run input_data/test_vid.MP4 --camera-id home-road
```

Run without make/model API calls:

```bash
Car-Census run input_data/test_vid.MP4 --skip-classify
```

Run without annotated video rendering:

```bash
Car-Census run input_data/test_vid.MP4 --skip-render
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
  report.csv
```

## Notes

- Input videos are expected to be 30 fps by default. `video.fps` controls source-frame timestamps and render output FPS; `video.fps_tolerance` controls how much OpenCV-reported input FPS drift is accepted before the run fails.
- Analysis can run at a lower cadence for faster tracking. Set `analysis.fps` to the desired tracking rate, such as `10`; rendering still processes every decoded input frame and writes at `video.fps`.
- Counting uses tracked vehicles inside the configured polygon zone. Older camera profiles with `count_line` are still supported.
- By default, `tracker.ignore_edge_touches: true` ignores detections and tracker outputs whose boxes touch the source-frame edge or selected camera crop edge. Increase `tracker.edge_margin_px` to ignore boxes that are near, but not exactly on, the edge.
- `analysis/frames.jsonl` contains raw tracker output. `analysis/render_frames.jsonl` is generated for annotation only and does not affect counts, crops, or make/model classification.
- Render smoothing generates `analysis/render_frames.jsonl` for annotation only. By default, `render.smoothing.observed_box_smoothing: local_linear` fits short centered linear motion per track over box center and size, using `render.smoothing.observed_smoothing_window: 5` and clamping each adjustment to `render.smoothing.observed_smoothing_max_shift_ratio: 0.10` of the raw box. Set `observed_box_smoothing: none` to preserve raw observed boxes exactly, or `observed_box_smoothing: causal_average` to opt into legacy moving-average smoothing with `supervision.DetectionsSmoother`; `render.smoothing.history_length` only applies in causal-average mode and can visibly lag behind moving objects. Bridge and interpolation records are render-only and do not affect counts, crops, make/model classification, raw tracks, or reports. Generated interpolation frames may carry display state such as `counted`, but count events remain anchored to raw analysis records. `render.smoothing.max_interpolation_gap_seconds` controls source-frame gap bridging; `null` uses a cadence-based default. No tail extrapolation is generated.
- Tracking uses Roboflow `trackers` BoT-SORT. Camera motion compensation is disabled by default because the expected camera setup is static. For moving cameras, set `tracker.enable_cmc: true` and choose a `tracker.cmc_method`; supported CMC methods are `sparseOptFlow`, `orb`, `sift`, and `ecc`.
- The installed BoT-SORT implementation associates tracks using Kalman motion, IoU, and detection confidence; it does not use appearance/ReID features. `tracker.max_reassociation_gap_seconds` retires IDs that return after a longer absence instead of allowing a stale ID to attach to another vehicle. Set it to `null` to disable this guard.
- `tracker.lost_track_buffer` is a 30-FPS-equivalent value that the `trackers` package scales by the analysis FPS; it is not a direct count of analysis frames.
- Rendering shows make, model, generation, and variation when available.
- The live render overlay shows the top counted makes with origin flags and
  proportional bars once accepted MMR labels are available.
- `report.csv` contains one row per identified vehicle. It preserves the
  detailed MMR fields and affirmative tags as boolean columns with matching
  confidence columns so downstream analytics can aggregate the CSV as needed.
