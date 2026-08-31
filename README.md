# Car-Census

Car counting and make/model/year identification from video footage.

## What It Does

`Car-Census` takes a video, analyzes only a configured polygon zone, tracks vehicles with BoT-SORT, selects the best crops per track, reuses safe results from a shared retrieval cache, sends unresolved crops to TrafficEye/Eyedea for make/model recognition, then renders a clean annotated video with offline-smoothed boxes and exports a detailed vehicle CSV.

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

## Configuration

[`configs/default.yaml`](configs/default.yaml) is the authoritative source for
all application defaults. Pydantic validates the resulting configuration but
does not provide a second set of operational defaults.

Pass `--config PATH` to any supported command to overlay only the settings you
want to change. Values are resolved in this order, with later sources taking
precedence:

```text
configs/default.yaml < custom --config file < accelerator/device CLI options
```

This means a custom file can be small, for example:

```yaml
mmr:
  batch_size: 16
  batch_grid_columns: 4
```

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

Classification sends one selected crop per vehicle to TrafficEye when a safe
retrieval result is unavailable. The default is one crop per request
(`mmr.batch_size: 1`, `mmr.batch_grid_columns: 1`). Each request supplies a
manual BOX covering the crop and requests only `MMR`; `DETECTION`, OCR, and
plate detection are not requested.

TrafficEye responses are stored as `api_confirmed` evidence rather than
immutable ground truth. Exact image/request matches are reusable across runs.
Approximate retrieval uses OpenRouter's multimodal
`google/gemini-embedding-2` image embeddings at 768 dimensions, combined with
an independent perceptual hash gate. The default is `enforce`; a near match is
reused only when both gates pass, and the cosine-distance gate uses the
threshold produced by `cache calibrate` (see below). Until a calibration
artifact exists, enforce mode fails closed and falls back to TrafficEye for
near matches. Near matches reuse make/model/generation and a variation only
when the nearby evidence agrees; color, view, tags, and detection metadata are
not copied from a similar image.

Set the OpenRouter key before creating new embeddings:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Embedding responses are cached by image SHA, model, and dimension under
`.mmr-cache/embeddings`. If OpenRouter is unavailable, TrafficEye still runs
and the API result is retained for exact image/request reuse without an
embedding.

The shared store location is configured relative to `project.output_root` with
`project.retrieval_cache_dir` and defaults to `.mmr-cache`.

The shared cache is organized by purpose:

- `responses/` stores raw TrafficEye response-cache entries.
- `embeddings/` stores one durable embedding response per image/model/dimension.
- `retrieval/` stores auditable records, source images, and lookup audit events.
- `batch_grids/` is created only when composite MMR batching is used.

Move legacy response JSON files from the cache root into `responses/` with:

```bash
Car-Census cache organize
```

To seed the cache from selected existing runs without making TrafficEye
classification calls (embedding requests may be made):

```bash
Car-Census cache seed \
  output/full-frame-IMG_5383_1440-20260605T181928Z \
  output/full-frame-IMG_5386_1440-20260605T190145Z \
  output/IMG_5458_1440-IMG_5458_1440-20260609T183644Z \
  output/IMG_5512_1440-IMG_5512_1440-20260610T171931Z \
  output/IMG_5581_1440-IMG_5581_1440-20260611T070923Z
```

This imports only accepted labels and their source crops. Unaccepted labels or
labels whose crop is missing are reported and skipped. Use `--cache-dir PATH`
to override the configured destination.

To compact an already-seeded cache and normalize legacy batch coordinates in
stored evidence, run:

```bash
python -m car_census_cli cache compact
```

Legacy retrieval records are never mutated during embedding migration. Re-embed
their retained image bytes and create auditable superseding records with:

```bash
Car-Census cache migrate-embeddings
```

Before relying on enforce-mode reuse, compare same-identity and conflicting-
identity distances:

```bash
Car-Census cache calibrate
```

Calibration reports a usable threshold only when the configured minimum evidence
exists and same-identity distances remain strictly below conflicting-identity
distances. On success it persists the threshold as `retrieval/calibration.json`
in the shared cache, and runtime retrieval uses that calibrated value as the
cosine-distance gate. It exits unsuccessfully when evidence is insufficient or
overlapping, leaves any previous artifact in place, and enforcement fails
closed rather than reusing the configured threshold.

Composite batching is opt-in. For example, set `mmr.batch_size: 16` and
`mmr.batch_grid_columns: 4` for a 4x4 grid. Composite requests likewise supply
one manual BOX per occupied grid cell and request only `MMR`. Composite images
are saved under `mmr/batch_grids/` with JSON sidecars mapping source crop paths
to grid cells. The full TrafficEye response is preserved under
`mmr/labels.json[*].raw`; common MMR fields such as make, model, generation,
color, tags, and the selected detection box are also promoted to typed label
fields.

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

Write analysis to an explicit run directory:

```bash
Car-Census analyze input_data/test_vid.MP4 --camera-id home-road --run-dir output/home-road-test
```

If that directory already contains a valid run for the same video and camera
profile, rerun analysis with `--overwrite`. The replacement is staged first, so
an analysis failure leaves the previous run intact. A successful overwrite
replaces the whole run and intentionally removes its old classification,
render, smoothing, and report artifacts.

```bash
Car-Census analyze input_data/test_vid.MP4 --camera-id home-road --run-dir output/home-road-test --overwrite
```

Artifacts are written to `output/<run-id>/`. Automatically generated run IDs
use the video name and a compact UTC timestamp:

```text
IMG_5581_1440--20260611-070923Z
IMG_5386_1440_20s--20260605-124655Z
IMG_5383_1440_20s--camera-IMG_5458_1440--20260609-155031Z
```

The camera component is omitted for full-frame analysis and when it already
matches the video name. If multiple runs start during the same second, later
runs receive readable suffixes such as `--02` and `--03`. Existing output
directories are not migrated or renamed. Explicit `--run-dir` names continue
to be used exactly as provided.

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

output/.mmr-cache/
  <request-hash>.json
  embeddings/
    <image-sha>-<model-hash>-<dimensions>.json
  retrieval/
    records/
    images/
    lookup_audit.jsonl
```

The `.mmr-cache` directory is shared by runs created under the same output
root. It retains the original crop bytes, request contract, raw API response,
and visual representation needed to audit or re-embed retrieval decisions.

## Notes

- Input videos are expected to be 30 fps by default. `video.fps` controls source-frame timestamps and render output FPS; `video.fps_tolerance` controls how much OpenCV-reported input FPS drift is accepted before the run fails.
- Analysis can run at a lower cadence for faster tracking. Set `analysis.fps` to the desired tracking rate, such as `10`; rendering still processes every decoded input frame and writes at `video.fps`.
- Counting uses tracked vehicles inside the configured polygon zone. Camera profiles with `count_line` use Supervision's finite `LineZone` with the bounding-box bottom center as the crossing anchor.
- By default, `tracker.ignore_edge_touches: true` ignores detections and tracker outputs whose boxes touch the source-frame edge or selected camera crop edge. Increase `tracker.edge_margin_px` to ignore boxes that are near, but not exactly on, the edge.
- `analysis/frames.jsonl` contains raw tracker output. `analysis/render_frames.jsonl` is generated for annotation only and does not affect counts, crops, or make/model classification.
- Render smoothing generates `analysis/render_frames.jsonl` for annotation only. By default, `render.smoothing.observed_box_smoothing: local_linear` fits short centered linear motion per track over box center and size, using `render.smoothing.observed_smoothing_window: 5` and clamping each adjustment to `render.smoothing.observed_smoothing_max_shift_ratio: 0.10` of the raw box. Set `observed_box_smoothing: none` to preserve raw observed boxes exactly, or `observed_box_smoothing: causal_average` to opt into legacy moving-average smoothing with `supervision.DetectionsSmoother`; `render.smoothing.history_length` only applies in causal-average mode and can visibly lag behind moving objects. Bridge and interpolation records are render-only and do not affect counts, crops, make/model classification, raw tracks, or reports. Generated interpolation frames may carry display state such as `counted`, but count events remain anchored to raw analysis records. `render.smoothing.max_interpolation_gap_seconds` controls source-frame gap bridging; `null` uses a cadence-based default. No tail extrapolation is generated.
- Tracking uses Roboflow `trackers` BoT-SORT. Camera motion compensation is disabled by default because the expected camera setup is static. For moving cameras, set `tracker.enable_cmc: true` and choose a `tracker.cmc_method`; supported CMC methods are `sparseOptFlow`, `orb`, `sift`, and `ecc`.
- The installed BoT-SORT implementation associates tracks using Kalman motion, IoU, and detection confidence; it does not use appearance/ReID features. `tracker.max_reassociation_gap_seconds` retires IDs that return after a longer absence instead of allowing a stale ID to attach to another vehicle. Set it to `null` to disable this guard.
- World-space (homography) reassociation gate: when a camera profile defines a `homography` with at least 4 matching pixel-to-world point pairs, a track ID returning within `tracker.world_reassociation_max_gap_seconds` is retained only if its road-plane displacement is physically plausible — `distance_m <= min(tracker.world_reassociation_max_distance_m, tracker.world_reassociation_max_speed_mps * gap_seconds)`. Gaps beyond the world ceiling, or runs without a calibrated profile, fall back to strict retirement. Master switch: `tracker.world_reassociation_enabled` (`true` by default; inert without calibration). Diagnostics counters `world_reassociation_observations_accepted` and `world_reassociation_tracks_retained` are reported in detection stats.
- Post-hoc sequential-duplicate merges respect the same physics: when a homography is available and `tracker.sequential_duplicate_max_implied_speed_mps` is set, a track pair is merged only if the implied road-plane speed between endpoints stays within the limit. `analysis/sequential_duplicates.json` records `world_gate_active` plus per-pair `world_handoff_distance_m` / `implied_speed_mps` metrics.
- With a calibrated homography, each track summary in `tracks.jsonl` also carries road-plane speed statistics (`speed_mps_median`, `speed_mps_max`) computed from consecutive observations, so detection dropouts yield the average speed across the gap.
- Camera profiles live in `configs/cameras/<camera_id>.yaml` and support an optional `homography` block (`source_points` in pixels on the road plane, `target_points` in meters). Pick well-separated, non-collinear reference points whose real-world positions you know; profiles ship with a commented template and no active calibration.
- RF-DETR's default `0.15` confidence floor is deliberately lower than BoT-SORT's `0.30` high-confidence threshold, enabling the tracker's native low-confidence second association pass. Before tracking, Supervision class-agnostic NMS (`detector.nms_enabled: true`, `detector.nms_iou_threshold: 0.80`) removes highly overlapping detections even when RF-DETR assigns competing labels such as `car` and `truck`. Set `detector.nms_enabled: false` for comparison runs.
- `tracker.lost_track_buffer` is a 30-FPS-equivalent value that the `trackers` package scales by the analysis FPS; it is not a direct count of analysis frames. The default `60` keeps IDs alive through ~2 s of missed analysis frames at 10 analysis FPS, matching `tracker.world_reassociation_max_gap_seconds`: the buffer decides how long BoT-SORT tries to re-attach an ID, while the world gate decides whether each re-attachment is physically safe. Extending the buffer further without also extending the world ceiling only widens the window the gate polices.
- Rendering shows make, model, generation, and variation when available.
- The live render overlay shows the top counted makes with origin flags and
  proportional bars once accepted MMR labels are available.
- `report.csv` contains one row per identified vehicle. It preserves the
  detailed MMR fields and affirmative tags as boolean columns with matching
  confidence columns so downstream analytics can aggregate the CSV as needed.
