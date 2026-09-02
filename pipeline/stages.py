from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import AppConfig, CameraProfile
from models import MMRResult
from storage.run_store import RunStore

AnalyzeStage = Callable[
    [Path, AppConfig, CameraProfile, Path, RunStore],
    RunStore,
]

ClassifyStage = Callable[
    [AppConfig, RunStore, CameraProfile],
    dict[int, MMRResult],
]

WriteSkippedClassificationBatchGridsStage = Callable[
    [AppConfig, RunStore],
    list[Path],
]

RenderStage = Callable[
    [AppConfig, CameraProfile, Path, RunStore, bool],
    Path,
]

ReportStage = Callable[
    [RunStore],
    dict[str, object],
]


@dataclass(frozen=True, slots=True)
class PipelineStages:
    analyze_video: AnalyzeStage
    classify_tracks: ClassifyStage
    write_skipped_classification_batch_grids: WriteSkippedClassificationBatchGridsStage
    render_video: RenderStage
    generate_reports: ReportStage
