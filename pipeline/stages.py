from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from config import AppConfig, CameraProfile
    from storage.run_store import RunStore


class AnalyzeStage(Protocol):
    def __call__(
        self,
        project_root: Path,
        config: "AppConfig",
        profile: "CameraProfile",
        video_path: Path,
        run_store: "RunStore",
    ) -> "RunStore": ...


class ClassifyStage(Protocol):
    def __call__(
        self, config: "AppConfig", run_store: "RunStore"
    ) -> dict[int, Any]: ...


class WriteSkippedClassificationBatchGridsStage(Protocol):
    def __call__(self, config: "AppConfig", run_store: "RunStore") -> list[Path]: ...


class RenderStage(Protocol):
    def __call__(
        self,
        config: "AppConfig",
        profile: "CameraProfile",
        video_path: Path,
        run_store: "RunStore",
        allow_unclassified_annotations: bool,
    ) -> Path: ...


class ReportStage(Protocol):
    def __call__(self, run_store: "RunStore") -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class PipelineStages:
    analyze_video: AnalyzeStage
    classify_tracks: ClassifyStage
    write_skipped_classification_batch_grids: WriteSkippedClassificationBatchGridsStage
    render_video: RenderStage
    generate_reports: ReportStage
