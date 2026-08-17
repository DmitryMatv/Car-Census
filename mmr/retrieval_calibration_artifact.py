from __future__ import annotations

from pathlib import Path

import orjson
from pydantic import BaseModel


class RetrievalCalibrationArtifact(BaseModel):
    schema_version: int
    created_at: str
    embedding_model: str
    embedding_dimensions: int
    phash_max_hamming_distance: int
    threshold: float
    same_identity_pairs: int
    conflicting_identity_pairs: int
    maximum_same_identity_distance: float
    minimum_conflicting_identity_distance: float


def calibration_artifact_path(store_root: Path) -> Path:
    return store_root / "calibration.json"


def save_calibration_artifact(
    store_root: Path, artifact: RetrievalCalibrationArtifact
) -> None:
    store_root.mkdir(parents=True, exist_ok=True)
    path = calibration_artifact_path(store_root)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_bytes(
        orjson.dumps(artifact.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    temporary_path.replace(path)


def load_calibration_artifact(
    store_root: Path,
    *,
    embedding_model: str,
    embedding_dimensions: int,
    phash_max_hamming_distance: int,
) -> RetrievalCalibrationArtifact | None:
    path = calibration_artifact_path(store_root)
    if not path.exists():
        return None
    try:
        artifact = RetrievalCalibrationArtifact.model_validate(
            orjson.loads(path.read_bytes())
        )
    except (OSError, ValueError):
        return None
    if (
        artifact.embedding_model != embedding_model
        or artifact.embedding_dimensions != embedding_dimensions
        or artifact.phash_max_hamming_distance != phash_max_hamming_distance
    ):
        return None
    return artifact
