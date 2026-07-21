from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Literal
from uuid import uuid4

from storage.run_store import RunStore

logger = logging.getLogger(__name__)


class RunDirectoryError(ValueError):
    """Raised when an explicit analysis run directory is unsafe to use."""


class AnalysisRunTransaction:
    def __init__(
        self,
        *,
        run_dir: Path,
        overwrite: bool,
        video_path: Path,
        camera_id: str,
    ) -> None:
        self.target = run_dir.expanduser().absolute()
        self.overwrite = overwrite
        self.video_path = video_path.expanduser().resolve()
        self.camera_id = camera_id
        self._staging_root: Path | None = None

    def __enter__(self) -> RunStore:
        self._validate_target()
        self.target.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{self.target.name}.staging-",
                dir=self.target.parent,
            )
        )
        self._staging_root = staging_root
        store = RunStore(staging_root)
        store.ensure_directories()
        return store

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        _ = exc_value, traceback
        if exc_type is not None:
            self._remove_staging()
            return False

        staging_root = self._staging_root
        if staging_root is None:
            raise RuntimeError("Analysis run transaction was not started")

        try:
            staging_store = RunStore(staging_root)
            staging_store.validate_analysis_artifacts()
            staging_store.rebase_analysis_paths(self.target)
            self._promote(staging_root)
        except BaseException:
            self._remove_staging()
            raise
        self._staging_root = None
        return False

    def _validate_target(self) -> None:
        if self.target.is_symlink():
            raise RunDirectoryError(
                f"Run directory must not be a symlink: {self.target}"
            )
        if not self.target.exists():
            return
        if not self.target.is_dir():
            raise RunDirectoryError(f"Run path is not a directory: {self.target}")
        if next(self.target.iterdir(), None) is None:
            return

        try:
            existing_store = RunStore.from_existing(self.target)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise RunDirectoryError(
                "Refusing to replace a nonempty directory that is not a valid "
                f"Car-Census run: {self.target}"
            ) from exc

        if not self.overwrite:
            raise RunDirectoryError(
                f"Run directory already contains a completed run: {self.target}. "
                "Pass --overwrite to replace it."
            )

        manifest = existing_store.manifest.read()
        existing_video = manifest.video_path.expanduser().resolve()
        if existing_video != self.video_path:
            raise RunDirectoryError(
                "Refusing to overwrite a run created for a different video: "
                f"{manifest.video_path}"
            )
        if manifest.camera_id != self.camera_id:
            raise RunDirectoryError(
                "Refusing to overwrite a run created for a different camera "
                f"profile: {manifest.camera_id!r}"
            )

    def _promote(self, staging_root: Path) -> None:
        backup_root: Path | None = None
        if self.target.exists():
            backup_root = self.target.with_name(
                f".{self.target.name}.backup-{uuid4().hex}"
            )
            self.target.rename(backup_root)

        try:
            staging_root.rename(self.target)
        except BaseException:
            if backup_root is not None and backup_root.exists():
                try:
                    backup_root.rename(self.target)
                except OSError as restore_error:
                    raise RuntimeError(
                        "Failed to promote the staged run and failed to restore "
                        f"the prior run. Backup remains at: {backup_root}"
                    ) from restore_error
            raise

        if backup_root is not None:
            try:
                shutil.rmtree(backup_root)
            except OSError:
                logger.warning(
                    "New analysis was promoted, but the prior run backup could "
                    "not be removed: %s",
                    backup_root,
                    exc_info=True,
                )

    def _remove_staging(self) -> None:
        if self._staging_root is not None and self._staging_root.exists():
            shutil.rmtree(self._staging_root, ignore_errors=True)
        self._staging_root = None
