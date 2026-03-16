"""
Pipeline Checkpoint Manager for Model Training Traceability.

Records pipeline state at each stage: data hashes, shapes, timestamps,
git SHA, parameters, and metrics. Enables full audit trail and
reproducibility for PhD-level research.
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StageCheckpoint:
    """Immutable record of a single pipeline stage execution."""

    stage_name: str
    stage_index: int
    status: str  # "started", "completed", "failed", "skipped"
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    input_hash: str = ""
    output_hash: str = ""
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    n_patients: int = 0
    n_features: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error_message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PipelineManifest:
    """Complete traceability record for one pipeline run."""

    run_id: str
    pipeline_version: str
    started_at: str
    completed_at: str = ""
    status: str = "running"
    git_sha: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    python_version: str = ""
    platform_info: str = ""
    config_hash: str = ""
    seed: int = 42
    stages: List[StageCheckpoint] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Checkpoint manifest saved to {path}")

    @staticmethod
    def load(path: Path) -> "PipelineManifest":
        with open(path) as f:
            data = json.load(f)
        stages = [StageCheckpoint(**s) for s in data.pop("stages", [])]
        manifest = PipelineManifest(**data)
        manifest.stages = stages
        return manifest


class CheckpointTracker:
    """
    Tracks pipeline execution with per-stage checkpoints.

    Usage:
        tracker = CheckpointTracker(output_dir, run_id="exp_001")
        with tracker.stage("ingestion", index=0) as ckpt:
            df = ingest(...)
            ckpt.output_shape = list(df.shape)
            ckpt.output_hash = tracker.hash_dataframe(df)
            ckpt.metrics = {"n_rows": len(df)}
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str = "",
        pipeline_version: str = "0.1.0",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.manifest = PipelineManifest(
            run_id=self.run_id,
            pipeline_version=pipeline_version,
            started_at=datetime.now().isoformat(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform_info=platform.platform(),
            seed=seed,
        )
        self._capture_git_state()

    def _capture_git_state(self) -> None:
        """Record git SHA, branch, and dirty status."""
        try:
            self.manifest.git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            self.manifest.git_sha = "unknown"
        try:
            self.manifest.git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            self.manifest.git_branch = "unknown"
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            self.manifest.git_dirty = len(status) > 0
        except Exception:
            self.manifest.git_dirty = False

    def stage(self, name: str, index: int) -> "_StageContext":
        """Context manager for tracking a pipeline stage."""
        return _StageContext(self, name, index)

    def record_stage(self, checkpoint: StageCheckpoint) -> None:
        """Append a completed stage checkpoint to the manifest."""
        self.manifest.stages.append(checkpoint)
        self._save_incremental()

    def finalize(self, status: str = "completed") -> Path:
        """Mark the pipeline run as complete and save the final manifest."""
        self.manifest.completed_at = datetime.now().isoformat()
        self.manifest.status = status
        manifest_path = self.checkpoint_dir / f"{self.run_id}_manifest.json"
        self.manifest.save(manifest_path)
        return manifest_path

    def _save_incremental(self) -> None:
        """Save manifest after each stage for crash recovery."""
        path = self.checkpoint_dir / f"{self.run_id}_manifest.json"
        self.manifest.save(path)

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """Compute a deterministic hash of a DataFrame for traceability."""
        h = hashlib.sha256()
        h.update(str(df.shape).encode())
        h.update(str(sorted(df.columns.tolist())).encode())
        # Hash a sample of values for speed on large DataFrames
        sample_size = min(1000, len(df))
        if sample_size > 0:
            sample = df.head(sample_size)
            for col in sample.columns:
                vals = sample[col].astype(str).values
                h.update(b"".join(v.encode() for v in vals))
        return h.hexdigest()[:16]

    @staticmethod
    def hash_config(config_dict: Dict) -> str:
        """Hash a configuration dictionary for change detection."""
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def save_artifact(self, name: str, data: Any, stage_name: str = "") -> Path:
        """Save an artifact (DataFrame, dict, array) to disk."""
        artifact_dir = self.checkpoint_dir / "artifacts" / stage_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            path = artifact_dir / f"{name}.parquet"
            data.to_parquet(path, index=False, engine="pyarrow")
        elif isinstance(data, np.ndarray):
            path = artifact_dir / f"{name}.npy"
            np.save(path, data)
        elif isinstance(data, dict):
            path = artifact_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            path = artifact_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump({"value": str(data)}, f, indent=2)

        logger.debug(f"Saved artifact {name} to {path}")
        return path


class _StageContext:
    """Context manager for tracking a single pipeline stage."""

    def __init__(self, tracker: CheckpointTracker, name: str, index: int):
        self.tracker = tracker
        self.checkpoint = StageCheckpoint(
            stage_name=name,
            stage_index=index,
            status="started",
        )

    def __enter__(self) -> StageCheckpoint:
        self.checkpoint.started_at = datetime.now().isoformat()
        self._start_time = time.time()
        logger.info(f"{'='*72}")
        logger.info(f"STAGE {self.checkpoint.stage_index}: {self.checkpoint.stage_name.upper()}")
        logger.info(f"{'='*72}")
        return self.checkpoint

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start_time
        self.checkpoint.duration_seconds = round(elapsed, 2)
        self.checkpoint.completed_at = datetime.now().isoformat()

        if exc_type is not None:
            self.checkpoint.status = "failed"
            self.checkpoint.error_message = str(exc_val)
            logger.error(
                f"STAGE {self.checkpoint.stage_index} FAILED after "
                f"{elapsed:.1f}s: {exc_val}"
            )
        else:
            self.checkpoint.status = "completed"
            logger.info(
                f"STAGE {self.checkpoint.stage_index} completed in "
                f"{elapsed:.1f}s "
                f"[output: {self.checkpoint.output_shape}]"
            )

        self.tracker.record_stage(self.checkpoint)
        return False  # Do not suppress exceptions
