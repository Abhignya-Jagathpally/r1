"""
MLflow integration for experiment tracking and model registry.

Provides:
  - ExperimentTracker: Automatic logging of runs, parameters, metrics
  - ModelRegistry: Model versioning, staging, production transitions
  - Run comparison utilities
  - Artifact management
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path

import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Container for run metadata."""

    run_id: str
    experiment_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]
    status: str
    start_time: str
    end_time: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentTracker:
    """
    MLflow experiment tracking with auto-logging and utilities.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "file:./mlruns",
        backend_store_uri: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI
            backend_store_uri: Backend store URI for artifact and metadata storage
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = None
        self.experiment_id = None
        self.current_run = None

        self._setup_mlflow(tracking_uri, backend_store_uri)

    def _setup_mlflow(
        self,
        tracking_uri: str,
        backend_store_uri: Optional[str] = None,
    ) -> None:
        """Configure MLflow."""
        mlflow.set_tracking_uri(tracking_uri)

        if backend_store_uri:
            mlflow.set_artifact_uri(backend_store_uri)

        self.client = MlflowClient(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = self.client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id
        except AttributeError:
            self.experiment_id = self.client.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow experiment '{self.experiment_name}' initialized")

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Optional tags for the run

        Returns:
            run_id
        """
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags or {},
        )
        logger.info(f"Started run: {run_name} ({self.current_run.info.run_id})")

        return self.current_run.info.run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if self.current_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended run: {self.current_run.info.run_id}")
            self.current_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_metric_history(
        self,
        metric_name: str,
        values: List[float],
    ) -> None:
        """Log metric history across steps."""
        for step, value in enumerate(values):
            mlflow.log_metric(metric_name, value, step=step)
        logger.debug(f"Logged {len(values)} steps for metric '{metric_name}'")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file or directory.

        Args:
            path: Local file or directory path
            artifact_path: Subdirectory in artifact storage
        """
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
        logger.debug(f"Logged artifact: {path}")

    def log_dict(
        self,
        data: Dict,
        filename: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log a dictionary as JSON artifact.

        Args:
            data: Dictionary to log
            filename: Filename for the JSON file
            artifact_path: Subdirectory in artifact storage
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            mlflow.log_artifact(str(filepath), artifact_path=artifact_path)
        logger.debug(f"Logged dict to {filename}")

    def log_model(
        self,
        model,
        model_name: str,
        model_type: str = "sklearn",
    ) -> None:
        """
        Log a model.

        Args:
            model: Trained model
            model_name: Name for the model
            model_type: Model framework ('sklearn', 'xgboost', 'keras', etc.)
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path=model_name)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path=model_name)
        elif model_type == "keras":
            mlflow.keras.log_model(model, artifact_path=model_name)
        else:
            mlflow.log_artifact(str(model), artifact_path=model_name)

        logger.info(f"Logged model: {model_name}")

    def get_run_info(self, run_id: Optional[str] = None) -> RunInfo:
        """
        Get information about a run.

        Args:
            run_id: Run ID (default: current run)

        Returns:
            RunInfo object
        """
        if run_id is None:
            if self.current_run is None:
                raise ValueError("No active run")
            run_id = self.current_run.info.run_id

        run = self.client.get_run(run_id)

        return RunInfo(
            run_id=run.info.run_id,
            experiment_name=self.experiment_name,
            params=run.data.params,
            metrics=run.data.metrics,
            tags=run.data.tags,
            artifacts=[],  # Would require additional API calls
            status=run.info.status,
            start_time=datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            end_time=(
                datetime.fromtimestamp(run.info.end_time / 1000).isoformat()
                if run.info.end_time
                else None
            ),
        )

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs

        Returns:
            DataFrame with run comparison
        """
        runs_data = []

        for run_id in run_ids:
            run = self.client.get_run(run_id)

            row = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                **run.data.params,
                **{f"metric_{k}": v for k, v in run.data.metrics.items()},
            }
            runs_data.append(row)

        df = pd.DataFrame(runs_data)
        logger.info(f"Comparison of {len(run_ids)} runs generated")

        return df

    def get_best_run(
        self,
        metric_name: str,
        mode: str = "max",
    ) -> Optional[RunInfo]:
        """
        Get best run in experiment.

        Args:
            metric_name: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            RunInfo for best run, or None if no runs found
        """
        filter_string = f"metrics.{metric_name} < 100000"  # Dummy filter to get all runs
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
        )

        if not runs:
            return None

        best_run = None
        best_value = -np.inf if mode == "max" else np.inf

        for run in runs:
            if metric_name not in run.data.metrics:
                continue

            value = run.data.metrics[metric_name]

            if (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
                best_value = value
                best_run = run

        if best_run is None:
            return None

        return self.get_run_info(best_run.info.run_id)

    def list_runs(self, limit: int = 100) -> pd.DataFrame:
        """
        List all runs in experiment.

        Args:
            limit: Maximum number of runs to return

        Returns:
            DataFrame of run information
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=limit,
        )

        runs_data = []
        for run in runs:
            row = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                **run.data.params,
                **{f"metric_{k}": v for k, v in run.data.metrics.items()},
            }
            runs_data.append(row)

        df = pd.DataFrame(runs_data)
        logger.info(f"Listed {len(df)} runs")

        return df


class ModelRegistry:
    """
    MLflow Model Registry for model versioning and promotion.
    """

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """Initialize model registry."""
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str,
        description: str = "",
    ) -> str:
        """
        Register a model from a run.

        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact within run
            model_name: Name for registered model
            description: Model description

        Returns:
            Model URI
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name} (version {result.version})")
            return result.name
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def get_model_version(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ):
        """
        Get model version metadata.

        Args:
            model_name: Registered model name
            version: Model version (default: latest)
            stage: Model stage ('Production', 'Staging', 'Archived')

        Returns:
            Model version metadata
        """
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            return versions[0] if versions else None
        else:
            return self.client.get_model_version(model_name, version)

    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str,
    ) -> None:
        """
        Promote model to a new stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ('Production', 'Staging', 'Archived')
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(f"Promoted {model_name} v{version} to {stage}")

    def list_models(self) -> List[str]:
        """Get list of registered models."""
        registered_models = self.client.list_registered_models()
        return [model.name for model in registered_models]

    def get_model_versions(self, model_name: str) -> pd.DataFrame:
        """
        Get all versions of a model.

        Args:
            model_name: Registered model name

        Returns:
            DataFrame with version information
        """
        versions = self.client.get_latest_versions(model_name)

        versions_data = []
        for version in versions:
            versions_data.append({
                "version": version.version,
                "stage": version.current_stage,
                "created_timestamp": datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
            })

        return pd.DataFrame(versions_data)


# Helper for numpy import in ExperimentTracker
import numpy as np
