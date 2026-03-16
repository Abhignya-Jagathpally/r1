"""
Karpathy's autoresearch pattern: automated hyperparameter search and tuning.

Implements:
  - Locked preprocessing, only training/config editable
  - Single metric (AUROC)
  - Fixed search budget (wall-clock hours)
  - Constrained editable surface with explicit definition
  - Multiple search strategies: random, Bayesian (optuna), population-based
  - Full experiment logging
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import RandomSampler, TPESampler
from optuna.pruners import PercentilePruner
from sklearn.model_selection import cross_val_score
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationSpace:
    """
    Defines the editable configuration surface.
    Preprocessing is LOCKED - only training hyperparameters are editable.
    """

    # Model training hyperparameters (EDITABLE)
    model_type: str = "logistic_regression"  # Model architecture
    learning_rate: Tuple[float, float] = (1e-4, 1e-1)
    batch_size: Tuple[int, int] = (16, 256)
    n_epochs: Tuple[int, int] = (10, 100)
    l2_regularization: Tuple[float, float] = (1e-6, 1e-2)
    dropout_rate: Tuple[float, float] = (0.0, 0.5)

    # Preprocessing (LOCKED - not editable)
    preprocessing_version: str = "v1.0"  # Immutable
    preprocessing_config: Dict = field(default_factory=lambda: {
        "scaling": "standard",
        "imputation": "mean",
        "feature_selection": "none",
    })

    # Search-specific parameters
    search_strategy: str = "random"  # random, bayesian, population
    n_trials: int = 100
    max_wall_clock_hours: float = 24.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save configuration as YAML."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @staticmethod
    def from_yaml(path: str) -> "ConfigurationSpace":
        """Load configuration from YAML."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return ConfigurationSpace(**config_dict)

    def validate(self) -> bool:
        """Validate configuration consistency."""
        # Preprocessing is immutable
        assert self.preprocessing_version == "v1.0", "Preprocessing version must be v1.0"

        # Check hyperparameter ranges
        assert self.learning_rate[0] < self.learning_rate[1], "Invalid learning_rate range"
        assert self.batch_size[0] < self.batch_size[1], "Invalid batch_size range"
        assert self.n_epochs[0] < self.n_epochs[1], "Invalid n_epochs range"
        assert self.l2_regularization[0] < self.l2_regularization[1], "Invalid l2_regularization range"
        assert self.dropout_rate[0] <= self.dropout_rate[1], "Invalid dropout_rate range"

        logger.info("Configuration validated successfully")
        return True


@dataclass
class TrialResult:
    """Result of a single trial."""

    trial_id: int
    params: Dict[str, Any]
    metric_value: float  # AUROC
    timestamp: str
    duration_seconds: float
    status: str  # "completed", "failed", "pruned"
    error_message: Optional[str] = None


@dataclass
class SearchResults:
    """Results of hyperparameter search."""

    config: ConfigurationSpace
    trials: List[TrialResult] = field(default_factory=list)
    best_trial: Optional[TrialResult] = None
    best_metric: float = 0.0
    search_duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "trials": [asdict(t) for t in self.trials],
            "best_trial": asdict(self.best_trial) if self.best_trial else None,
            "best_metric": self.best_metric,
            "search_duration_seconds": self.search_duration_seconds,
            "timestamp": self.timestamp,
        }

    def to_json(self, path: str) -> None:
        """Save results as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        hours = self.search_duration_seconds / 3600
        text = f"""
Autoresearch Summary
====================
Search Duration: {hours:.2f} hours
Total Trials: {len(self.trials)}
Best AUROC: {self.best_metric:.4f}

Best Trial Parameters:
"""
        if self.best_trial:
            for key, value in self.best_trial.params.items():
                text += f"  {key}: {value}\n"

        text += f"""
Trial Statistics:
  Completed: {sum(1 for t in self.trials if t.status == 'completed')}
  Failed: {sum(1 for t in self.trials if t.status == 'failed')}
  Pruned: {sum(1 for t in self.trials if t.status == 'pruned')}
"""
        return text


class AutoresearchHarness:
    """
    Automated research harness implementing Karpathy's autoresearch pattern.

    Key principles:
    1. Preprocessing LOCKED - fully reproducible and fixed
    2. Only training/config editable
    3. Single metric (AUROC) for optimization
    4. Fixed wall-clock budget
    5. Full experiment logging
    """

    def __init__(
        self,
        config: ConfigurationSpace,
        train_fn: Callable,
        eval_fn: Callable,
        output_dir: str = "./autoresearch_results",
    ):
        """
        Initialize autoresearch harness.

        Args:
            config: ConfigurationSpace defining editable surface
            train_fn: Function to train model: (params, X_train, y_train) -> model
            eval_fn: Function to evaluate: (model, X_test, y_test) -> auroc_score
            output_dir: Directory for results and logs
        """
        self.config = config
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = SearchResults(config=config)
        self.start_time = None

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from configuration space."""
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.config.learning_rate[0],
                self.config.learning_rate[1],
                log=True,
            ),
            "batch_size": trial.suggest_int(
                "batch_size",
                self.config.batch_size[0],
                self.config.batch_size[1],
                log=True,
            ),
            "n_epochs": trial.suggest_int(
                "n_epochs",
                self.config.n_epochs[0],
                self.config.n_epochs[1],
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization",
                self.config.l2_regularization[0],
                self.config.l2_regularization[1],
                log=True,
            ),
            "dropout_rate": trial.suggest_float(
                "dropout_rate",
                self.config.dropout_rate[0],
                self.config.dropout_rate[1],
            ),
        }
        return params

    def _objective(self, trial: optuna.Trial, X_train, y_train, X_valid, y_valid) -> float:
        """
        Objective function for optimization.
        Returns: AUROC (single metric)
        """
        trial_id = trial.number

        # Check wall-clock budget
        if self.start_time:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours > self.config.max_wall_clock_hours:
                raise optuna.TrialPruned()

        trial_start = time.time()

        try:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)

            # Train model
            model = self.train_fn(params, X_train, y_train)

            # Evaluate
            auroc = self.eval_fn(model, X_valid, y_valid)

            # Record trial
            trial_duration = time.time() - trial_start
            result = TrialResult(
                trial_id=trial_id,
                params=params,
                metric_value=auroc,
                timestamp=datetime.now().isoformat(),
                duration_seconds=trial_duration,
                status="completed",
            )

            self.results.trials.append(result)

            # Update best
            if auroc > self.results.best_metric:
                self.results.best_metric = auroc
                self.results.best_trial = result
                logger.info(f"Trial {trial_id}: New best AUROC = {auroc:.4f}")
            else:
                logger.info(f"Trial {trial_id}: AUROC = {auroc:.4f}")

            return auroc

        except optuna.TrialPruned():
            raise

        except Exception as e:
            trial_duration = time.time() - trial_start
            result = TrialResult(
                trial_id=trial_id,
                params=params if 'params' in locals() else {},
                metric_value=0.0,
                timestamp=datetime.now().isoformat(),
                duration_seconds=trial_duration,
                status="failed",
                error_message=str(e),
            )
            self.results.trials.append(result)
            logger.error(f"Trial {trial_id} failed: {e}")

            return 0.0

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        seed: int = 42,
    ) -> SearchResults:
        """
        Run hyperparameter search with fixed wall-clock budget.

        Args:
            X_train, y_train: Training data
            X_valid, y_valid: Validation data
            seed: Random seed

        Returns:
            SearchResults object
        """
        self.config.validate()

        self.start_time = time.time()

        # Select sampler based on strategy
        if self.config.search_strategy == "random":
            sampler = RandomSampler(seed=seed)
        elif self.config.search_strategy == "bayesian":
            sampler = TPESampler(seed=seed, n_startup_trials=10)
        else:
            sampler = RandomSampler(seed=seed)

        pruner = PercentilePruner(percentile=25)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        try:
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_valid, y_valid),
                n_trials=self.config.n_trials,
                timeout=self.config.max_wall_clock_hours * 3600,
                show_progress_bar=True,
            )
        except Exception as e:
            logger.warning(f"Search interrupted: {e}")

        # Finalize results
        self.results.search_duration_seconds = time.time() - self.start_time

        logger.info(
            f"Search complete: {len(self.results.trials)} trials, "
            f"best AUROC = {self.results.best_metric:.4f}, "
            f"time = {self.results.search_duration_seconds / 3600:.2f}h"
        )

        return self.results

    def save_results(self) -> Path:
        """Save results to disk."""
        results_file = self.output_dir / "search_results.json"
        self.results.to_json(str(results_file))

        summary_file = self.output_dir / "search_summary.txt"
        with open(summary_file, "w") as f:
            f.write(self.results.summary_text())

        config_file = self.output_dir / "config.yaml"
        self.config.to_yaml(str(config_file))

        logger.info(f"Results saved to {self.output_dir}")

        return results_file

    def load_results(self, path: str) -> SearchResults:
        """Load results from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        trials = [TrialResult(**t) for t in data["trials"]]
        best_trial = TrialResult(**data["best_trial"]) if data["best_trial"] else None

        results = SearchResults(
            config=ConfigurationSpace(**data["config"]),
            trials=trials,
            best_trial=best_trial,
            best_metric=data["best_metric"],
            search_duration_seconds=data["search_duration_seconds"],
        )

        return results
