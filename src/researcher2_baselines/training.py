"""
Training Loop for Baseline Models

Implements cross-validation with patient-level splits, train-only normalization,
and MLflow tracking for experiment management.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from .baselines import BaselineModel
from .model_registry import ModelRegistry
from .evaluation import BaselineEvaluator

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Training orchestration for baseline models with cross-validation,
    patient-level splitting, and MLflow integration.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        cv_splits: int = 5,
        patient_level_splits: bool = True,
        use_mlflow: bool = True,
        mlflow_experiment: str = "myeloma_baselines",
    ):
        """
        Initialize trainer.

        Args:
            registry: ModelRegistry instance (creates default if None)
            cv_splits: Number of CV folds
            patient_level_splits: Whether to split by patient ID (recommended)
            use_mlflow: Enable MLflow tracking
            mlflow_experiment: MLflow experiment name
        """
        self.registry = registry or ModelRegistry()
        self.cv_splits = cv_splits
        self.patient_level_splits = patient_level_splits
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.mlflow_experiment = mlflow_experiment
        self.evaluator = BaselineEvaluator()

        if self.use_mlflow:
            mlflow.set_experiment(mlflow_experiment)

    def split_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Dict[str, np.ndarray],
        patient_ids: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits with optional patient-level stratification.

        Args:
            X: Feature matrix
            y: Labels dictionary with 'time' and 'event'
            patient_ids: Patient identifiers for group splitting (required if patient_level_splits=True)
            random_state: Random seed

        Returns:
            List of (train_idx, val_idx) tuples

        Raises:
            ValueError: If patient_level_splits=True but patient_ids is None
        """
        n_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)

        if self.patient_level_splits:
            if patient_ids is None:
                raise ValueError(
                    "patient_level_splits=True but patient_ids not provided. "
                    "Pass patient_ids array for group-based splitting."
                )
            gkf = GroupKFold(n_splits=self.cv_splits)
            splits = list(gkf.split(X, groups=patient_ids))
        else:
            # Stratified split by event status
            event_status = y["event"]
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=random_state)
            splits = list(skf.split(X, y=event_status))

        return splits

    def train_baseline(
        self,
        model_name: str,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        **model_kwargs
    ) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Train a single baseline model.

        Args:
            model_name: Name of model from registry
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **model_kwargs: Override model hyperparameters

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"Training {model_name} baseline...")

        # Create model from registry
        model = self.registry.create(model_name, **model_kwargs)

        # Fit on training data
        model.fit(X_train, y_train)

        metrics = {"model_name": model_name, "fit_status": "success"}

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            y_pred = np.clip(y_pred, 0, 1)

            try:
                auroc = roc_auc_score(y_val["event"], y_pred)
                brier = brier_score_loss(y_val["event"], y_pred)
                metrics["val_auroc"] = auroc
                metrics["val_brier"] = brier
                logger.info(f"  Validation AUROC: {auroc:.4f}, Brier: {brier:.4f}")
            except Exception as e:
                logger.warning(f"  Could not compute validation metrics: {e}")

        return model, metrics

    def cross_validate(
        self,
        model_name: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Dict[str, np.ndarray],
        patient_ids: Optional[np.ndarray] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Run cross-validation for a baseline model.

        Args:
            model_name: Name of model from registry
            X: Feature matrix
            y: Labels dictionary
            patient_ids: Patient identifiers for group splitting
            **model_kwargs: Override model hyperparameters

        Returns:
            Dictionary with CV results and metrics
        """
        logger.info(f"Cross-validating {model_name} with {self.cv_splits} folds...")

        splits = self.split_data(X, y, patient_ids=patient_ids)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"  Fold {fold_idx + 1}/{self.cv_splits}")

            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]

            y_train = {k: v[train_idx] for k, v in y.items()}
            y_val = {k: v[val_idx] for k, v in y.items()}

            # Train and evaluate
            model, metrics = self.train_baseline(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
                **model_kwargs
            )

            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        # Aggregate results
        cv_results = {
            "model_name": model_name,
            "n_folds": self.cv_splits,
            "fold_metrics": fold_metrics,
        }

        # Compute aggregate metrics
        if "val_auroc" in fold_metrics[0]:
            aurocs = [m["val_auroc"] for m in fold_metrics]
            cv_results["mean_auroc"] = np.mean(aurocs)
            cv_results["std_auroc"] = np.std(aurocs)
            logger.info(f"Cross-validation AUROC: {cv_results['mean_auroc']:.4f} ± {cv_results['std_auroc']:.4f}")

        if "val_brier" in fold_metrics[0]:
            briers = [m["val_brier"] for m in fold_metrics]
            cv_results["mean_brier"] = np.mean(briers)
            cv_results["std_brier"] = np.std(briers)
            logger.info(f"Cross-validation Brier: {cv_results['mean_brier']:.4f} ± {cv_results['std_brier']:.4f}")

        return cv_results

    def train_all_baselines(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        model_names: Optional[List[str]] = None,
        mlflow_run_name: str = "baseline_comparison",
    ) -> Dict[str, Tuple[BaselineModel, Dict[str, Any]]]:
        """
        Train all baseline models on provided data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            model_names: List of model names to train (None = all)
            mlflow_run_name: Name for MLflow run

        Returns:
            Dictionary mapping model names to (model, metrics) tuples
        """
        if model_names is None:
            model_names = list(self.registry.list_models().keys())

        results = {}

        if self.use_mlflow:
            mlflow.start_run(run_name=mlflow_run_name)

        try:
            for model_name in model_names:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Training {model_name}")
                logger.info(f"{'=' * 60}")

                model, metrics = self.train_baseline(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )

                results[model_name] = (model, metrics)

                # Log to MLflow
                if self.use_mlflow:
                    with mlflow.start_nested_run(run_name=model_name):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(key, value)
                        mlflow.log_param("model_name", model_name)

        finally:
            if self.use_mlflow:
                mlflow.end_run()

        return results

    def cross_validate_all(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Dict[str, np.ndarray],
        patient_ids: Optional[np.ndarray] = None,
        model_names: Optional[List[str]] = None,
        mlflow_run_name: str = "baseline_cv",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Cross-validate all baseline models.

        Args:
            X: Feature matrix
            y: Labels dictionary
            patient_ids: Patient identifiers for group splitting
            model_names: List of model names to train (None = all)
            mlflow_run_name: Name for MLflow run

        Returns:
            Dictionary mapping model names to CV results
        """
        if model_names is None:
            model_names = list(self.registry.list_models().keys())

        cv_results = {}

        if self.use_mlflow:
            mlflow.start_run(run_name=mlflow_run_name)

        try:
            for model_name in model_names:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Cross-validating {model_name}")
                logger.info(f"{'=' * 60}")

                cv_result = self.cross_validate(
                    model_name,
                    X,
                    y,
                    patient_ids=patient_ids,
                )

                cv_results[model_name] = cv_result

                # Log to MLflow
                if self.use_mlflow:
                    with mlflow.start_nested_run(run_name=model_name):
                        for key, value in cv_result.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(key, value)

        finally:
            if self.use_mlflow:
                mlflow.end_run()

        return cv_results
