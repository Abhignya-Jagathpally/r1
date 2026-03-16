"""
CLI for Running Baseline Model Comparison

Trains and evaluates all baseline models, outputs comparison table,
and logs results to MLflow.

Usage:
    python run_baselines.py --data-path data/myeloma.csv --output-dir results/
    python run_baselines.py --help
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .model_registry import ModelRegistry
from .training import BaselineTrainer
from .evaluation import BaselineEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(
    data_path: str,
    target_col: str = "progression",
    time_col: str = "follow_up_months",
    patient_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict, Optional[np.ndarray]]:
    """
    Load data from CSV file.

    Args:
        data_path: Path to CSV file
        target_col: Binary target column name
        time_col: Time/duration column name
        patient_col: Patient ID column name (for group splitting)

    Returns:
        Tuple of (features_df, labels_dict, patient_ids)

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If required columns missing
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples, {len(df.columns)} features")

    # Validate required columns
    required_cols = [target_col, time_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract features
    feature_cols = [col for col in df.columns if col not in required_cols and col != patient_col]
    X = df[feature_cols].copy()

    # Extract labels
    y = {
        "event": df[target_col].values.astype(int),
        "time": df[time_col].values.astype(float),
    }

    # Extract patient IDs if available
    patient_ids = None
    if patient_col and patient_col in df.columns:
        patient_ids = df[patient_col].values
        logger.info(f"Using {len(np.unique(patient_ids))} unique patients for group splitting")

    logger.info(f"Features: {len(feature_cols)}, Positive events: {np.sum(y['event'])}")

    return X, y, patient_ids


def run_baselines(
    data_path: str,
    output_dir: str = "results",
    cv_folds: int = 5,
    use_mlflow: bool = True,
    model_names: Optional[list] = None,
    target_col: str = "progression",
    time_col: str = "follow_up_months",
    patient_col: Optional[str] = None,
) -> Dict:
    """
    Run baseline model comparison.

    Args:
        data_path: Path to CSV file
        output_dir: Directory for output files
        cv_folds: Number of cross-validation folds
        use_mlflow: Whether to log to MLflow
        model_names: List of models to run (None = all)
        target_col: Binary target column name
        time_col: Time column name
        patient_col: Patient ID column name

    Returns:
        Dictionary with results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Load data
    X, y, patient_ids = load_data(
        data_path,
        target_col=target_col,
        time_col=time_col,
        patient_col=patient_col,
    )

    # Initialize trainer and evaluator
    registry = ModelRegistry()
    trainer = BaselineTrainer(
        registry=registry,
        cv_splits=cv_folds,
        patient_level_splits=(patient_ids is not None),
        use_mlflow=use_mlflow,
        mlflow_experiment="myeloma_baselines",
    )
    evaluator = BaselineEvaluator()

    # Select models
    if model_names is None:
        model_names = list(registry.list_models().keys())
    logger.info(f"Running baselines: {', '.join(model_names)}")

    # Cross-validate all models
    logger.info("\n" + "=" * 100)
    logger.info("CROSS-VALIDATING BASELINE MODELS")
    logger.info("=" * 100)

    cv_results = trainer.cross_validate_all(
        X=X,
        y=y,
        patient_ids=patient_ids,
        model_names=model_names,
        mlflow_run_name="baseline_cv_comparison",
    )

    # Compile results
    summary_rows = []
    for model_name, cv_result in cv_results.items():
        row = {
            "Model": model_name,
            "Mean_AUROC": cv_result.get("mean_auroc", np.nan),
            "Std_AUROC": cv_result.get("std_auroc", np.nan),
            "Mean_Brier": cv_result.get("mean_brier", np.nan),
            "Std_Brier": cv_result.get("std_brier", np.nan),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Log comparison table
    logger.info("\n" + "=" * 100)
    logger.info("BASELINE MODEL COMPARISON (Cross-Validation Results)")
    logger.info("=" * 100)
    logger.info("Benchmark: AUROC 0.78 ± 0.02 at 3-month\n")
    logger.info(summary_df.to_string(index=False))
    logger.info("=" * 100 + "\n")

    # Save summary
    summary_path = output_path / "baseline_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    # Train final models on full data
    logger.info("\n" + "=" * 100)
    logger.info("TRAINING FINAL MODELS")
    logger.info("=" * 100)

    final_models = trainer.train_all_baselines(
        X_train=X,
        y_train=y,
        model_names=model_names,
        mlflow_run_name="baseline_final_training",
    )

    # Save models
    import pickle

    models_path = output_path / "models"
    models_path.mkdir(exist_ok=True)

    for model_name, (model, metrics) in final_models.items():
        model_file = models_path / f"{model_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} to {model_file}")

    return {
        "cv_results": cv_results,
        "summary_df": summary_df,
        "final_models": final_models,
        "output_dir": str(output_path),
    }


def main():
    """Command-line interface for baseline comparison."""
    parser = argparse.ArgumentParser(
        description="Run baseline model comparison for Multiple Myeloma progression prediction"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to CSV data file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to run (default: all). Options: LOCF, MovingAverage, CoxPH, "
             "RandomSurvivalForest, XGBoost, CatBoost, LogisticRegression, TabPFN",
    )

    parser.add_argument(
        "--target-col",
        type=str,
        default="progression",
        help="Binary target column name (default: progression)",
    )

    parser.add_argument(
        "--time-col",
        type=str,
        default="follow_up_months",
        help="Time/duration column name (default: follow_up_months)",
    )

    parser.add_argument(
        "--patient-col",
        type=str,
        default=None,
        help="Patient ID column for group-based splitting (optional)",
    )

    args = parser.parse_args()

    # Run baselines
    results = run_baselines(
        data_path=args.data_path,
        output_dir=args.output_dir,
        cv_folds=args.cv_folds,
        use_mlflow=not args.no_mlflow,
        model_names=args.models,
        target_col=args.target_col,
        time_col=args.time_col,
        patient_col=args.patient_col,
    )

    logger.info(f"\nResults saved to {results['output_dir']}")
    logger.info("Baseline comparison complete!")


if __name__ == "__main__":
    main()
