# Multiple Myeloma Digital Twin - Evaluation Framework

PhD Researcher 5: MLOps and Evaluation Expert

A comprehensive evaluation framework and MLOps infrastructure for the Multiple Myeloma digital twin prediction pipeline.

## Overview

This framework provides production-grade evaluation, monitoring, and hyperparameter optimization infrastructure for clinical time-to-event prediction models.

### Key Components

1. **Data Splits** (`splits.py`)
   - Patient-level stratified splits preventing data leakage
   - Temporal cross-validation respecting clinical timelines
   - Automated leakage detection
   - Comprehensive split audit reports

2. **Calibration** (`calibration.py`)
   - Platt scaling, isotonic regression, temperature scaling
   - Calibration curves, ECE, Hosmer-Lemeshow test
   - Train-only fitting to prevent overfitting

3. **Metrics** (`metrics.py`)
   - Time-dependent AUROC (Uno's)
   - Integrated Brier Score
   - Harrell's and Uno's concordance indices
   - Net reclassification index
   - Decision curve analysis
   - Bootstrap confidence intervals

4. **MLflow Tracking** (`mlflow_tracking.py`)
   - Experiment setup and auto-logging
   - Model registry and versioning
   - Run comparison utilities
   - Artifact management

5. **Autoresearch** (`autoresearch.py`)
   - Karpathy's autoresearch pattern
   - Preprocessing locked, only training config editable
   - Single metric optimization (AUROC)
   - Fixed wall-clock search budget
   - Random, Bayesian (Optuna), and population-based search

6. **Reproducibility** (`reproducibility.py`)
   - Dockerfile generation
   - DVC pipeline definitions
   - Seed management across frameworks
   - Environment snapshots
   - Git hash logging

7. **Reporting** (`reporting.py`)
   - Auto-generated markdown reports
   - Model comparison dashboards
   - DeLong test for statistical comparison
   - LaTeX table generation
   - Calibration and ROC curve plots

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the Docker container
docker build -f docker/Dockerfile -t mm-digital-twin:latest .
```

## Quick Start

### 1. Basic Usage

```python
from researcher4_evaluation.splits import PatientLevelSplit, TemporalCrossValidator
from researcher4_evaluation.metrics import SurvivalMetrics
from researcher4_evaluation.mlflow_tracking import ExperimentTracker

# Setup experiment tracking
tracker = ExperimentTracker(experiment_name="mm_predictions")

# Start a run
run_id = tracker.start_run("baseline_model")

# Create train/test split
splitter = PatientLevelSplit(test_size=0.2)
train_idx, test_idx = splitter.split(df, patient_col="patient_id")

# Log hyperparameters and metrics
tracker.log_params({
    "model": "logistic_regression",
    "learning_rate": 0.01,
})

# Evaluate model
auroc = SurvivalMetrics.unos_auc(time, event, predictions)
tracker.log_metrics({"auroc": auroc})

# End run
tracker.end_run()
```

### 2. Temporal Cross-Validation

```python
from researcher4_evaluation.splits import TemporalCrossValidator

cv = TemporalCrossValidator(n_splits=5, method="expanding")
folds = cv.split(df, time_col="visit_date", patient_col="patient_id")

for train_idx, valid_idx in folds:
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    # Train and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_valid)[:, 1]
```

### 3. Automated Hyperparameter Search

```python
from researcher4_evaluation.autoresearch import AutoresearchHarness, ConfigurationSpace

# Define search space
config = ConfigurationSpace(
    learning_rate=(1e-4, 1e-1),
    batch_size=(16, 256),
    n_epochs=(10, 100),
    search_strategy="bayesian",
    max_wall_clock_hours=24.0,
)

# Define training and evaluation functions
def train_fn(params, X, y):
    model = LogisticRegression(C=1/params['l2_regularization'])
    model.fit(X, y)
    return model

def eval_fn(model, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, model.predict_proba(X)[:, 1])

# Run autoresearch
harness = AutoresearchHarness(config, train_fn, eval_fn)
results = harness.search(X_train, y_train, X_valid, y_valid)
harness.save_results()
```

### 4. Model Calibration

```python
from researcher4_evaluation.calibration import (
    PlattScaler,
    IsotonicCalibrator,
    TemperatureScaler,
    CalibrationAnalyzer,
)

# Fit calibration on training data
platt = PlattScaler()
platt.fit(y_train, y_pred_train)

# Apply to test predictions
calibrated_pred = platt.predict_proba(y_pred_test)

# Evaluate calibration
metrics = CalibrationAnalyzer.evaluate(y_test, calibrated_pred)
print(f"ECE: {metrics.ece:.4f}")
print(f"Hosmer-Lemeshow p-value: {metrics.hl_pvalue:.4f}")
```

### 5. Split Validation

```python
from researcher4_evaluation.splits import LeakageDetector

detector = LeakageDetector(time_col="timestamp", patient_col="patient_id")

# Check for leakage
leakage_detected, details = detector.detect_leakage(df_train, df_valid)

# Generate audit report
report = detector.audit_split(df, folds, event_col="event")
print(report.summary_text())
report.to_json("split_audit.json")
```

### 6. Reporting

```python
from researcher4_evaluation.reporting import ExperimentReporter, DeLongTest

reporter = ExperimentReporter(output_dir="./reports")

# Generate plots
reporter.generate_roc_curve(y_test, y_pred, title="ROC Curve")
reporter.generate_calibration_plot(y_test, y_pred)

# Compare two models
z_stat, p_value = DeLongTest.compare(y_test, y_pred_1, y_pred_2)

# Generate comparison report
models_metrics = {
    "Model A": {"auroc": 0.85, "ece": 0.04},
    "Model B": {"auroc": 0.87, "ece": 0.03},
}
reporter.generate_model_comparison_report(models_metrics)
```

### 7. Reproducibility Setup

```python
from researcher4_evaluation.reproducibility import ReproducibilityManager

manager = ReproducibilityManager(output_dir="./reproducibility")

# Setup seeds and capture environment
manager.setup(seed=42)

# Generate Dockerfile
manager.generate_dockerfile(
    base_image="python:3.10-slim",
    requirements_file="requirements.txt"
)

# Generate DVC pipeline
manager.generate_dvc_pipeline()

# Generate summary
manager.generate_summary()
```

## Design Principles

### Autoresearch Pattern

Following Karpathy's autoresearch methodology:

1. **Preprocessing Locked**: All preprocessing is finalized and immutable
   - Ensures reproducibility
   - Prevents subtle leakage from preprocessing edits
   - Version controlled preprocessing

2. **Constrained Editable Surface**: Only training hyperparameters are tunable
   - Clear boundaries on what can be modified
   - All modifications logged and tracked
   - Explicit configuration space definition

3. **Single Metric**: Optimize AUROC (standardized, interpretable)
   - Reduces confusion from multiple metrics
   - Clear optimization objective
   - Easy to compare across runs

4. **Fixed Search Budget**: Wall-clock hours, not trials
   - Real-world constraint
   - Prevents runaway searches
   - Enables scheduling in production

5. **Complete Logging**: Every decision is recorded
   - Full experiment history
   - Reproducible results
   - Scientific rigor

### Data Leakage Prevention

Multiple layers of leakage detection:

- **Patient-level splits**: No patient appears in both train/test
- **Temporal ordering**: Future data never used for training
- **Automated checks**: Detect overlapping timeframes
- **Audit reports**: Detailed leakage analysis

### Clinical Rigor

- **Time-dependent metrics**: Handle censoring appropriately (Uno's AUROC)
- **Calibration analysis**: Ensure predicted probabilities are trustworthy
- **Decision curve analysis**: Evaluate clinical utility, not just discrimination
- **Statistical testing**: DeLong test for comparing models

## File Structure

```
researcher4_evaluation/
├── __init__.py                 # Package initialization
├── splits.py                   # Data splitting & leakage detection
├── calibration.py              # Probabilistic calibration
├── metrics.py                  # Time-dependent survival metrics
├── mlflow_tracking.py          # Experiment tracking & registry
├── autoresearch.py             # Hyperparameter optimization
├── reproducibility.py          # Reproducibility infrastructure
├── reporting.py                # Automated reporting
├── requirements.txt            # Python dependencies
└── README.md                   # This file

docker/
└── Dockerfile                  # Container image definition
```

## API Reference

### Key Classes

- `PatientLevelSplit`: Ensures patient-level data integrity
- `TemporalCrossValidator`: Time-aware cross-validation
- `StratifiedGroupKFold`: Stratified by event, grouped by patient
- `LeakageDetector`: Automated leakage detection and auditing
- `PlattScaler`, `IsotonicCalibrator`, `TemperatureScaler`: Calibration methods
- `CalibrationAnalyzer`: Calibration evaluation
- `SurvivalMetrics`: Time-dependent evaluation metrics
- `ExperimentTracker`: MLflow experiment management
- `ModelRegistry`: Model versioning and promotion
- `AutoresearchHarness`: Hyperparameter search framework
- `ReproducibilityManager`: Reproducibility coordination
- `ExperimentReporter`: Automated report generation
- `DeLongTest`: Statistical model comparison

## Performance Notes

- **Leakage detection**: O(n²) for patient pair comparisons
- **Metrics**: Efficiently vectorized NumPy operations
- **Hyperparameter search**: Scales with n_trials and wall-clock budget
- **Bootstrapping**: Optional for confidence intervals

## Contributing

When extending the framework:

1. Maintain immutable preprocessing philosophy
2. Add metrics to `metrics.py` for new evaluation dimensions
3. Update `__init__.py` for new exports
4. Follow type hints throughout
5. Add docstrings with clinical context

## References

- Karpathy, A. (2019). A Recipe for Training Neural Networks
- Uno, H., et al. (2003). On the C-statistics for evaluating overall performance
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines
- DeLong, E. R., et al. (1988). Comparing the Areas under Two ROC Curves
- Vickers, A. J., & Elkin, E. B. (2006). Decision Curve Analysis

## License

Internal use only - Multiple Myeloma Digital Twin Project
