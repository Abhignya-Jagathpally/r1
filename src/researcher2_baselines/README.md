# Multiple Myeloma Baseline Models

Production-quality baseline modeling suite for the Multiple Myeloma digital twin pipeline. Implements classical survival and risk prediction models with unified interface, cross-validation, and comprehensive evaluation.

## Features

### Implemented Baseline Models

1. **LOCF (Last Observation Carried Forward)** - Naive temporal baseline
2. **Moving Average** - Temporal aggregation with configurable window
3. **Cox Proportional Hazards** - Semi-parametric survival model (lifelines)
4. **Random Survival Forest** - Ensemble survival method (scikit-survival)
5. **XGBoost Snapshot** - Gradient boosting for binary progression (3-month)
6. **CatBoost Snapshot** - Categorical boosting for binary progression
7. **Logistic Regression** - Linear baseline for binary progression
8. **TabPFN** - Neural tabular prediction (with XGBoost fallback)

### Model Interface

All models implement a unified interface:

```python
from researcher2_baselines import ModelRegistry

# Create model from registry
registry = ModelRegistry()
model = registry.create("XGBoost", learning_rate=0.05)

# Training
model.fit(X_train, y_train)

# Prediction - risk scores (0-1)
risk_scores = model.predict(X_test)

# Calibrated probabilities at multiple horizons
proba = model.predict_proba(X_test, horizons=[3, 6, 12])
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Baseline Comparison

```bash
# Run all baselines with cross-validation
python -m researcher2_baselines.run_baselines \
    --data-path data/myeloma.csv \
    --output-dir results/ \
    --cv-folds 5

# Run specific models with patient-level splitting
python -m researcher2_baselines.run_baselines \
    --data-path data/myeloma.csv \
    --models XGBoost CatBoost CoxPH \
    --patient-col patient_id \
    --output-dir results/
```

### Usage in Python

```python
import pandas as pd
import numpy as np
from researcher2_baselines import (
    ModelRegistry,
    BaselineTrainer,
    BaselineEvaluator,
)

# Load data
df = pd.read_csv("data/myeloma.csv")
X = df[feature_cols]
y = {
    "event": df["progression"].values,
    "time": df["follow_up_months"].values,
}
patient_ids = df["patient_id"].values

# Initialize components
registry = ModelRegistry()
trainer = BaselineTrainer(registry=registry, cv_splits=5)
evaluator = BaselineEvaluator()

# Cross-validate all models
cv_results = trainer.cross_validate_all(
    X=X,
    y=y,
    patient_ids=patient_ids,
)

# Train final models
final_models = trainer.train_all_baselines(X, y)

# Evaluate
for name, (model, metrics) in final_models.items():
    y_pred = model.predict(X_test)
    results = evaluator.evaluate_model(
        y_true=y_test["event"],
        y_pred=y_pred,
        times=y_test["time"],
        events=y_test["event"],
        time_horizons=[3, 6, 12],
        model_name=name,
    )
```

## Evaluation Metrics

### Primary Metrics

- **AUROC** - Area under ROC curve with 95% bootstrap CI
- **Brier Score** - Mean squared error of probabilities
- **Calibration ECE** - Expected Calibration Error
- **C-Index** - Concordance index for survival predictions
- **Time-Dependent AUROC** - Discrimination at multiple horizons (3, 6, 12 months)

### Benchmark Performance

Target: AUROC 0.78 ± 0.02 at 3-month progression prediction

## File Structure

```
researcher2_baselines/
├── __init__.py              # Package exports
├── baselines.py             # All baseline model implementations
├── model_registry.py        # Factory pattern + hyperparameter defaults
├── training.py              # Training loop + cross-validation
├── evaluation.py            # Metrics + calibration + bootstrap CIs
├── run_baselines.py         # CLI tool
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Model Registry & Hyperparameters

The `ModelRegistry` provides:

1. **Default hyperparameters** - Vetted defaults for each model
2. **Search spaces** - Ranges for Bayesian optimization
3. **Factory method** - Create models with custom parameters

```python
from researcher2_baselines import ModelRegistry

registry = ModelRegistry()

# List all models
print(registry.list_models())

# Get default parameters
params = registry.get_default_params("XGBoost")

# Get search space for tuning
search_space = registry.get_search_space("XGBoost")

# Create model with custom params
model = registry.create("XGBoost", learning_rate=0.05, max_depth=8)

# Register custom model
registry.register(
    "MyModel",
    MyModelClass,
    default_params={"param1": value1},
    search_space=[...],
    description="My custom baseline"
)
```

## Training with Cross-Validation

The `BaselineTrainer` handles:

1. **Patient-level splitting** - Groups by patient ID to avoid data leakage
2. **Train-only normalization** - Fitted on training set only
3. **MLflow tracking** - Automatic experiment logging
4. **Multiple CV strategies** - GroupKFold or StratifiedKFold

```python
from researcher2_baselines import BaselineTrainer, ModelRegistry

trainer = BaselineTrainer(
    registry=ModelRegistry(),
    cv_splits=5,
    patient_level_splits=True,
    use_mlflow=True,
)

# Cross-validate single model
cv_results = trainer.cross_validate(
    model_name="XGBoost",
    X=X,
    y=y,
    patient_ids=patient_ids,
)

# Train all models
results = trainer.train_all_baselines(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)
```

## Calibration

All classification models use **Platt scaling** for probability calibration:

```python
# Automatic in XGBoost, CatBoost, LogisticRegression
model = registry.create("XGBoost")
model.fit(X_train, y_train)  # Internally applies Platt scaling

# Get calibrated probabilities
proba = model.predict(X_test)  # Already calibrated
```

## MLflow Integration

Results are automatically logged to MLflow:

```bash
# View results
mlflow ui
```

Logged metrics:
- AUROC / Brier for each fold
- Mean/std across folds
- Model hyperparameters
- Training duration

## Data Format

Expected input:

```python
# Features (any numeric format)
X = pd.DataFrame or np.ndarray  # (n_samples, n_features)

# Labels (dictionary format)
y = {
    "event": np.array([0, 1, 0, ...]),      # Binary progression (0/1)
    "time": np.array([3.2, 5.1, 2.8, ...])  # Follow-up in months
}

# Optional: Patient IDs for group splitting
patient_ids = np.array(["P001", "P002", ...])
```

## Missing Data Handling

- Features with NaN values should be imputed before passing to models
- No built-in imputation (recommended: use sklearn.impute.SimpleImputer)

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Numerical computing |
| pandas | ≥1.3 | Data manipulation |
| scikit-learn | ≥1.0 | ML utilities, calibration |
| scipy | ≥1.7 | Statistics |
| xgboost | ≥1.5 | Gradient boosting |
| catboost | ≥1.0 | Categorical boosting |
| lifelines | ≥0.27 | Cox PH, survival metrics |
| scikit-survival | ≥0.17 | Random Survival Forest |
| mlflow | ≥1.30 | Experiment tracking |

## Limitations & Future Work

1. **TabPFN** - Currently uses XGBoost fallback (integrate actual TabPFN model)
2. **Temporal data** - LOCF/MA designed for single visits (not sequence)
3. **Censoring handling** - Some models don't account for censoring
4. **Feature selection** - No built-in feature importance/selection
5. **Imbalance handling** - No class weighting by default

## Performance Notes

- **Random Survival Forest** - Slowest to train, best calibration
- **XGBoost/CatBoost** - Fast, good discrimination, requires calibration
- **Logistic Regression** - Fastest, good baseline
- **Cox PH** - Good for interpreted coefficients

## References

- lifelines: https://lifelines.readthedocs.io/
- scikit-survival: https://scikit-survival.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- CatBoost: https://catboost.ai/

## License

Internal use for Multiple Myeloma digital twin project.

## Contact

PhD Researcher 3 - Baseline Modeling Expert
