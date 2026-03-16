# API Reference - Multiple Myeloma Baseline Models

## Core Classes

### `BaselineModel` (Abstract Base Class)

All baseline models inherit from this class.

#### Methods

##### `fit(X_train, y_train, **kwargs) -> BaselineModel`
Fit model to training data.

**Parameters:**
- `X_train` (pd.DataFrame or np.ndarray): Feature matrix (n_samples, n_features)
- `y_train` (dict): Labels with keys:
  - `'time'` (np.ndarray): Event times/follow-up duration
  - `'event'` (np.ndarray): Binary event indicators (0/1)
- `**kwargs`: Model-specific arguments

**Returns:** Self (for chaining)

**Raises:** ValueError if required data format invalid

**Example:**
```python
y_train = {
    'time': np.array([3.2, 5.1, 2.8, ...]),
    'event': np.array([0, 1, 0, ...])
}
model.fit(X_train, y_train)
```

##### `predict(X_test) -> np.ndarray`
Generate risk scores (0-1).

**Parameters:**
- `X_test` (pd.DataFrame or np.ndarray): Test features

**Returns:** np.ndarray of shape (n_samples,) with risk scores

**Raises:** ValueError if model not fitted

**Example:**
```python
risk_scores = model.predict(X_test)  # [0.1, 0.8, 0.3, ...]
```

##### `predict_proba(X_test, horizons=None) -> dict`
Generate calibrated progression probabilities at multiple horizons.

**Parameters:**
- `X_test` (pd.DataFrame or np.ndarray): Test features
- `horizons` (list): Time horizons in months. Default: [3, 6, 12]

**Returns:** Dict mapping months -> np.ndarray of probabilities

**Example:**
```python
proba = model.predict_proba(X_test)
# {3: [0.1, 0.8, 0.3, ...], 6: [...], 12: [...]}

proba = model.predict_proba(X_test, horizons=[1, 6])
# {1: [...], 6: [...]}
```

---

## Baseline Models

### 1. `LOCFBaseline`
Last Observation Carried Forward - naive baseline.

**Parameters:**
- `name` (str): Model name. Default: "LOCF"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:** None

**Example:**
```python
model = LOCFBaseline()
model.fit(X_train, y_train)
scores = model.predict(X_test)
```

---

### 2. `MovingAverageBaseline`
Temporal baseline with moving window.

**Parameters:**
- `window` (int): Window size. Default: 3
- `name` (str): Model name. Default: "MA"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:**
- `window`: int, range [1, 10]

**Example:**
```python
model = MovingAverageBaseline(window=5)
model.fit(X_train, y_train)
```

---

### 3. `CoxPHBaseline`
Cox Proportional Hazards model (lifelines).

**Parameters:**
- `name` (str): Model name. Default: "CoxPH"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:** None

**Notes:**
- Semi-parametric survival model
- Assumes proportional hazards
- Returns normalized partial hazard scores

**Example:**
```python
model = CoxPHBaseline()
model.fit(X_train, y_train)
```

---

### 4. `RandomSurvivalForestBaseline`
Random Survival Forest ensemble (scikit-survival).

**Parameters:**
- `n_estimators` (int): Number of trees. Default: 100
- `max_depth` (int): Maximum tree depth. Default: 10
- `name` (str): Model name. Default: "RSF"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:**
- `n_estimators`: int, range [50, 300]
- `max_depth`: int, range [5, 20]

**Notes:**
- Handles censoring properly
- Good calibration
- Computationally expensive

**Example:**
```python
model = RandomSurvivalForestBaseline(n_estimators=200, max_depth=15)
model.fit(X_train, y_train)
```

---

### 5. `XGBoostSnapshotBaseline`
Gradient boosting for binary progression (3-month).

**Parameters:**
- `max_depth` (int): Maximum tree depth. Default: 6
- `learning_rate` (float): Learning rate. Default: 0.1
- `n_estimators` (int): Number of boosting rounds. Default: 100
- `name` (str): Model name. Default: "XGBoost"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:**
- `max_depth`: int, range [3, 12]
- `learning_rate`: float (log scale), range [0.01, 0.3]
- `n_estimators`: int, range [50, 300]

**Notes:**
- Includes Platt scaling calibration
- Fast training
- Good discrimination

**Example:**
```python
model = XGBoostSnapshotBaseline(max_depth=8, learning_rate=0.05)
model.fit(X_train, y_train)
scores = model.predict(X_test)
```

---

### 6. `CatBoostSnapshotBaseline`
Categorical Boosting for binary progression.

**Parameters:**
- `depth` (int): Tree depth. Default: 6
- `learning_rate` (float): Learning rate. Default: 0.1
- `iterations` (int): Number of iterations. Default: 100
- `name` (str): Model name. Default: "CatBoost"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:**
- `depth`: int, range [3, 12]
- `learning_rate`: float (log scale), range [0.01, 0.3]
- `iterations`: int, range [50, 300]

**Notes:**
- Handles categorical features automatically
- Includes Platt scaling calibration
- Good gradient boosting alternative

**Example:**
```python
model = CatBoostSnapshotBaseline(depth=8)
model.fit(X_train, y_train)
```

---

### 7. `LogisticRegressionBaseline`
Linear logistic regression classifier.

**Parameters:**
- `name` (str): Model name. Default: "LogReg"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:** None

**Notes:**
- Fastest baseline
- Interpretable coefficients
- Includes Platt scaling calibration

**Example:**
```python
model = LogisticRegressionBaseline()
model.fit(X_train, y_train)
```

---

### 8. `TabPFNBaseline`
TabPFN for tabular prediction (XGBoost fallback).

**Parameters:**
- `name` (str): Model name. Default: "TabPFN"
- `seed` (int): Random seed. Default: 42

**Hyperparameter Search Space:** None

**Notes:**
- Currently uses XGBoost as fallback
- Real TabPFN/TabPFNv2 can be integrated
- Placeholder for neural tabular methods

**Example:**
```python
model = TabPFNBaseline()
model.fit(X_train, y_train)
```

---

## Registry & Factory

### `ModelRegistry`

Factory pattern for creating and managing models.

#### Methods

##### `create(model_name: str, **kwargs) -> BaselineModel`
Create a model instance.

**Parameters:**
- `model_name` (str): One of:
  - "LOCF", "MovingAverage", "CoxPH", "RandomSurvivalForest"
  - "XGBoost", "CatBoost", "LogisticRegression", "TabPFN"
- `**kwargs`: Override default hyperparameters

**Returns:** Instantiated BaselineModel

**Raises:** ValueError if model not registered

**Example:**
```python
registry = ModelRegistry()
model = registry.create("XGBoost", learning_rate=0.05, max_depth=8)
```

##### `list_models() -> dict`
List all registered models.

**Returns:** Dict mapping model names to descriptions

**Example:**
```python
models = registry.list_models()
# {'LOCF': 'Last Observation...', 'XGBoost': '...', ...}
```

##### `get_default_params(model_name: str) -> dict`
Get default hyperparameters.

**Parameters:**
- `model_name` (str): Model name

**Returns:** Dict of default parameters

**Example:**
```python
params = registry.get_default_params("XGBoost")
# {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100, ...}
```

##### `get_search_space(model_name: str) -> list`
Get hyperparameter search space for tuning.

**Parameters:**
- `model_name` (str): Model name

**Returns:** List of HyperparameterSpace objects

**Example:**
```python
space = registry.get_search_space("XGBoost")
# [HyperparameterSpace(...), HyperparameterSpace(...), ...]
```

##### `register(model_name, model_class, default_params, search_space, description)`
Register a custom model.

**Parameters:**
- `model_name` (str): Unique name
- `model_class` (type): Class inheriting from BaselineModel
- `default_params` (dict): Default hyperparameters
- `search_space` (list): HyperparameterSpace objects
- `description` (str): Model description

---

## Training

### `BaselineTrainer`

Orchestrates model training with cross-validation.

#### Methods

##### `train_baseline(model_name, X_train, y_train, X_val=None, y_val=None, **model_kwargs)`

Train single model.

**Parameters:**
- `model_name` (str): Model name from registry
- `X_train` (pd.DataFrame or np.ndarray): Training features
- `y_train` (dict): Training labels
- `X_val` (optional): Validation features
- `y_val` (optional): Validation labels
- `**model_kwargs`: Override hyperparameters

**Returns:** Tuple of (trained_model, metrics_dict)

**Example:**
```python
trainer = BaselineTrainer()
model, metrics = trainer.train_baseline(
    "XGBoost",
    X_train, y_train,
    X_val, y_val,
    learning_rate=0.05
)
print(f"Validation AUROC: {metrics['val_auroc']:.4f}")
```

##### `cross_validate(model_name, X, y, patient_ids=None, **model_kwargs) -> dict`

Cross-validate model with patient-level splitting.

**Parameters:**
- `model_name` (str): Model name
- `X` (pd.DataFrame or np.ndarray): Features
- `y` (dict): Labels
- `patient_ids` (optional np.ndarray): Patient IDs for group splitting
- `**model_kwargs`: Override hyperparameters

**Returns:** Dict with CV results:
- `'mean_auroc'`, `'std_auroc'`
- `'mean_brier'`, `'std_brier'`
- `'fold_metrics'`: List of per-fold metrics

**Example:**
```python
cv_result = trainer.cross_validate(
    "XGBoost",
    X, y,
    patient_ids=patient_ids
)
print(f"CV AUROC: {cv_result['mean_auroc']:.4f} ± {cv_result['std_auroc']:.4f}")
```

##### `train_all_baselines(X_train, y_train, X_val=None, y_val=None, model_names=None) -> dict`

Train all models.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels
- `X_val` (optional): Validation features
- `y_val` (optional): Validation labels
- `model_names` (optional list): Specific models (None = all)

**Returns:** Dict mapping model names to (model, metrics) tuples

**Example:**
```python
results = trainer.train_all_baselines(X_train, y_train, X_val, y_val)
for name, (model, metrics) in results.items():
    print(f"{name}: {metrics.get('val_auroc', 'N/A'):.4f}")
```

---

## Evaluation

### `BaselineEvaluator`

Comprehensive evaluation with multiple metrics.

#### Methods

##### `auroc_score(y_true, y_pred, bootstrap_ci=True)`

Compute AUROC with bootstrap CI.

**Parameters:**
- `y_true` (np.ndarray): Binary labels
- `y_pred` (np.ndarray): Predicted probabilities
- `bootstrap_ci` (bool): Whether to compute CI. Default: True

**Returns:**
- If bootstrap_ci=False: float (AUROC)
- If bootstrap_ci=True: tuple (mean, lower, upper)

**Example:**
```python
evaluator = BaselineEvaluator()
auroc, lo, hi = evaluator.auroc_score(y_true, y_pred)
print(f"AUROC: {auroc:.4f} [{lo:.4f}, {hi:.4f}]")
```

##### `brier_score(y_true, y_pred, bootstrap_ci=True)`

Compute Brier score (MSE) with bootstrap CI.

**Returns:** float or (mean, lower, upper)

##### `calibration_metrics(y_true, y_pred, n_bins=10) -> dict`

Compute calibration error metrics.

**Returns:** Dict with:
- `'expected_calibration_error'`: ECE
- `'max_calibration_error'`: MCE
- `'bin_fractions'`: Empirical positive rate per bin
- `'bin_means'`: Mean prediction per bin

**Example:**
```python
cal = evaluator.calibration_metrics(y_true, y_pred)
print(f"ECE: {cal['expected_calibration_error']:.4f}")
```

##### `concordance_index(times, events, predictions, bootstrap_ci=True)`

Compute C-index (concordance) for survival data.

**Parameters:**
- `times` (np.ndarray): Follow-up times
- `events` (np.ndarray): Event indicators
- `predictions` (np.ndarray): Predicted risk scores
- `bootstrap_ci` (bool): Compute CI. Default: True

**Returns:** float or (mean, lower, upper)

##### `time_dependent_auroc(times, events, predictions, time_horizons, bootstrap_ci=False) -> dict`

Compute AUROC at multiple time horizons.

**Parameters:**
- `times` (np.ndarray): Follow-up times (months)
- `events` (np.ndarray): Event indicators
- `predictions` (np.ndarray): Risk scores
- `time_horizons` (list): Time points [3, 6, 12]
- `bootstrap_ci` (bool): Compute CI. Default: False

**Returns:** Dict mapping horizon -> AUROC

**Example:**
```python
td_auroc = evaluator.time_dependent_auroc(
    times, events, predictions,
    time_horizons=[3, 6, 12]
)
print(f"AUROC@3mo: {td_auroc[3]:.4f}")
```

##### `evaluate_model(y_true, y_pred, times=None, events=None, time_horizons=None, model_name="Model") -> dict`

Comprehensive evaluation with all metrics.

**Returns:** Dict with:
- AUROC (with CI)
- Brier score (with CI)
- C-index (with CI)
- Calibration (ECE, MCE)
- Time-dependent AUROC

**Example:**
```python
metrics = evaluator.evaluate_model(
    y_test["event"],
    y_pred,
    times=y_test["time"],
    events=y_test["event"],
    time_horizons=[3, 6, 12],
    model_name="XGBoost"
)
```

##### `benchmark_comparison(model_results, target_auroc=0.78, target_auroc_ci=0.02) -> pd.DataFrame`

Create comparison table vs. benchmark.

**Parameters:**
- `model_results` (dict): Dict from evaluate_model calls
- `target_auroc` (float): Benchmark AUROC. Default: 0.78
- `target_auroc_ci` (float): CI width. Default: 0.02

**Returns:** pd.DataFrame with comparison results

**Example:**
```python
comparison_df = evaluator.benchmark_comparison({
    'XGBoost': metrics1,
    'CatBoost': metrics2,
})
print(comparison_df)
```

---

## Data Format

### Input Data Structure

```python
# Features (required)
X = pd.DataFrame(...)  # or np.ndarray
# shape: (n_samples, n_features)

# Labels (required)
y = {
    'event': np.array([0, 1, 0, ...]),      # Binary (0/1)
    'time': np.array([3.2, 5.1, 2.8, ...])  # Duration in months
}

# Patient IDs (optional, for group splitting)
patient_ids = np.array(['P001', 'P002', ...])
```

---

## Configuration

### Default Hyperparameters

Accessed via registry:

```python
params = registry.get_default_params("ModelName")
```

Model-specific defaults configured in `model_registry.py`.

### Search Spaces

For Bayesian optimization:

```python
space = registry.get_search_space("ModelName")
# Returns list of HyperparameterSpace objects
```

---

## Logging & MLflow

Models automatically log to MLflow when trainer has `use_mlflow=True`:

```python
trainer = BaselineTrainer(use_mlflow=True)
# Automatically logs to MLflow experiment "myeloma_baselines"

mlflow.ui()  # View results
```

---

## Exceptions

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Model must be fitted before prediction` | Called predict() before fit() | Call fit() first |
| `ValueError: Model 'X' not registered` | Invalid model name | Use registry.list_models() |
| `ValueError: patient_level_splits=True but patient_ids not provided` | Missing patient IDs | Pass patient_ids array |
| `FileNotFoundError: Data file not found` | Bad data path | Check file path |

