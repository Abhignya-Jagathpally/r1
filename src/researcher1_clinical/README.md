# Multiple Myeloma Digital Twin Pipeline

Production-quality data ingestion, cleansing, and feature engineering for the MMRF CoMMpass dataset.

## Overview

This package implements a complete clinical data pipeline for MM digital twin development:

```
CoMMpass Flat Files (CSV/TSV)
    ↓
[Ingestion] → Raw DataFrame
    ↓
[Cleansing] → Harmonized, Imputed, Normalized
    ↓
[Feature Engineering] → Temporal & Clinical Features
    ↓
[Data Splitting] → Train / Val / Test
    ↓
Analysis-Ready Parquet Files
```

## Key Features

### Data Ingestion (`data_ingestion.py`)
- Loads CoMMpass CSV/TSV files from `data/raw/`
- Extracts and standardizes:
  - **Lab values**: Serum M-protein, FLC (kappa/lambda/ratio), hemoglobin, calcium, creatinine, albumin, β2-microglobulin, LDH
  - **Treatment**: Line, transplant flags (autologous, allogeneic)
  - **Genetics**: Cytogenetics, FISH (del13, del17p, t(4;14), t(14;16), gain1q)
  - **Staging**: ISS, R-ISS
  - **Endpoints**: PFS, OS, time-to-progression, relapse
- Output: Long-format DataFrame with `patient_id`, `visit_id`, `timepoint`

### Data Cleansing (`cleansing.py`)
- **Unit harmonization**: Ensures consistent units across visits
- **Long format**: One row per patient per timepoint
- **Missingness masking**: Audit trail of originally missing values
- **Winsorization**: Clinician-reviewed bounds prevent unrealistic values
- **Imputation strategies**: MICE (default), KNN, or median
- **Normalization**: StandardScaler with frozen parameters

**Frozen preprocessing contract**: Once fit on training data, all preprocessing parameters (imputation models, scaler statistics) are locked and applied identically to test/holdout data. Changes require version bumping.

### Feature Engineering (`feature_engineering.py`)
- **Temporal slopes**: Linear trends in lab values over lookback windows
- **Rolling windows**: Mean and std deviation over 90-day windows
- **Time-since-treatment**: Days since last treatment initiation
- **SLiM-CRAB criteria**: Assessment of myeloma defining criteria
- **Trajectory aggregations**: 3, 6, 12-month windows with {mean, std, delta, max, min}

### Data Splitting (`splits.py`)
- **Patient-level splits**: All visits of a patient stay together (prevents leakage)
- **Time-aware splits**: Temporally ordered train/val/test
- **Stratified group k-fold**: Cross-validation with patient grouping and outcome balancing

### Pipeline Orchestration (`pipeline.py`)
- CLI with argparse
- End-to-end orchestration with logging
- Config-driven execution (JSON or CLI arguments)
- Train mode: Fit preprocessing on training data
- Apply mode: Use frozen preprocessing on new data

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

### Quick Start

```bash
python -m researcher1_clinical.pipeline --mode train --data-dir data/raw --output-dir data/processed
```

### Advanced: Config File

Create `config.json`:

```json
{
  "data_dir": "data/raw",
  "output_dir": "data/processed",
  "imputation_strategy": "mice",
  "split_strategy": "stratified_group_kfold",
  "temporal_windows": [90, 180, 365],
  "stratify_column": "pfs_event"
}
```

Run with config:

```bash
python -m researcher1_clinical.pipeline --config config.json --mode train
```

### Programmatic API

```python
from pathlib import Path
from researcher1_clinical import CoMMpassIngester, DataCleaner, FeatureEngineer, DataSplitter

# Ingest
ingester = CoMMpassIngester("data/raw")
raw_df = ingester.ingest()

# Cleanse (train)
cleaner = DataCleaner(imputation_strategy="mice")
cleaner.fit(raw_df)
cleaned_df, missingness_mask = cleaner.apply(raw_df)

# Engineer features
engineer = FeatureEngineer()
engineered_df = engineer.engineer(cleaned_df)

# Split
from researcher1_clinical.splits import SplitConfig
config = SplitConfig(strategy="stratified_group_kfold")
splitter = DataSplitter(config)
train_df, val_df, test_df = splitter.split(engineered_df)
```

## Output Files

| File | Description |
|------|-------------|
| `raw_ingested.parquet` | Raw data from CoMMpass (post-ingestion) |
| `cleaned.parquet` | Cleaned, imputed, normalized data |
| `engineered.parquet` | Full feature set including temporal features |
| `train.parquet` | Training fold |
| `val.parquet` | Validation fold |
| `test.parquet` | Test fold |
| `preprocessing_state.json` | Frozen preprocessing parameters (for reproducibility) |

## Production Considerations

### Preprocessing Contract

The pipeline implements a **frozen preprocessing contract**:

1. **Training phase**: Fit imputation and normalization on training data only
2. **Locked parameters**: All preprocessing params (scaler mean/std, imputation models) are frozen
3. **Test/holdout**: Apply frozen params identically to all test data
4. **Versioning**: Changes to cleansing/feature engineering require explicit version bumping

This ensures:
- No data leakage from test into training
- Reproducible preprocessing across runs
- Audit trail of preprocessing choices

### Logging

All operations emit structured logs. Enable verbose output:

```bash
python -m researcher1_clinical.pipeline --mode train --verbose
```

### Error Handling

- Missing CoMMpass files raise `FileNotFoundError`
- Unit mismatches logged as warnings
- Imputation failures tracked per patient/timepoint
- All NaN/error values preserved in missingness masks

## Clinical Context

### CRAB vs SLiM-CRAB

**CRAB** (original MM defining criteria, deprecated):
- **C**alcium ≥ 11 mg/dL
- **R**enal dysfunction: Creatinine ≥ 2 mg/dL
- **A**nemia: Hemoglobin < 10 g/dL
- **B**one lesions: Lytic lesions on imaging

**SLiM-CRAB** (current standard, IMWG 2014):
- **S**: Clonality ≥60% plasma cells
- **Li**: Light chain ratio: Involved/Uninvolved FLC ≥ 100
- **M**: MRI: ≥1 focal lesion on DWI
- Plus original CRAB criteria (C, R, A, B)

### Lab Value Bounds

Winsorization bounds (parameterizable in `WinsorizeConfig`):

| Lab | Lower | Upper | Unit |
|-----|-------|-------|------|
| Serum M-protein | 0.0 | 10.0 | g/dL |
| FLC Kappa | 0.33 | 19.4 | mg/L |
| FLC Lambda | 0.27 | 26.3 | mg/L |
| Hemoglobin | 5.0 | 18.0 | g/dL |
| Calcium | 6.0 | 14.0 | mg/dL |
| Creatinine | 0.3 | 10.0 | mg/dL |
| Albumin | 1.0 | 5.5 | g/dL |
| β2-microglobulin | 0.5 | 50.0 | mg/L |
| LDH | 100 | 2000 | U/L |

Adjust via `WinsorizeConfig` for clinical domain validation.

## Architecture Notes

### Separation of Concerns

- **Ingestion**: File I/O, format detection, basic parsing
- **Cleansing**: Unit harmonization, missingness handling, imputation, normalization
- **Feature Engineering**: Temporal feature computation, clinical summaries
- **Splitting**: Data partitioning strategies with patient-level grouping

### Type Hints

All functions include full type hints for IDE support and type checking:

```bash
mypy src/researcher1_clinical --strict
```

### Testing (Future)

Production deployment should include:
- Unit tests for each module
- Integration tests for full pipeline
- Regression tests for preprocessing contract
- Clinical validation of feature distributions

## License

Internal research use. Multiple Myeloma Research Foundation (MMRF) CoMMpass dataset.

## Contact

PhD Researcher 2 - Clinical Data Engineering
