<div align="center">

# R1 — Multiple Myeloma Digital Twin Pipeline

**End-to-end clinical AI for MM progression prediction on MMRF CoMMpass (IA20+)**

*Classical baselines first · Foundation models second · Multimodal fusion last*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)](#testing)
[![Pipeline](https://img.shields.io/badge/pipeline-9%20stages-blueviolet.svg)](#architecture)
[![License](https://img.shields.io/badge/license-research%20use-orange.svg)](#license)
[![Dataset](https://img.shields.io/badge/data-MMRF%20CoMMpass-teal.svg)](https://research.themmrf.org)

<br>

<img src="assets/pipeline_architecture.png" alt="Pipeline Architecture" width="100%">

</div>

---

## Highlights

- **9-stage fishbone orchestrator** with full checkpoint traceability (data hash, shape, timing, params, metrics, git SHA)
- **Patient-level stratified splits** — no visit-level leakage across train/val/test
- **Frozen preprocessing contract** — MICE imputation fit on training folds only, serialized via pickle
- **Classical → Deep → Fusion** modeling hierarchy: LogReg, XGBoost, RSF, CoxPH → DeepHit, TFT → Multimodal attention fusion
- **Karpathy-style autoresearch** — locked preprocessing, Optuna Bayesian HP search, single-metric optimization
- **Bootstrap evaluation** — AUROC with 95% CI, Brier score, calibration ECE, concordance index

---

## Results

Results are categorized by data provenance. Different data tiers produce fundamentally different result quality.

### Tier 1: GDC Metadata-Only Prototype

The GDC Cases API provides open metadata for ~995 MMRF-COMMPASS patients (~6 usable features: age, gender, race, ISS stage, vital status, survival endpoints). **No lab values, no longitudinal visits, no treatment data.**

<div align="center">
<img src="assets/model_performance.png" alt="Model Performance" width="90%">
</div>

<br>

| Model | Val AUROC | Test AUROC | Test 95% CI | Brier | Task Type |
|:------|:---------:|:----------:|:-----------:|:-----:|:----------|
| **LogisticRegression** | 0.604 | **0.758** | [0.673, 0.829] | 0.140 | snapshot classification |
| **XGBoost** | 0.857 | 0.703 | [0.621, 0.780] | 0.149 | snapshot classification |
| **RandomSurvivalForest** | 0.846 | 0.539 | [0.489, 0.585] | 0.268 | survival analysis |

> **Note:** These results reflect the limited discriminative power of demographics-only features. They are **not comparable** to published benchmarks that used full longitudinal lab data. See [benchmark comparison](#benchmark-target) below.

### Tier 2: Full CoMMpass Flat Files — *Planned*

With MMRF IA20 flat files (free registration at [research.themmrf.org](https://research.themmrf.org)), the pipeline gains longitudinal lab values (M-protein, FLC, hemoglobin, calcium, creatinine, albumin, B2M, LDH), treatment lines, transplant status, and per-visit records enabling temporal feature engineering (slopes, rolling windows, SLiM-CRAB).

**Status**: Architecture tested on Tier 1. Training on full flat files requires Researcher Gateway access.

### Tier 3: Full Molecular Data — *Planned*

With dbGaP-controlled data (phs000748): WES, RNA-seq, CNV for multimodal fusion.

**Status**: DeepHit, TFT, and multimodal fusion modules are implemented but require Tier 2/3 data for meaningful training.

### Benchmark Target

| Source | Metric | Value | Data Used |
|:-------|:-------|:-----:|:----------|
| npj Digital Medicine 2025 | 3-month AUROC (internal) | 0.78 ± 0.02 | Full longitudinal labs (Tier 2) |
| npj Digital Medicine 2025 | AUROC (external, GMMG-MM5) | 0.87 ± 0.01 | Full longitudinal labs (Tier 2) |

> **Caveat:** This benchmark used full longitudinal lab data with IPCW-weighted survival metrics. Tier 1 results use snapshot classification on demographics-only features. The two are not directly commensurate.

---

## Architecture

<div align="center">
<img src="assets/data_flow.png" alt="Data Flow" width="65%">
</div>

<br>

Each stage writes a checkpoint (data hash, shape, timing, parameters, metrics, git SHA) for full traceability. Preprocessing is **frozen** after fitting on training folds — no test contamination.

### Modeling Hierarchy

```
Layer 1: Classical Baselines     →  LogReg, XGBoost, CoxPH, RSF
Layer 2: Foundation Models       →  DeepHit (competing risks), TFT (multi-horizon)
Layer 3: Multimodal Fusion       →  Attention-based late fusion across modalities
```

### Evaluation Protocol

- Patient-level splits only (no visit leakage)
- Snapshot classification and survival evaluation reported separately (`task_type`)
- Train-only fitting of normalization/imputation (frozen preprocessing contract)
- Naive time-dependent AUROC clearly labeled (no IPCW weighting)

### Execution Profile

<div align="center">
<img src="assets/stage_timing.png" alt="Stage Timing" width="70%">
</div>

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install torch        # optional: DeepHit, TFT, multimodal fusion
pip install catboost     # optional: CatBoost baseline
```

### 2. Get Data

See [DATA_ACCESS.md](DATA_ACCESS.md) for full details.

```bash
# Option A: GDC open metadata (Tier 1, ~6 features, no registration)
python main.py --provision-data

# Option B: MMRF Researcher Gateway (Tier 2, free registration)
# Download IA20 flat files from https://research.themmrf.org → data/raw/
```

### 3. Run

```bash
python main.py                           # full pipeline
python main.py --dry-run                 # show plan only
python main.py --stage baselines         # resume from stage
python main.py --seed 123 --verbose      # custom config
```

### 4. Test

```bash
pytest tests/ -v    # 7 tests: leakage, winsorization, SLiM-CRAB, serialization
```

### 5. Outputs

All results go to `results/` (or `--output-dir`):

| File | Description |
|:-----|:------------|
| `01_raw_ingested.parquet` | Raw ingested data |
| `02_cleaned.parquet` | Cleaned, imputed data |
| `03_engineered.parquet` | Engineered features |
| `04_train/val/test.parquet` | Patient-level splits |
| `05_baseline_results.json` | Baseline model metrics |
| `07_evaluation_results.json` | Test evaluation with bootstrap CIs |
| `08_RESEARCH_TAKEAWAYS.md` | Auto-generated research report |
| `checkpoints/*_manifest.json` | Full traceability manifest |
| `autoresearch/` | Optuna search results |

---

## Repository Structure

```
r1/
├── main.py                          # Fishbone orchestrator (9 stages)
├── requirements.txt                 # Top-level dependencies
├── DATA_ACCESS.md                   # Data source access matrix
├── src/
│   ├── researcher1_clinical/        # Ingestion, cleansing, feature engineering, splits
│   ├── researcher2_baselines/       # Classical models + evaluation
│   ├── researcher3_temporal/        # Deep learning (DeepHit, TFT, multimodal fusion)
│   ├── researcher4_evaluation/      # Autoresearch, calibration, metrics, MLflow
│   └── shared/utils/               # Checkpoints, data provisioning, GDC client
├── tests/                           # 7 correctness tests (pytest)
├── docs/                            # Literature review (44+ papers)
├── pipelines/                       # Nextflow + Snakemake (planned integration)
├── docker/                          # Container definition (planned)
└── assets/                          # README figures
```

---

## Autoresearch (Karpathy Pattern)

Constrained agentic tuning with locked preprocessing:

| Parameter | Value |
|:----------|:------|
| **Locked surface** | `data_ingestion.py`, `cleansing.py`, `feature_engineering.py` |
| **Editable surface** | Model hyperparameters (LR, batch size, epochs, dropout, L2) |
| **Metric** | Validation AUROC |
| **Search** | Optuna TPE (Bayesian), configurable wall-clock budget |

> **Caveat:** On Tier 1 GDC metadata, autoresearch may report high validation AUROC due to the small feature set with clean demographic patterns. This does not generalize. Results should be interpreted in the context of the data tier used.

---

## Key Decisions

| Decision | Rationale |
|:---------|:----------|
| Parquet over CSV | Columnar, typed, 5–10x faster I/O |
| Patient-level splits | Prevents visit-level leakage (critical for longitudinal data) |
| MICE imputation on train only | Avoids test contamination |
| Classical baselines first | Establishes interpretable floor before deep learning |
| Frozen preprocessing (pickle) | Real serialization ensures apples-to-apples comparison |
| Checkpoint every stage | Full audit trail for PhD-level reproducibility |
| No winsorization on FLC/M-protein | Extreme values are the disease signal in MM |

---

## Limitations

1. **No prospective validation.** All results are retrospective. Zero prospective RCTs have validated AI predictors in MM (field-wide gap).
2. **GDC demo uses metadata only.** The publicly runnable demo has no lab values, no treatment data, and no longitudinal features.
3. **Benchmark comparison is not commensurate.** The npj Digital Medicine 2025 benchmark used full longitudinal labs with IPCW-weighted metrics.
4. **Advanced models are architecture-only.** DeepHit, TFT, and multimodal fusion are implemented but require Tier 2/3 data for meaningful training.
5. **Nextflow/Snakemake pipelines are not integrated.** They define equivalent workflows but are disconnected from the Python orchestrator.

---

## Literature Review

The `docs/` directory contains a structured review of 44+ papers across MM clinical AI: 10 documented contradictions with root cause analysis, 3 concept lineage trees (ISS evolution, MRD, ML methodology), 5 critical research gaps with cost estimates ($3.5M–$5.5M program), and hidden assumptions the field relies on but never tests.

**Key finding:** Zero prospective RCTs validating AI predictors in MM. This is the primary barrier to clinical adoption.

---

## Citation

```bibtex
@software{r1_mm_digital_twin,
  title   = {R1: Multiple Myeloma Digital Twin Pipeline},
  author  = {Jagathpally, Abhignya},
  year    = {2026},
  url     = {https://github.com/Abhignya-Jagathpally/r1}
}
```

---

<div align="center">

**License:** Research use. See individual data source licenses for CoMMpass (MMRF) and GDC data terms.

</div>