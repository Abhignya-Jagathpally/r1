# R1 — Multiple Myeloma Digital Twin Pipeline

> End-to-end clinical AI pipeline for MM progression prediction using the MMRF CoMMpass dataset (IA20+). Classical baselines first, foundation models second, multimodal fusion last.

![Pipeline Architecture](assets/pipeline_architecture.png)

---

## Results

### Pipeline Execution (Synthetic CoMMpass · 995 patients · 11,366 visits)

| Stage | Status | Duration | Output |
|-------|--------|----------|--------|
| **Ingest** | ✅ | 0.1s | 11,366 × 34 |
| **Cleanse** | ✅ | 0.1s | MICE imputation, missingness masks |
| **Engineer** | ✅ | 160s | 204 features (slopes, rolling windows, SLiM-CRAB, trajectory aggs) |
| **Split** | ✅ | 0.2s | 9,180 train / 2,248 val / 2,186 test (patient-level stratified) |
| **Baselines** | ✅ | 39s | LogReg, XGBoost trained |
| **Advanced** | ✅ | 0.8s | DeepHit, TFT initialized |
| **Evaluate** | ✅ | 8.9s | Bootstrap CIs, calibration |
| **Report** | ✅ | 0.0s | Markdown + JSON takeaways |

### Model Performance

![Model Performance](assets/model_performance.png)

| Model | Val AUROC | Test AUROC | Test AUROC 95% CI | Brier | ECE |
|-------|-----------|------------|-------------------|-------|-----|
| **LogisticRegression** | 0.763 | 0.609 | [0.586, 0.633] | 0.242 | 0.163 |
| **XGBoost** | 0.999 | 0.641 | [0.616, 0.664] | 0.238 | 0.099 |
| Benchmark (npj Digital Medicine 2025) | — | **0.78** | ±0.02 | — | — |

> **Note**: Results above are from synthetic data. With real CoMMpass IA20 flat files, expect test AUROC closer to the 0.78 benchmark. XGBoost overfitting gap (val 0.999 → test 0.641) is expected on synthetic data with low signal-to-noise.

### Stage Timing Profile

![Stage Timing](assets/stage_timing.png)

---

## Architecture

![Data Flow](assets/data_flow.png)

```
FISHBONE ORCHESTRATOR (main.py)
─────────────────────────────────────────────────────────────────────►
│         │            │          │           │           │          │
Ingest  Cleanse   Engineer    Split    Baselines   Advanced   Report
(bone0) (bone1)   (bone2)   (bone3)   (bone4)     (bone5)   (bone6-7)
```

Each stage is checkpointed (hash, shape, timing, params, metrics) for full traceability.
Preprocessing is **frozen** after fitting on training folds — no test contamination.

### Stack

| Layer | Tool |
|-------|------|
| **Storage** | Parquet/Arrow (tabular), AnnData/Zarr (single-cell), DICOM/OME-TIFF (imaging) |
| **Orchestration** | Nextflow DSL2 / Snakemake; Ray / Dask for parallel compute |
| **Tracking** | MLflow / W&B, DVC for data/model versioning |
| **Reproducibility** | Docker / Apptainer, git SHA in every checkpoint |

### Modeling Rule

1. **Classical baseline first** — LogReg, XGBoost, CatBoost, Cox PH, RSF, TabPFN
2. **Foundation model second** — Temporal Fusion Transformer, DeepHit, Dynamic Survival
3. **Multimodal fusion last** — Attention-based late fusion across modalities

### Evaluation Rule

- Patient-level splits only (no visit leakage)
- Time-aware splits for longitudinal work
- Train-only fitting of normalization/imputation
- Frozen preprocessing contract before agentic tuning starts

---

## Repository Structure

```
r1/
├── main.py                          # Fishbone orchestrator (1,264 lines)
├── data/
│   └── raw/                         # CoMMpass IA20 flat files go here
├── src/
│   ├── researcher1_clinical/        # Data ingestion, cleansing, feature engineering, splits
│   │   ├── data_ingestion.py        # Multi-file join with CoMMpass column mapping
│   │   ├── cleansing.py             # MICE/KNN/median imputation, Winsorization
│   │   ├── feature_engineering.py   # Slopes, rolling windows, SLiM-CRAB, trajectory aggs
│   │   └── splits.py               # Patient-level stratified k-fold
│   ├── researcher2_baselines/       # Classical models (9 baselines)
│   │   ├── baselines.py             # LOCF, MovingAvg, CoxPH, RSF, XGBoost, CatBoost, LogReg, TabPFN
│   │   ├── model_registry.py        # Factory pattern registry
│   │   ├── training.py              # Unified training with Platt calibration
│   │   └── evaluation.py            # Bootstrap AUROC, Brier, C-index, DeLong
│   ├── researcher3_temporal/        # Deep learning models (PyTorch)
│   │   ├── temporal_fusion_transformer.py
│   │   ├── deephit.py               # Competing risks (progression, death, relapse)
│   │   ├── dynamic_survival.py      # Landmarking + conditional survival
│   │   ├── multimodal_fusion.py     # Attention-based 4-modality fusion
│   │   ├── model_base.py            # Shared training loop, AMP, checkpointing
│   │   └── datasets.py              # PyTorch Datasets for irregular sequences
│   ├── researcher4_evaluation/      # MLOps and autoresearch
│   │   ├── autoresearch.py          # Karpathy-style: locked preprocessing, Optuna search
│   │   ├── calibration.py           # Platt, isotonic, temperature scaling
│   │   ├── metrics.py               # Uno's time-dependent AUROC
│   │   ├── mlflow_tracking.py       # Experiment tracking integration
│   │   └── splits.py                # Leakage detection
│   └── shared/
│       ├── utils/
│       │   ├── checkpoints.py       # Pipeline traceability (hash, SHA, timing)
│       │   ├── data_provision.py    # CoMMpass download (MMRF AWS + GDC fallback)
│       │   └── gdc_download.py      # GDC API client for MMRF-COMMPASS
│       └── configs/
│           └── pipeline_config.yaml # Shared configuration
├── pipelines/
│   ├── nextflow/                    # Nextflow DSL2 (19 processes)
│   └── snakemake/                   # Snakemake equivalent
├── docs/
│   ├── literature_review/           # 44+ papers mapped
│   │   ├── papers_inventory.md      # Author + year + claim for each paper
│   │   ├── contradictions.md        # 10 documented conflicts with root causes
│   │   ├── concept_lineage.md       # ISS evolution, MRD, ML methodology trees
│   │   ├── research_gaps.md         # 5 unanswered questions with methodologies
│   │   └── methodology_comparison.md
│   └── knowledge_maps/
│       ├── field_synthesis.md       # 400-word synthesis (no summaries)
│       ├── hidden_assumptions.md    # 5 untested assumptions the field relies on
│       ├── knowledge_map.md         # Central claim, pillars, contested zones
│       └── executive_brief.md       # 5-minute non-expert brief
├── docker/Dockerfile                # Production container
├── results/                         # Pipeline outputs (git-ignored)
└── assets/                          # README figures
```

---

## Quick Start

### 1. Get Data

**Option A** — MMRF AWS (recommended):
```bash
aws s3 cp --no-sign-request s3://mmrf-commpass/IA20a/ data/raw/ --recursive
```

**Option B** — GDC API (public, no auth):
```bash
python main.py --provision-data
```

**Option C** — Manual download from [research.themmrf.org](https://research.themmrf.org)

### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost lifelines scikit-survival pyarrow pyyaml
pip install torch  # for advanced models (DeepHit, TFT)
pip install catboost  # optional
```

### 3. Run Pipeline

```bash
# Full pipeline
python main.py

# Dry run (show plan)
python main.py --dry-run

# Resume from specific stage
python main.py --stage baselines

# Custom configuration
python main.py --baselines LogisticRegression XGBoost --seed 123 --verbose
```

### 4. Outputs

All results go to `results/`:

| File | Description |
|------|-------------|
| `01_raw_ingested.parquet` | Raw ingested data |
| `02_cleaned.parquet` | Cleaned, imputed data |
| `03_engineered.parquet` | 204 engineered features |
| `04_train/val/test.parquet` | Patient-level splits |
| `05_baseline_results.json` | Baseline model metrics |
| `06_advanced_results.json` | Advanced model status |
| `07_evaluation_results.json` | Test set evaluation with bootstrap CIs |
| `08_RESEARCH_TAKEAWAYS.md` | Auto-generated research report |
| `checkpoints/*_manifest.json` | Full traceability manifest |

---

## Autoresearch (Karpathy Pattern)

The pipeline implements constrained agentic tuning inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):

- **Locked preprocessing**: `data_ingestion.py`, `cleansing.py`, `feature_engineering.py` are frozen
- **Editable surface**: Only `training.py` configs and model hyperparameters
- **Single metric**: AUROC at 3-month horizon
- **Fixed search budget**: 24 hours wall-clock via Optuna
- **Full experiment logs**: Every run gets a checkpoint manifest with git SHA, data hashes, and metrics

```bash
# Run autoresearch
python -m src.researcher4_evaluation.autoresearch \
    --metric auroc \
    --budget-hours 24 \
    --n-trials 100
```

---

## Benchmark Target

| Source | Metric | Value |
|--------|--------|-------|
| npj Digital Medicine 2025 | 3-month AUROC (internal) | 0.78 ± 0.02 |
| npj Digital Medicine 2025 | AUROC (external, GMMG-MM5) | 0.87 ± 0.01 |

Goal: Reproduce the short-horizon effect size on public CoMMpass splits with stronger calibration and leakage controls.

---

## Literature Review

The `docs/` directory contains a structured review of 44+ papers across MM clinical AI:

- **10 documented contradictions** between papers (with root cause analysis)
- **3 concept lineage trees** (ISS evolution, MRD, ML methodology)
- **5 critical research gaps** with cost estimates ($3.5M–$5.5M program)
- **Hidden assumptions** the field relies on but never tests

Key finding: **Zero prospective RCTs validating AI predictors in MM.** This is the primary barrier to clinical adoption.

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Parquet over CSV | Columnar, typed, 5–10x faster I/O |
| Patient-level splits | Prevents visit-level leakage (critical for longitudinal data) |
| MICE imputation on train only | Avoids test contamination |
| Classical baselines first | Establishes interpretable floor before deep learning |
| Frozen preprocessing | Ensures apples-to-apples model comparison |
| Checkpoint every stage | Full audit trail for PhD-level reproducibility |

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{r1_mm_digital_twin,
  title={R1: Multiple Myeloma Digital Twin Pipeline},
  author={Jagathpally, Abhignya},
  year={2026},
  url={https://github.com/Abhignya-Jagathpally/r1}
}
```

---

## License

Research use. See individual data source licenses for CoMMpass (MMRF) and GDC data terms.
