# R1 -- Multiple Myeloma Digital Twin Pipeline

> End-to-end clinical AI pipeline for MM progression prediction using the MMRF CoMMpass dataset (IA20+). Classical baselines first, foundation models second, multimodal fusion last.

![Pipeline Architecture](assets/pipeline_architecture.png)

---

## Results

Results are categorized by data provenance. Different data tiers produce fundamentally different result quality.

### Tier 1: GDC Metadata-Only Prototype

The GDC Cases API provides open metadata for ~995 MMRF-COMMPASS patients: age at diagnosis, gender, race, ISS stage, vital status, and days to death/recurrence (~6 usable features). **No lab values, no longitudinal visits, no treatment data.**

Results from `python main.py --provision-data --output-dir ./test_run --verbose`:

| Model | Val AUROC | Test AUROC | Test 95% CI | Brier | Task Type |
|-------|-----------|------------|-------------|-------|-----------|
| **LogisticRegression** | 0.604 | 0.758 | [0.673, 0.829] | 0.140 | snapshot classification |
| **XGBoost** | 0.857 | 0.703 | [0.621, 0.780] | 0.149 | snapshot classification |
| **RandomSurvivalForest** | failed | -- | -- | -- | (negative survival times in GDC data) |

These results reflect the limited discriminative power of demographics-only features. They are **not comparable** to published benchmarks that used full longitudinal lab data. See the [benchmark comparison caveat](#benchmark-target) below.

### Tier 2: Full CoMMpass Flat Files (requires MMRF Researcher Gateway) -- Planned

With MMRF IA20 flat files (free registration at [research.themmrf.org](https://research.themmrf.org)), the pipeline gains:
- Longitudinal lab values (M-protein, FLC, hemoglobin, calcium, creatinine, albumin, B2M, LDH)
- Treatment lines, transplant status, response assessments
- Per-visit records enabling temporal features (slopes, rolling windows, SLiM-CRAB)

**Status**: Architecture tested on Tier 1. Training on full flat files requires Researcher Gateway access.

### Tier 3: Full Molecular Data (requires dbGaP approval) -- Planned

With dbGaP-controlled data (phs000748): WES, RNA-seq, CNV for multimodal fusion.

**Status**: DeepHit, TFT, and multimodal fusion modules are implemented but have not completed training runs. These are architecture-only.

### Benchmark Target

| Source | Metric | Value | Data Used |
|--------|--------|-------|-----------|
| npj Digital Medicine 2025 | 3-month AUROC (internal) | 0.78 +/- 0.02 | Full longitudinal labs (Tier 2) |
| npj Digital Medicine 2025 | AUROC (external, GMMG-MM5) | 0.87 +/- 0.01 | Full longitudinal labs (Tier 2) |

**Caveat**: This benchmark used full longitudinal lab data with IPCW-weighted survival metrics. Tier 1 results use snapshot classification on demographics-only features. The two are not directly commensurate.

---

## Architecture

```
FISHBONE ORCHESTRATOR (main.py) — 9 stages
──────────────────────────────────────────────────────────────────────────────►
│         │            │          │           │           │          │         │
Ingest  Cleanse   Engineer    Split    Baselines   Advanced  Evaluate  Report
(bone0) (bone1)   (bone2)   (bone3)   (bone4)     (bone5)  (bone6)  (bone8)
                                                      │
                                                 Autoresearch
                                                   (bone7)
```

Each stage writes a checkpoint (data hash, shape, timing, parameters, metrics, git SHA) for full traceability. Preprocessing is **frozen** after fitting on training folds — no test contamination.

### Modeling Rule

1. **Classical baseline first** -- LogReg, XGBoost, CatBoost, Cox PH, RSF
2. **Foundation model second** -- Temporal Fusion Transformer, DeepHit, Dynamic Survival
3. **Multimodal fusion last** -- Attention-based late fusion across modalities

### Evaluation Rule

- Patient-level splits only (no visit leakage)
- Snapshot classification and survival evaluation reported separately (`task_type` parameter)
- Train-only fitting of normalization/imputation (frozen preprocessing contract)
- Naive time-dependent AUROC clearly labeled (no IPCW weighting; see evaluation.py docstrings)

---

## Repository Structure

```
r1/
├── main.py                          # Fishbone orchestrator (9 stages)
├── requirements.txt                 # Top-level dependencies
├── DATA_ACCESS.md                   # Data source access matrix
├── data/
│   └── raw/                         # CoMMpass IA20 flat files go here
├── src/
│   ├── researcher1_clinical/        # Data ingestion, cleansing, feature engineering, splits
│   │   ├── data_ingestion.py        # Multi-file join with CoMMpass column mapping
│   │   ├── cleansing.py             # MICE/KNN/median imputation, instrument-error winsorization
│   │   ├── feature_engineering.py   # Slopes, rolling windows, IMWG SLiM-CRAB, trajectory aggs
│   │   └── splits.py               # Patient-level stratified k-fold
│   ├── researcher2_baselines/       # Classical models
│   │   ├── baselines.py             # LOCF, MovingAvg, CoxPH, RSF, XGBoost, CatBoost, LogReg, TabPFN
│   │   ├── model_registry.py        # Factory pattern registry
│   │   ├── training.py              # Cross-validation with patient-level splits
│   │   └── evaluation.py            # Bootstrap AUROC, Brier, C-index (task_type aware)
│   ├── researcher3_temporal/        # Deep learning models (PyTorch, requires torch)
│   │   ├── temporal_fusion_transformer.py  # Multi-horizon prediction
│   │   ├── deephit.py               # Competing risks (progression, death, relapse)
│   │   ├── dynamic_survival.py      # Landmarking + conditional survival (not yet wired into main.py)
│   │   ├── multimodal_fusion.py     # Attention-based 4-modality fusion
│   │   ├── model_base.py            # Shared training loop, AMP, checkpointing
│   │   └── datasets.py              # PyTorch Datasets for irregular sequences
│   ├── researcher4_evaluation/      # MLOps and autoresearch
│   │   ├── autoresearch.py          # Karpathy-style: locked preprocessing, Optuna search
│   │   ├── calibration.py           # Platt, isotonic, temperature scaling
│   │   ├── metrics.py               # Time-dependent survival metrics
│   │   ├── mlflow_tracking.py       # Experiment tracking (optional, requires mlflow)
│   │   └── splits.py                # Leakage detection
│   └── shared/
│       ├── utils/
│       │   ├── checkpoints.py       # Pipeline traceability (hash, SHA, timing)
│       │   ├── data_provision.py    # CoMMpass download (MMRF AWS + GDC fallback)
│       │   └── gdc_download.py      # GDC Cases API client (open metadata only)
│       └── configs/
│           └── pipeline_config.yaml # Shared configuration
├── tests/
│   └── test_pipeline.py             # 7 correctness tests (pytest)
├── pipelines/                       # Workflow definitions (not used by main.py)
│   ├── nextflow/                    # Nextflow DSL2 — planned integration with main.py
│   └── snakemake/                   # Snakemake equivalent — planned integration
├── docs/
│   ├── literature_review/           # 44+ papers: contradictions, gaps, lineage trees
│   └── knowledge_maps/              # Field synthesis, hidden assumptions, executive brief
├── docker/Dockerfile                # Container definition (planned, not yet tested with main.py)
└── assets/                          # README figures
```

**Notes on scaffolding**: `pipelines/nextflow/`, `pipelines/snakemake/`, and `docker/Dockerfile` exist as planned infrastructure. They define equivalent workflows but are **not currently integrated** with the Python fishbone orchestrator (`main.py`). The actual pipeline runs entirely through `main.py`. Similarly, `dynamic_survival.py` is implemented but not yet wired into the fishbone stages.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install torch    # optional: for DeepHit, TFT
pip install catboost # optional: CatBoost baseline
```

### 2. Get Data

See [DATA_ACCESS.md](DATA_ACCESS.md) for full details on data tiers and access.

**Option A** -- GDC Cases API (Tier 1, open metadata only):
```bash
python main.py --provision-data
```
Returns case-level metadata (~995 patients, ~6 features). No lab values or treatment data.

**Option B** -- MMRF Researcher Gateway (Tier 2, free registration):
Download IA20 flat files from [research.themmrf.org](https://research.themmrf.org) and place in `data/raw/`.

### 3. Run

```bash
python main.py                           # full pipeline
python main.py --dry-run                 # show plan only
python main.py --stage baselines         # resume from stage
python main.py --seed 123 --verbose      # custom config
```

### 4. Test

```bash
pytest tests/ -v
```

### 5. Outputs

All results go to `results/` (or `--output-dir`):

| File | Description |
|------|-------------|
| `01_raw_ingested.parquet` | Raw ingested data |
| `02_cleaned.parquet` | Cleaned, imputed data |
| `03_engineered.parquet` | Engineered features |
| `04_train/val/test.parquet` | Patient-level splits |
| `05_baseline_results.json` | Baseline model metrics |
| `07_evaluation_results.json` | Test set evaluation with bootstrap CIs |
| `08_RESEARCH_TAKEAWAYS.md` | Auto-generated research report |
| `preprocessing_state.pkl` | Frozen preprocessing parameters |
| `checkpoints/*_manifest.json` | Full traceability manifest |
| `autoresearch/` | Optuna search results (if autoresearch stage runs) |

---

## Autoresearch (Karpathy Pattern)

Constrained agentic tuning with locked preprocessing:

- **Locked**: `data_ingestion.py`, `cleansing.py`, `feature_engineering.py` are frozen
- **Editable**: Only model hyperparameters (learning rate, batch size, epochs, dropout, L2)
- **Metric**: Validation AUROC
- **Budget**: Configurable wall-clock via Optuna TPE sampler

**Caveat**: On Tier 1 GDC metadata, autoresearch may report high validation AUROC (up to 0.94) due to the small feature set with clean demographic patterns. This does not generalize to test data or real clinical use. Autoresearch results should be interpreted in the context of the data tier used.

---

## Limitations

1. **No prospective validation.** All results are retrospective. Zero prospective RCTs have validated AI predictors in MM (field-wide gap).
2. **GDC demo uses metadata only.** The publicly runnable demo has no lab values, no treatment data, and no longitudinal features. Results are proof-of-concept, not clinical findings.
3. **Benchmark comparison is not commensurate.** The npj Digital Medicine 2025 benchmark used full longitudinal labs (Tier 2) with IPCW-weighted metrics. Tier 1 snapshot classification results cannot be meaningfully compared.
4. **Advanced models are architecture-only.** DeepHit, TFT, and multimodal fusion are implemented but have not completed training runs (requires Tier 2/3 data).
5. **Nextflow/Snakemake pipelines are not integrated.** They define equivalent workflows but are disconnected from the Python orchestrator. Planned for future integration.

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Parquet over CSV | Columnar, typed, 5-10x faster I/O |
| Patient-level splits | Prevents visit-level leakage (critical for longitudinal data) |
| MICE imputation on train only | Avoids test contamination |
| Classical baselines first | Establishes interpretable floor before deep learning |
| Frozen preprocessing (pickle) | Real serialization ensures apples-to-apples comparison |
| Checkpoint every stage | Full audit trail for PhD-level reproducibility |
| No winsorization on FLC/M-protein | Extreme values are the disease signal in MM |

---

## Literature Review

The `docs/` directory contains a structured review of 44+ papers across MM clinical AI:

- **10 documented contradictions** between papers (with root cause analysis)
- **3 concept lineage trees** (ISS evolution, MRD, ML methodology)
- **5 critical research gaps** with cost estimates ($3.5M--$5.5M program)
- **Hidden assumptions** the field relies on but never tests

Key finding: **Zero prospective RCTs validating AI predictors in MM.** This is the primary barrier to clinical adoption.

---

## Citation

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
