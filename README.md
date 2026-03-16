# Multiple Myeloma Clinical AI Research Program

## Overview

This repository contains the complete research program for **Multiple Myeloma (MM) Progression Prediction using Machine Learning and Deep Learning**. It integrates multi-modal longitudinal clinical data (genomics, blood work, imaging, clinical outcomes) to:

1. **Predict disease progression, treatment response, and survival** with transformer-based models
2. **Synthesize current knowledge** in the field with critical analysis of assumptions and contested zones
3. **Define production-ready pipelines** (Nextflow DSL2 and Snakemake) for data integration, feature engineering, and model training
4. **Enable prospective validation** toward clinical deployment

---

## Repository Structure

```
/sessions/clever-hopeful-allen/r1/
├── docs/
│   └── knowledge_maps/
│       ├── field_synthesis.md          # What the field believes, what's proven, key unanswered questions
│       ├── hidden_assumptions.md       # Unstated assumptions in published work
│       ├── knowledge_map.md            # Structured outline: central claim, pillars, contested zones
│       └── executive_brief.md          # 5-minute summary for smart non-experts
├── pipelines/
│   ├── nextflow/
│   │   ├── main.nf                     # Nextflow DSL2 pipeline (complete workflow)
│   │   └── nextflow.config             # Nextflow configuration (profiles, resources)
│   └── snakemake/
│       ├── Snakefile                   # Snakemake pipeline (equivalent to Nextflow)
│       └── config.yaml                 # Snakemake parameters
└── README.md                           # This file
```

---

## Knowledge Synthesis Documents

### 1. **field_synthesis.md** (400 words max)
- **What the field collectively believes**: Multi-modal integration, genomic stratification, transformer architectures
- **What remains contested**: Foundation model utility vs. established ML; prospective validation gaps
- **What's proven**: Deep learning survival models outperform Cox; SCOPE transformer jointly predicts PFS/OS/AE
- **Single most important unanswered question**: Can AI-guided treatment selection be validated prospectively to improve OS/PFS?

### 2. **hidden_assumptions.md** (5 assumptions with consequences)
1. Blood work alone captures disease burden sufficiently (Non-secretory MM escapes detection)
2. Genomic subtypes remain stable across treatment (Clonal evolution undermines static stratification)
3. Historical cohorts generalize to modern therapy (Domain shift with CAR-T, bispecific antibodies)
4. Model uncertainty → clinical decision uncertainty (Physicians distrust probabilistic outputs)
5. Accessible data = causal drivers (Proxies vs. true mechanisms)

### 3. **knowledge_map.md** (Structured outline)
- **Central claim**: ML + deep learning can predict MM progression/treatment response better than ISS/RISS
- **5 supporting pillars**: Genomic stratification, temporal dynamics, multi-modal fusion, non-invasive prediction, foundation models
- **2 contested zones**: Foundation model superiority; prospective validation pathways
- **2 frontier questions**: Prospective AI-guided treatment validation; MRD dynamics integration
- **3 must-read papers**: SCOPE, blood-work progression, CoMMpass data integration

### 4. **executive_brief.md** (Non-expert summary)
- **One-sentence proof**: "Deep learning outperforms ISS in retrospective cohorts (AUC 0.82 vs. 0.77)"
- **One honest admission**: "We don't know if AI-picked treatments save lives yet"
- **One real-world implication**: "If validated, could enable precision dosing and reduce side effects by 15–20%"

---

## Pipeline Definitions

### Nextflow (main.nf + nextflow.config)

**Architecture**: DSL2 modular processes with strict channel semantics and publish directives.

**Processes** (19 total):
1. **Ingestion**: `ingest_genomics`, `ingest_clinical`, `ingest_labs`, `ingest_imaging`
2. **Cleansing**: `cleanse_genomics`, `cleanse_labs`
3. **Feature extraction**: `extract_radiomics_features`
4. **Feature engineering**: `engineer_features`
5. **Splitting**: `create_data_splits`
6. **Training**: `train_baseline_models`, `train_advanced_models`
7. **Evaluation**: `evaluate_models`
8. **Reporting**: `generate_report`

**Key Features**:
- Multi-profile support (local, cluster, GPU, Docker, Singularity)
- Proper channel management (emit named outputs)
- Labeled processes for resource allocation
- Execution reports (timeline, DAG, trace)

---

## Data Flow

```
Raw Data (VCF, CSV, Parquet, DICOM)
         ↓
    ┌────┴────┐
    ↓         ↓
 Ingestion & QC
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Cleansing & Normalization
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Feature Extraction (Radiomics)
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Feature Engineering
 (Temporal, Genomic, Clinical)
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Train/Val/Test Splits
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Baseline Model Training
 (XGBoost, RF, LR)
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Advanced Model Training
 (SCOPE Transformer, DeepSurv, TFT)
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Evaluation & Metrics
    ↓         ↓
    ├─────────┤
    ↓         ↓
 Report Generation (HTML)
```

---

## Key Research Insights

### What Works
1. Transformer-based joint outcome prediction (SCOPE)
2. Genomic subtype stratification (12-subtype MMRF classification)
3. Deep survival models (DeepHit, DeepSurv with C-index 0.80+)
4. Longitudinal lab trajectory modeling

### What Remains Uncertain
1. Foundation models in clinical settings
2. Prospective treatment allocation validation
3. Multi-modal integration optimal approach
4. Temporal stability of genomic risk

### Critical Gaps
1. Prospective RCT validation needed
2. Regulatory pathway unclear
3. Real-world generalization untested
4. Causal inference not established

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-15 | Initial release: knowledge synthesis + Nextflow + Snakemake pipelines |

For complete documentation, see individual knowledge map files and inline comments in pipeline definitions.
