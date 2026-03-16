# MM Digital Twin Pipeline — Research Takeaways

**Run ID**: run_20260315_205500
**Date**: 2026-03-15 20:57
**Git SHA**: ab6943155005ee01f708ac1faf31bd9305d4019c
**Pipeline Version**: 0.1.0
**Seed**: 42

---

## 1. Pipeline Execution Summary

| Stage | Status | Duration (s) | Output Shape |
|-------|--------|-------------|--------------|
| ingest | completed | 0.0 | [995, 39] |
| cleanse | completed | 0.1 | [995, 38] |
| engineer | completed | 8.6 | [995, 74] |
| split | completed | 0.2 | [796, 199, 199] |
| baselines | completed | 26.9 | [2] |
| advanced | completed | 0.0 | [3] |
| evaluate | completed | 2.0 | [2] |
| autoresearch | completed | 90.0 | [1] |

---

## 2. Baseline Model Results

| Model | Val AUROC | Val Brier | Status |
|-------|-----------|-----------|--------|
| LogisticRegression | 0.6043 | 0.1560 | ok |
| XGBoost | 0.8571 | 0.1286 | ok |
| RandomSurvivalForest | N/A | N/A | observed time contains values smaller zero |

---

## 3. Advanced Model Status

| Model | Status | Note |
|-------|--------|------|
| deephit | skipped |  |
| temporal_fusion_transformer | skipped |  |
| multimodal_fusion | skipped |  |

---

## 4. Test Set Evaluation

| Model | AUROC | AUROC 95% CI | Brier | ECE |
|-------|-------|-------------|-------|-----|
| LogisticRegression | 0.7582 | [0.6734, 0.8294] | 0.1401 | 0.0899 |
| XGBoost | 0.7030 | [0.6209, 0.7798] | 0.1491 | 0.0787 |

---

## 5. Key Takeaways

- **Best performing model**: LogisticRegression achieved AUROC 0.7582 on the held-out test set.
- **Below benchmark**: The best model (0.7582) is below the published benchmark of 0.78. Consider hyperparameter tuning or additional feature engineering.
- **Baselines trained**: 2/3 models converged.
- **Data leakage**: Patient-level splitting enforced throughout. Verify via checkpoint manifest.
- **Frozen preprocessing**: All imputation and normalization parameters fitted on training fold only (no test contamination).

---

## 6. Reproducibility

- **Git SHA**: `ab6943155005ee01f708ac1faf31bd9305d4019c`
- **Git Branch**: `main`
- **Git Dirty**: True
- **Python**: 3.12.3
- **Platform**: Linux-6.8.0-57-generic-x86_64-with-glibc2.39
- **Random Seed**: 42
- **Imputation**: mice
- **Split Strategy**: stratified_group_kfold

### Checkpoint Manifest
Full traceability log saved to: `results/checkpoints/run_20260315_205500_manifest.json`

---

## 7. Limitations & Next Steps

1. **Prospective validation needed**: All results are retrospective on CoMMpass data.
2. **Clonal evolution**: Genomic subtypes may shift at relapse, limiting static stratification.
3. **Modern therapies**: CAR-T and bispecific antibody outcomes are underrepresented in training data.
4. **Non-secretory MM**: Blood-work-only models miss ~5% of patients without measurable M-protein.
5. **Calibration**: Verify probability calibration before clinical deployment (Platt/isotonic scaling).

---
*Generated automatically by the MM Digital Twin Pipeline.*