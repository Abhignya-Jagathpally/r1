# Knowledge Map: Multiple Myeloma Clinical AI

## Central Claim

**Machine learning and deep learning models can integrate multi-modal longitudinal data (genomics, blood work, imaging, clinical outcomes) to predict disease progression, treatment response, and survival in multiple myeloma with superior precision to standard risk scores, enabling individualized treatment strategies and early intervention.**

---

## Supporting Pillars

### Pillar 1: Genomic Risk Stratification Outperforms Clinical Staging
- **Evidence:** MMRF CoMMpass identified 12 genomic subtypes with prognostic separation sharper than ISS/RISS alone.
- **Key studies:** Genomic Basis of Multiple Myeloma Subtypes (MMRF CoMMpass); Individualized Risk Model for Myeloma (IRMMa) using 1,062 CoMMpass patients.
- **Clinical impact:** High-risk subtypes (del(17p), t(4;14), gain(1q)) now inform treatment intensification; del(17p) patients shifted to doublet/triplet induction.

### Pillar 2: Temporal Dynamics Require Sequence Models, Not Static Snapshots
- **Evidence:** Transformer-based models (SCOPE, TFT variants) jointly forecast PFS, OS, adverse events, and biomarker trajectories, outperforming single-endpoint survival models.
- **Key studies:** "Joint AI-driven event prediction and longitudinal modeling in newly diagnosed and relapsed multiple myeloma" (npj Digital Medicine, 2024); "Development and external validation of temporal fusion transformer models" for ICU forecasting (eClinicalMedicine, 2024).
- **Mechanisms:** Attention mechanisms capture long-range dependencies; gating layers suppress irrelevant features; multi-task learning reduces overfitting.

### Pillar 3: Multi-Modal Fusion (Genomics + Labs + Imaging) Improves Discrimination
- **Evidence:** Digital twins and multimodal foundation models (MUSK, UNI) integrate imaging phenotypes, proteomic/genomic risk, and real-time labs to forecast counterfactual outcomes.
- **Key studies:** "Harnessing multimodal data integration to advance precision oncology"; "Multimodal deep learning approaches for precision oncology" (2024).
- **Clinical rationale:** Imaging burden (lytic lesions, extramedullary disease) drives bone complications and mortality independently of serum M-spike; genomic subtypes predict therapy sensitivity independent of lab kinetics.

### Pillar 4: Non-Invasive (Blood Work Alone) Prediction Is Feasible but Imperfect
- **Evidence:** Recent models predict progression events from routine CBC, chemistry, M-spike alone with AUC 0.80+; outperform ISS in internal validation.
- **Key studies:** "Predicting progression events in multiple myeloma from routine blood work" (Nature Digital Medicine, 2025); SCORPIO model for immunotherapy response using clinical factors + labs.
- **Limitations:** Non-secretory myeloma, early-stage genomic complexity, and bone marrow infiltration escape detection by serum biomarkers; genomic-integrated models remain superior.

### Pillar 5: Foundation Models and In-Context Learning Accelerate Deployment
- **Evidence:** TabPFN-2.5 scales to 50,000 samples and 2,000 features; MUSK and UNI enable transfer learning across cancer types; agentic LLMs can operationalize clinical guidelines at scale.
- **Key studies:** "Accurate predictions on small data with a tabular foundation model" (Nature, 2024); "Foundation models in clinical oncology" (Nature Cancer, 2024).
- **Caveat:** Recent benchmarks show TabPFN competitive but not superior to tuned ML baselines in clinical cohorts; deployment requires careful local validation.

---

## Contested Zones

### Zone 1: Do Foundation Models Outperform Established ML in Clinical Practice?
- **Claim A:** TabPFN, vision-language models, and LLMs offer superior generalization and faster training.
- **Claim B:** On real clinical cohorts, established ML (XGBoost, random forests) remain equal or superior; foundation models carry 5–10× computational cost.
- **Current evidence:** "Established Machine Learning Matches Tabular Foundation Models in Clinical Predictions" (medRxiv, 2025) found TabPFN exceeded best ML baseline in only 16.7% of 12 binary clinical tasks.
- **Open question:** Do foundation models excel in low-data regimes (<1,000 samples) or data-scarce modalities (omics, rare subgroups)?

### Zone 2: Prospective Validation and Regulatory Pathways
- **Claim A:** Retrospective AUC/C-index >0.80 is sufficient for clinical use; real-world evidence from EHR deployments confirms value.
- **Claim B:** Lack of prospective randomized trials comparing AI-guided treatment selection vs. standard care prevents regulatory approval and clinical adoption; real-world evidence confounded by selection bias.
- **Current evidence:** All major models (SCOPE, IRMMa, SCORPIO) are retrospectively validated; no Phase II/III RCT yet published showing survival benefit of AI-directed treatment.
- **Clinical bottleneck:** FDA and major hematology societies (ASH, ASCO) have not endorsed AI-selected regimens; guidelines remain risk-stratification based (ISS/RISS + cytogenetics).

### Zone 3: Temporal Stability of Genomic Subtypes and Clonal Evolution
- **Claim A:** 12-subtype classification at diagnosis remains prognostically relevant across all treatment lines and relapse episodes.
- **Claim B:** Clonal selection during induction/maintenance therapy shifts dominant subtype; treatment-emergent resistance subtypes are not captured by baseline genomics.
- **Current evidence:** Correlation of changes in subclonal architecture with progression (MMRF CoMMpass); limited serial sequencing data in published cohorts.
- **Implication:** Models assuming static subtype may under-predict relapse in high-dose chemotherapy responders or miss clonal escape driven by novel mutations.

---

## Frontier Questions

### Frontier 1: Can AI-Guided Treatment Selection Be Validated Prospectively?
**Why it matters:** The single greatest barrier to clinical adoption is lack of prospective evidence that individualized treatment (guided by AI) improves OS or PFS vs. standard-of-care allocation.

**Research pathways:** (1) Pragmatic RCT in RRMM comparing AI-selected doublet vs. physician-selected doublet. (2) Real-world evidence framework leveraging EHR data with causal inference (instrumental variables, propensity matching) to control for selection bias. (3) Integration of AI risk scores into existing trial designs (e.g., TOURMALINE-2) as secondary endpoint for exploratory subgroup analysis.

### Frontier 2: How Do Minimal Residual Disease (MRD) Dynamics Integrate Into Survival Prediction?
**Why it matters:** MRD status (by flow cytometry, NGS) is a rising surrogate for long-term outcomes; models trained on M-spike kinetics alone may miss MRD-driven relapse.

**Research pathways:** Incorporate MRD time-series (longitudinal NGS) into transformer models. Develop joint models predicting both MRD clearance kinetics and OS. Validate on CoMMpass subset with available MRD data.

---

## Must-Read Papers

1. **"Joint AI-driven event prediction and longitudinal modeling in newly diagnosed and relapsed multiple myeloma"** (npj Digital Medicine, 2024)
   - SCOPE transformer model; jointly predicts PFS, OS, adverse events, biomarker trajectories; externally validated on TOURMALINE and RRMM cohorts.
   - Why essential: State-of-the-art multi-task survival framework; demonstrates external generalization.

2. **"Predicting progression events in multiple myeloma from routine blood work"** (npj Digital Medicine, 2025)
   - Hybrid neural network; forecasts future blood work; predicts IMWG progression from lab history alone.
   - Why essential: Clinically actionable (non-invasive); addresses generalizability of blood-work-only predictions.

3. **"MMRF-CoMMpass Data Integration and Analysis for Identifying Prognostic Markers"** (Methods in Molecular Biology, 2020)
   - Data harmonization pipeline for 1,100+ patient cohort; 12-subtype genomic stratification; survival curves by subtype.
   - Why essential: Foundation dataset and metadata standards for all downstream myeloma ML work; open-access data via Virtual Lab.
