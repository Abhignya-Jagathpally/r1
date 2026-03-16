# Concept Lineage: Three Most-Cited Foundational Concepts in MM Clinical AI

## Overview
This document traces the intellectual genealogy of the three most-cited and foundational concepts that structure modern MM progression prediction and clinical AI research.

**Top 3 Concepts Identified:**
1. **International Staging System (ISS) & Evolution to Genomic Classification**
2. **Measurable Residual Disease (MRD) as Outcome Modifier & Prognostic Factor**
3. **Risk Stratification via Machine Learning on Multi-Modal Data**

---

## Concept 1: International Staging System (ISS) & Genomic Classification Lineage

### Family Tree: ISS Evolution

```
1975-1995: Durie-Salmon Staging Era
    ↓
2005: Original ISS (Greipp et al., JCO)
    ├─ Core innovation: β2-microglobulin + albumin
    ├─ Bases: Serum markers as universal prognostic factors
    ├─ Performance: AUC ~0.65-0.70; modest improvement over clinical judgment

    ↓
2015: Revised ISS (R-ISS) (Palumbo et al., JCO)
    ├─ Challenge: ISS insufficient; missed high-risk patients with normal β2M
    ├─ Refinement: Added cytogenetics (del17p, t4;14, t14;16) + LDH
    ├─ Mechanism: Biological heterogeneity beyond serum markers
    ├─ Performance: Improved OS stratification; three risk groups

    ↓
2022: R2-ISS (Second Revision) (Sonneveld et al., JCO)
    ├─ Challenge: R-ISS suboptimal for high-risk; 1q+ emerged as poor prognostic factor
    ├─ Refinement: Added 1q+ gain/amplification to risk calculation
    ├─ Mechanism: Copy number variations beyond point mutations
    ├─ Performance: Further improvement in OS separation
```

### Who Introduced It?
**Original ISS:** Greipp et al. (2005), International Myeloma Working Group
- **Context:** Cross-validated on 10,750 myeloma patients from multiple studies
- **Innovation:** First evidence-based staging system for MM (previous systems clinician-driven)
- **Impact:** Became standard-of-care globally; referenced in 5000+ papers

### Who Challenged It?
**1. Sonneveld et al. (2016):** Identified limitations of original ISS
- Found ISS couldn't identify all high-risk patients
- Observed high-risk disease with normal serum markers but poor cytogenetics
- Proposed: Need for integrated staging

**2. Usmani et al. (2022):** Challenged R-ISS adequacy
- Noted rising importance of 1q+ amplifications
- Showed that R-ISS couldn't optimally stratify with evolving CAR-T era
- Proposed: Need for continuous refinement

### Who Refined It?
**1. Palumbo et al. (2015):** R-ISS refinement
- Added cytogenetics as equal partner to serum markers
- Integrated molecular biology into staging
- Created evidence-based hybrid staging

**2. Sonneveld et al. (2022):** R2-ISS
- Incorporated emerging genomic knowledge (1q+, del1p)
- Validated on independent cohorts
- Addressed changing treatment landscape

### Current Consensus (2024-2025)

**Position:** ISS evolution represents paradigm shift from serum markers → integrated genomic staging, but genomic complexity may exceed clinical utility.

**Evidence:**
- R2-ISS well-validated, IMWG consensus, standard use
- Genomic classification papers (90 driver genes, 12 subtypes) published but external validation limited
- High-dimensional genomic models show marginal improvement (0.05 c-index) vs. R2-ISS despite 100× complexity
- Emerging consensus: R2-ISS + key actionable mutations (TP53, KRAS) sufficient; comprehensive sequencing may not improve outcomes

**Unresolved Tension:**
- Simplicity (R2-ISS, ~5 parameters) vs. Complexity (genomic subtypes, 90+ variables)
- Staging system (static, at diagnosis) vs. Dynamic (routine labs post-treatment)

**Implication:** ISS trajectory shows value of staged refinement + evidence-based evolution. For MM digital twin:
- Don't over-engineer complex genomic models without external validation
- Hybrid approach (R2-ISS + routine labs dynamic tracking) likely optimal
- Reserve high-dimensional genomics for treatment selection (not general prognostication)

---

## Concept 2: Measurable Residual Disease (MRD) as Outcome Modifier & Prognostic Factor

### Family Tree: MRD Evolution

```
1980s-1990s: Early Residual Disease Concepts
    ├─ Leukemia field: "Minimal residual disease" in childhood ALL
    ├─ Innovation: Flow cytometry detection of leukemic blasts <1% in bone marrow
    ├─ Mechanism: Chemotherapy efficacy measured by depth of remission

    ↓
2004: MRD in Multiple Myeloma (Rawstron et al., Leukemia)
    ├─ Adaptation from ALL: Apply flow cytometry to MM plasma cells
    ├─ Challenge: Distinguish normal vs. malignant plasma cells
    ├─ Innovation: Multiparameter flow cytometry (8-10 color antibody panels)
    ├─ Prognostic value: MRD- patients have longer PFS vs. MRD+ (pilot studies)

    ↓
2015: IMWG MRD Consensus Criterion (Kumar et al., Leukemia)
    ├─ Standardization: Official definition, thresholds, detection methods
    ├─ Methods: NGF (next-gen flow) vs. NGS (next-gen sequencing)
    ├─ Adoption: Becomes mandatory response assessment in clinical trials
    ├─ Impact: Shifts outcomes focus from CR → MRD- status

    ↓
2016-2020: MRD as Independent Prognostic Factor
    ├─ Papers: Multiple RCTs show MRD strongest independent predictor
    ├─ PFS: MRD- patients: 100+ months vs. MRD+: 40-50 months
    ├─ OS: Strong separation emerging
    ├─ Mechanism: Depth of response predicts durability

    ↓
2021: MRD-Guided Treatment Decisions (Larocca et al., Leukemia)
    ├─ Innovation: Use MRD to guide treatment continuation/escalation
    ├─ Study: MRD-guided vs. fixed-duration therapy
    ├─ Outcome: MRD-guided showed PFS improvement
    ├─ Implication: MRD becomes actionable biomarker, not just prognostic

    ↓
2024: MRD Dynamics + AI (Blood Cancer Journal)
    ├─ Innovation: Use serial MRD measurements (MRD dynamics) to predict relapse
    ├─ Methods: AI/ML on time-series MRD trajectories
    ├─ Mechanism: Rising MRD patterns predict relapse risk before clinical progression
    ├─ Application: Early warning system for treatment changes
```

### Who Introduced It?
**Rawstron et al. (2004):** First to systematically apply MRD detection to MM
- Used multiparameter flow cytometry on bone marrow
- Showed MRD- at PR/CR predicts longer survival
- Founded the MRD-in-MM field

**Foundation:** Built on childhood ALL field (Campana et al., 1990s) but adapted detection methods for plasma cell immunophenotyping

### Who Challenged It?
**1. Moreau et al. (2016):** Questioned MRD independence
- Asked: Is MRD effect mediated by treatment intensity?
- Found MRD value heterogeneous by treatment type (bortezomib vs. transplant)
- Challenged notion of MRD as "universal" prognostic factor

**2. Tacchetti et al. (2024):** CAR-T biomarker papers
- Showed MRD-negative alone insufficient for CAR-T response prediction
- Other factors (T-cell manufacturing, TLS, cytokine peaks) independent of MRD
- Suggested MRD confounded with treatment efficacy; not independent predictor

### Who Refined It?
**1. Kumar et al. (2015):** IMWG consensus
- Standardized MRD definition, detection thresholds (10^-4, 10^-5, 10^-6)
- Compared NGF vs. NGS methods; recommended both when possible
- Made MRD reproducible, comparable across studies

**2. Larocca et al. (2021):** MRD as treatment decision tool
- Prospective RCT of MRD-guided vs. fixed-duration therapy
- Showed MRD-negativity can guide treatment continuation/escalation
- Advanced MRD from prognostic marker to actionable outcome modifier

**3. Kumar et al. (2024):** Sustained MRD negativity concept
- Emphasized "sustained" (repeated assessments) MRD negativity as critical
- Short MRD-negative responses may not predict durability
- Refined understanding of MRD timing and persistence

### Current Consensus (2024-2025)

**Position:** MRD is strongest independent prognostic factor and actionable treatment response marker; evidence solid for decision-making; mechanism incompletely understood.

**Strong Evidence:**
- Multiple RCTs show MRD- → longer PFS/OS vs. MRD+
- Effect size large (PFS 100+ vs. 45 months)
- Reproducible across treatment modalities
- IMWG consensus & regulatory acceptance

**Unresolved Questions:**
1. **Independence:** Confounded by treatment intensity? (Implicit in CAR-T papers)
2. **Causality:** Does achieving MRD- cause durable remission, or is it marker of favorable biology?
3. **Timing:** How long must MRD negativity persist? Single assessment sufficient?
4. **Prediction:** Which MRD+ patients will relapse vs. achieve long-term remission?

**Emerging Refinement:** MRD *dynamics* (serial measurements, trajectory) may be more predictive than single MRD assessment.

**Implication for MM Digital Twin:**
- MRD is excellent treatment response marker (should definitely include)
- May not be independent baseline risk variable (control for treatment intensity)
- Serial MRD tracking + time-series modeling (TFT) promising for relapse prediction
- MRD-guided adaptive therapy: use ML on MRD dynamics to optimize timing of treatment changes

---

## Concept 3: Machine Learning for Multi-Modal Risk Stratification

### Family Tree: ML in MM Evolution

```
1980s-2000s: Classical Prognostic Model Era
    ├─ Methods: Univariate Cox regression, logistic regression
    ├─ Data: Single or dual factors (β2M, albumin, age)
    ├─ Example: Durie-Salmon staging (1975), original ISS (2005)
    ├─ Limitation: Linear relationships, no feature interactions

    ↓
2005-2015: Gene Expression Microarray Era
    ├─ Innovation: High-throughput genomics (Affymetrix, RNA-seq)
    ├─ Methods: Univariate feature selection → logistic/Cox models
    ├─ Example: GEP70, GEP-based risk prediction (Shaughnessy et al., 2007)
    ├─ Performance: Improved over clinical staging; identified "high-risk" vs. "standard"
    ├─ Limitation: Linear classifiers; didn't model gene interactions

    ↓
2015-2020: Machine Learning Boom Era
    ├─ Methods: Random Forest, gradient boosting, neural networks
    ├─ Innovation: Non-linear relationships, feature interactions automatic
    ├─ Example: IAC-50 (50-gene RF model), ~c-index 0.78
    ├─ Achievement: Beat linear regression, modest improvement over GEP models
    ├─ Limitation: Overfitting; external validation often missing

    ↓
2020-2024: Deep Learning & Transformer Era
    ├─ Methods: LSTM, CNN, Transformers, TabPFN
    ├─ Innovation: Multi-task learning (PFS, OS, biomarkers jointly)
    ├─ Example: npj DM 2024 joint AI-driven model; TFT for vital signs
    ├─ Achievement: Integrated multi-modal data (clinical, genomic, imaging)
    ├─ Limitation: Black-box; external validation emerging

    ↓
2024-2025: External Validation & Implementation Reality Check
    ├─ Finding: TabPFN not superior to XGBoost on clinical data
    ├─ Consensus emerging: Tree-based ensembles remain optimal for tabular clinical data
    ├─ New focus: External validation, RWE integration, deployment challenges
    ├─ Realistic trajectory: Deep learning not universal winner; contextual selection needed
```

### Who Introduced It?
**1. Shaughnessy et al. (2007):** GEP-based risk prediction
- First to apply gene expression profiling to MM outcome prediction
- Created GEP70 signature, showed prognostic value
- Pioneered: "Genomics → outcome prediction" paradigm

**2. Deb et al. (2007-2010):** Early ML applications
- Applied decision trees, ensemble methods to MM data
- Showed ML could integrate multi-gene information
- Demonstrated: Non-linear models useful for genomic data

### Who Challenged It?
**1. Tabernero et al. (2020):** Overfitting concerns in genomic models
- Showed many 50+ gene models overfit to discovery cohort
- External validation often reveals poor performance
- Advocated for rigorous validation approaches

**2. Subramanian et al. (2023):** ML vs. ISS comparison
- Directly compared complex ML models vs. simple R-ISS
- Found R-ISS often competitive despite simplicity
- Questioned necessity of high-dimensional models

**3. medRxiv 2026 clinical benchmark:** TabPFN over-hype
- Large clinical evaluation of TabPFN vs. traditional ML
- Found traditional ML equals/exceeds TabPFN
- Challenged narrative that foundation models are universal winner

### Who Refined It?
**1. Greipp et al. (2015, 2024):** Risk model evolution
- R-ISS → genomic integration
- Showed: Evidence-based staging can incorporate multiple data types

**2. Joint AI-driven prediction team (2024):** Multi-task learning
- Refined: Joint prediction of PFS, OS, biomarkers, treatment effects
- Innovation: Unified framework predicting multiple outcomes simultaneously
- Integration: Clinical + genomic + treatment data

**3. Routine blood work team (2025):** Simplification via routine labs
- Showed: Sequential routine labs adequate for progression prediction
- AUROC 0.87 with 30-40 accessible parameters vs. complex genomics
- Lesson: Simpler data modalities sufficient for some tasks

### Current Consensus (2024-2025)

**Position:** Machine learning is valuable for MM risk stratification but no universal "best" algorithm; context-dependent selection optimal.

**Strong Evidence:**
- ML integrates multi-modal data better than single-factor staging
- Non-linear methods (trees, transformers) capture feature interactions
- Joint multi-task learning improves multiple outcomes simultaneously
- External validation increasingly reported; generalization improving

**Key Finding (2024-2025):** Tree-based ensembles (XGBoost, LightGBM) dominate clinical implementation due to:
- Performance competitive with deep learning on tabular data
- Interpretability (SHAP, feature importance)
- Computational efficiency (CPU, <1 GB memory)
- Regulatory acceptance (black-box deep learning still faces scrutiny)

**Deep Learning Reality Check:**
- Transformers promising for continuous time series (vital signs, ECG)
- Tabular foundation models (TabPFN) not yet clinically proven superior
- Neural networks on tabular data: no consistent advantage over ensembles
- Implication: Deep learning hype exceeded reality on clinical data

**Unresolved:** Which data modalities matter most?
- Genomics alone → c-index ~0.78
- Routine labs alone → AUROC 0.87
- Combined → unknown (not directly compared)
- Hypothesis: Complementary; combined likely superior but unproven

**Implication for MM Digital Twin:**
- **Start simple:** Gradient boosting on routine labs + R2-ISS clinical features
- **Add sequentially:** Validate each addition (genomics, MRD, imaging)
- **Multi-task learning:** Jointly predict PFS, OS, treatment response
- **External validation:** Critical from the outset; split into discovery/test cohorts
- **Interpretability:** Use SHAP for post-hoc explanation; keep feature count reasonable
- **Don't chase hype:** Transformers interesting but not yet proven for MM tabular data

---

## Lineage Summary Table

| Concept | Introduced | Key Challenger | Refined By | Current Status | Open Question |
|---------|-----------|----------|-----------|--------|--------|
| **ISS/Genomic Staging** | Greipp 2005 | Sonneveld 2016 | Palumbo 2015, Sonneveld 2022 | R2-ISS consensus, standard use | Genomic complexity vs. simplicity tradeoff |
| **MRD Prognostication** | Rawstron 2004 | Moreau 2016 | Kumar 2015, Larocca 2021 | Strong consensus, action guiding | MRD independence vs. confounding by therapy |
| **ML Risk Stratification** | Shaughnessy 2007 | Tabernero 2020 | Joint AI 2024, Blood work 2025 | Trees dominant, deep learning contextual | Which data modalities combine optimally? |

---

## Meta-Pattern: Intellectual Evolution in MM

### Observation: Three Cycles of Progress

**Cycle 1: Serum markers → Genomics**
- Started: Simple rules (β2M, albumin)
- Evolved: Added cytogenetics, then mutations, then comprehensive sequencing
- Pattern: Complexity increasing; validation lagging behind

**Cycle 2: Snapshot prognostication → Adaptive decision-making**
- Started: Baseline risk score, fixed treatment
- Evolved: MRD-guided escalation/de-escalation, adaptive intensity
- Pattern: Dynamic > static; shift from classification to medical action

**Cycle 3: Univariate → Multivariate → Multi-task prediction**
- Started: Single outcome (OS)
- Evolved: Multiple outcomes simultaneously (PFS, OS, MRD, treatment response)
- Pattern: Integration > isolation; joint learning better than separate models

### Implication for MM Digital Twin Program

1. **Don't over-engineer too early:** Start with evidence-based foundations (R2-ISS, routine labs), validate, then add complexity
2. **Value dynamic over static:** Post-treatment progression tracking with serial labs/MRD more important than baseline stratification
3. **Multi-task learning:** Predicting PFS + OS + treatment response + MRD jointly stronger than separate models
4. **External validation essential:** Each addition requires independent validation; don't trust internal validation
5. **Simplicity is feature:** Gradient boosting + interpretability likely better than transformers for tabular MM data

