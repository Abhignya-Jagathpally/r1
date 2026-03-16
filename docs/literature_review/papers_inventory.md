# Multiple Myeloma AI & Digital Twin Literature Inventory

## Compiled Literature Review
**Scope:** MM progression prediction, clinical AI, CoMMpass dataset analysis, survival prediction, TabPFN, temporal fusion transformers, digital twins in oncology, routine blood work cancer prediction.

**Date:** March 2026 | **Benchmark Reference:** npj Digital Medicine 2025 MM progression AUROC 0.78 ± 0.02 (3-month), external validation AUROC 0.87 ± 0.01 (GMMG-MM5)

---

## PAPERS BY DOMAIN

### Domain 1: Multiple Myeloma Progression Prediction & AI Models

#### 1.1 Joint AI-driven Event Prediction and Longitudinal Modeling (2024)
**Citation:** Nature npj Digital Medicine 2024
**Core Claim:** Transformer-based hybrid models jointly predict PFS/OS/adverse events and forecast biomarkers, outperforming ISS and state-of-the-art deep learning on disease biomarker prediction.
**Key Details:**
- Trained on CoMMpass (N=703 newly diagnosed MM)
- Externally validated on relapsed/refractory (N=720)
- Superior performance vs ISS on multiple prognostic tasks
- Integrates clinical, genomic, and treatment data

#### 1.2 Predicting Progression Events from Routine Blood Work (2025)
**Citation:** npj Digital Medicine 2025
**Core Claim:** Hybrid neural network using only routine laboratory values predicts disease progression events with AUROC 0.78 ± 0.02 (3-month) and external validation AUROC 0.87 ± 0.01 on GMMG-MM5 dataset.
**Key Details:**
- Trained on CoMMpass (N=1,186 patients)
- Cost-effective, scalable "virtual human twin" for MM
- Baseline event identification AUROC 0.88 ± 0.01
- Progressively declines to AUROC 0.65 ± 0.01 at 12-month horizon
- **BENCHMARK MATCH:** This is the reference benchmark cited in the research brief

#### 1.3 AI-driven Prediction in Newly Diagnosed and Relapsed MM (2024)
**Citation:** npj Digital Medicine 2024
**Core Claim:** Joint prediction of PFS, OS, and adverse events with superior biomarker forecasting compared to existing staging systems and deep learning baselines.
**Key Details:**
- Transformer architecture for multi-horizon predictions
- Improved stratification compared to ISS
- Treatment effect assessment capability
- Handles both NDMM and RRMM cohorts

#### 1.4 Genomic Individualized Prediction Model (2024)
**Citation:** Journal of Clinical Oncology 2024
**Core Claim:** Machine learning model identifies 90 driver genes and 12 genomic subtypes for MM with superior accuracy in predicting OS and EFS than current prognostic models.
**Key Details:**
- Incorporates genomic classification
- Significantly outperforms ISS on survival endpoints
- Identifies disease subgroups with distinct prognosis
- Data-driven approach to genomic risk stratification

#### 1.5 Gene Expression-Based MM Staging & MGUS Progression (2024)
**Citation:** bioRxiv 2024.11.12
**Core Claim:** Machine learning models predict MM stage and MGUS→MM progression using gene expression data, enabling early risk identification.
**Key Details:**
- RNA-seq based predictions
- Identifies progression-prone MGUS patients
- Supports early intervention strategies

#### 1.6 Gene Expression-Based 50-Variable Model (IAC-50)
**Citation:** Clinical & Translational Hematology
**Core Claim:** Random forests model incorporating clinical and gene expression data predicts overall survival with c-index 0.818 (training) and 0.780 (validation).
**Key Details:**
- Integrates 50 gene expression variables
- Strong validation performance
- Combines clinical staging with genomic data

---

### Domain 2: Digital Twins in Oncology & Personalized Medicine

#### 2.1 Application of Digital Twins for Personalized Oncology (2025)
**Citation:** Nature Reviews Cancer 2025
**Core Claim:** Digital twins create dynamic virtual replicas of patient physiological/pathological states, enabling in silico simulation of treatment responses and personalized medicine at scale.
**Key Details:**
- Integrates multi-dimensional data streams
- Simulates response to immunotherapy, chemotherapy, radiation
- Real-time adaptation as condition evolves
- Clinical applications: precision treatment, radiotherapy optimization, surgical planning

#### 2.2 Digital Twins as Paradigm Shift for Precision Cancer Care (2021)
**Citation:** Nature Reviews Cancer 2021
**Core Claim:** Digital twins represent a transformative approach enabling simulation-based treatment planning and outcome prediction before clinical delivery.
**Key Details:**
- Emerging technology not yet ready for clinic
- In silico trial simulation capability
- Accelerates clinical intervention evaluation
- Addresses cost and duration of clinical trials

#### 2.3 Exploring Digital Twins in Cancer Treatment (2025)
**Citation:** Journal of Clinical Medicine 2025
**Core Claim:** Narrative review establishing digital twins across cancer treatment continuum from diagnostics to survivorship, with emphasis on heterogeneous tumor response modeling.
**Key Details:**
- Comprehensive review of applications
- Precision treatment selection emphasis
- Immuno-oncology modeling advances
- Emerging but not yet clinically implemented

#### 2.4 Digital Twins in Oncology: From Predictive Modelling to Treatment (2024)
**Citation:** ScienceDirect 2024
**Core Claim:** Digital twins integrate patient-specific data to enable personalized treatment strategies with improved outcomes and reduced adverse effects.
**Key Details:**
- Multi-modality treatment simulation
- Individualized therapy optimization
- Foundation for precision oncology

---

### Domain 3: Tabular Foundation Models (TabPFN) in Clinical Settings

#### 3.1 Accurate Predictions on Small Data with TabPFN (2024)
**Citation:** Nature 2024
**Core Claim:** Tabular Prior-data Fitted Network (TabPFN) is a foundation model outperforming all previous methods on small-to-medium datasets (up to 10K samples) with minimal training time.
**Key Details:**
- Transformer architecture using in-context learning
- TabPFN-2.5: supports up to 50K rows, 2K features
- Inference in ~2.8 seconds
- Solves classification/regression without fine-tuning

#### 3.2 Established ML Matches TabPFN in Clinical Predictions (2026)
**Citation:** medRxiv 2026
**Core Claim:** Large benchmark of TabPFN vs 12 established ML methods across 12 binary clinical tasks found TabPFN competitive but NOT consistently superior to strong baselines.
**Key Details:**
- Only exceeded best ML model in 16.7% of tasks
- Most AUROC differences ±0.01 (clinically insignificant)
- GPU-dependent (5.5× slower than reported)
- Questions clinical utility vs computational cost
- **CONTRADICTION INDICATOR:** Oversells performance vs actual clinical benchmarks

#### 3.3 TabPFN for Classification and Regression (2022)
**Citation:** ArXiv 2207.01848
**Core Claim:** TabPFN achieves state-of-the-art performance on small tabular datasets through meta-learning without requiring parameter updates.
**Key Details:**
- In-context learning paradigm
- Rapid inference (<1 second)
- Generalizes across diverse tabular problems
- Foundational architecture paper

---

### Domain 4: Temporal Fusion Transformers for Healthcare

#### 4.1 TFT for Interpretable Multi-horizon Time Series Forecasting (2019)
**Citation:** International Journal of Forecasting 2020
**Core Claim:** Temporal Fusion Transformer combines attention mechanisms with sequence-to-sequence architecture for interpretable, multi-horizon time series forecasting in complex domains.
**Key Details:**
- Multi-head attention for long-range dependencies
- Static enrichment layers for covariate integration
- Both probabilistic and point forecasts
- Interpretable attention weights

#### 4.2 Development and External Validation of TFT for Intraoperative BP Forecasting (2024)
**Citation:** eClinicalMedicine/Lancet 2024
**Core Claim:** TFT successfully predicts intraoperative blood pressure trajectories 7 minutes in advance, trained on 73K+ anesthesia patients with low-resolution vital sign data.
**Key Details:**
- Clinical application in perioperative medicine
- Low-frequency data (15-second sampling)
- Validated external generalization
- Real-time patient monitoring capability

#### 4.3 Simultaneous Forecasting of Vital Sign Trajectories in ICU (2025)
**Citation:** Scientific Reports 2025
**Core Claim:** TFT-multi predicts multiple vital signs simultaneously (BP, pulse, SpO2, temperature, RR) with global model architecture for ICU patients.
**Key Details:**
- Multi-variate multi-horizon prediction
- Real-time patient condition tracking
- Cascaded fine-tuning for individual patient adaptation
- Probabilistic forecasting for risk management

#### 4.4 Multivariate Multi-Horizon Time Series in Patient Monitoring (2025)
**Citation:** Computers in Biology and Medicine 2025
**Core Claim:** Cascaded fine-tuning of attention-based models (including TFT) improves generalizability to unseen patient profiles in continuous monitoring.
**Key Details:**
- Sequential patient-level fine-tuning strategy
- Enhanced generalization vs global models
- Applicable to personalized medicine

---

### Domain 5: Prognostic Markers & Staging Systems

#### 5.1 International Staging System Revisions (ISS, R-ISS, R2-ISS)
**Citation:** Journal of Clinical Oncology 2005, 2015, 2022
**Core Claim:** Iterative refinement of MM staging: original ISS (β2M, albumin) → R-ISS (add cytogenetics, LDH) → R2-ISS (add 1q+, chromosome 13 del).
**Key Details:**
- Original ISS: serum β2-microglobulin, albumin
- R-ISS (2015): adds del(17p), t(4;14), t(14;16), LDH
- R2-ISS (2022): adds 1q+ gain/amplification
- Improved survival stratification over time
- **BENCHMARK:** AI models frequently compared vs ISS

#### 5.2 Genomic Classification and Individualized Prognosis (2024)
**Citation:** Journal of Clinical Oncology 2024
**Core Claim:** Comprehensive genomic analysis identifies 90 driver genes and 12 molecular subtypes with superior prognostic accuracy vs clinical staging alone.
**Key Details:**
- NGS-based molecular classification
- Integrates CNAs and mutations
- Translocations: t(4;14), t(14;16), del(13q), del(17p) → poor prognosis
- Hyperdiploidy, t(11;14) → favorable prognosis
- KRAS/NRAS mutations (20-25%), TP53 (6-8%, poor)

#### 5.3 Real-World ISS (RW-ISS) (2025)
**Citation:** Blood Cancer Journal 2025
**Core Claim:** Real-world adaptation of ISS incorporating real-world evidence from registries shows performance differences vs clinical trial populations.
**Key Details:**
- Accounts for real-world patient heterogeneity
- Includes comorbidities not in classic ISS
- Bridges clinical trial-real world outcome gaps

---

### Domain 6: Measurable Residual Disease (MRD) as Prognostic Factor

#### 6.1 Role of MRD Assessment in Multiple Myeloma (2023)
**Citation:** Haematologica 2023
**Core Claim:** MRD negativity (via flow cytometry or NGS) is a strong, independent, and modifiable prognostic factor superior to traditional staging for outcome prediction.
**Key Details:**
- Detects residual malignant plasma cells post-treatment
- Improved PFS vs persistent MRD+ (104 vs 45 months median)
- Requires NGF or NGS for detection
- IMWG consensus criterion since 2015

#### 6.2 Making Clinical Decisions Based on MRD Improves Outcomes (2021)
**Citation:** Leukemia 2021
**Core Claim:** MRD-guided treatment decisions, including continuation/escalation, result in improved progression-free and overall survival compared to MRD-unguided care.
**Key Details:**
- "Sustained" MRD negativity is critical prognostic factor
- Durable remissions associated with deep MRD response
- Actionable biomarker for treatment adaptation

#### 6.3 Measurable Residual Disease Dynamics and AI (2024)
**Citation:** Blood Cancer Journal 2024
**Core Claim:** AI analysis of MRD dynamics and clonal diversity predicts relapse risk and treatment resistance.
**Key Details:**
- Machine learning on serial MRD assessments
- Clonal evolution tracking
- Early detection of treatment resistance

#### 6.4 MRD-Guided Therapy in Newly Diagnosed Myeloma (2024)
**Citation:** New England Journal of Medicine 2024
**Core Claim:** Prospective randomized trial of MRD-guided vs fixed-duration therapy shows survival benefit of adaptive MRD-based treatment strategies.
**Key Details:**
- Response-guided treatment decisions
- Reduces unnecessary treatment toxicity
- Improves long-term outcomes

---

### Domain 7: Routine Blood Work for Cancer Risk Prediction

#### 7.1 AI Tool Using Routine Blood Tests to Predict Immunotherapy Response (2025)
**Citation:** Memorial Sloan Kettering/Tisch Cancer Institute 2025
**Core Claim:** SCORPIO model predicts immunotherapy response and prognosis across cancer types using only routine clinical blood tests, outperforming current clinical tests.
**Key Details:**
- Uses widely available laboratory parameters
- Tested across multiple cancer types
- Outperforms traditional predictive biomarkers
- Scalable to global healthcare settings

#### 7.2 AI for Primary Care Cancer Risk Prediction from Routine Labs (2021)
**Citation:** BMJ Open 2021
**Core Claim:** Machine learning on routine laboratory tests identifies cancer risk within 90 days with high AUC, providing easy-to-use risk score in primary care.
**Key Details:**
- Accessible blood work data
- 90-day prediction window
- Primary care applicable
- Bridges diagnosis gap

#### 7.3 Machine Learning for Lung Cancer Screening via Routine Labs (2025)
**Citation:** Cancer Medicine 2025
**Core Claim:** ML models using routine blood tests and tumor markers enhance early screening for bronchogenic lung cancer with improved detection rates.
**Key Details:**
- Accessible biomarkers (CBC, tumor markers)
- Early detection capability
- Routine clinical data integration

#### 7.4 Cervical Cancer Prediction from Routine Blood Analysis (2025)
**Citation:** Scientific Reports 2025
**Core Claim:** Machine learning on routine blood hematology parameters predicts cervical cancer risk with individualized stratification.
**Key Details:**
- Routine CBC parameters
- Cost-effective screening tool
- Integrates hematological data

#### 7.5 Artificial Intelligence in Routine Blood Tests (2024)
**Citation:** Frontiers in Medical Engineering 2024
**Core Claim:** Comprehensive review of AI applications to routine blood work, demonstrating potential for disease prediction across diverse conditions.
**Key Details:**
- Multiple disease applications
- Routine lab data utilization
- Cost-effective approach

---

### Domain 8: CoMMpass Dataset & Real-World Evidence

#### 8.1 MMRF-CoMMpass Data Integration and Analysis (2020)
**Citation:** PMC 2020
**Core Claim:** CoMMpass is the largest and most complete MM longitudinal genomic-clinical dataset (1100+ patients, 8-year follow-up with 6-month intervals), with 600+ researchers using it globally.
**Key Details:**
- Baseline: RNA-seq, cytogenetics, QoL surveys
- Longitudinal: tissue, genetics, outcomes every 6 months
- Data released in staged interim releases (Jan/Jul 15)
- Access via dbGaP

#### 8.2 MMRFBiolinks R-Package for CoMMpass (2021)
**Citation:** PubMed 2021
**Core Claim:** MMRFBiolinks extends TCGABiolinks with 13 new functions for MMRF-CoMMpass data analysis, enabling RNA, genomic, and clinical data integration.
**Key Details:**
- Bioinformatics infrastructure
- Data integration tools
- Facilitates reproducible analyses

#### 8.3 CoMMpass as Open Data Resource (AWS) (2024)
**Citation:** Registry of Open Data on AWS
**Core Claim:** CoMMpass data available as open access resource on AWS, enabling global computational research and reproducibility.
**Key Details:**
- Cloud-accessible data
- Enables large-scale analysis
- Community research facilitation

#### 8.4 Real-World Evidence in Multiple Myeloma Registries (2024)
**Citation:** Blood Cancer Journal 2024
**Core Claim:** Real-world registries (Connect MM, HUMANS, INSIGHT MM) document significant outcome gaps vs clinical trials (75% higher mortality in RWE cohorts), indicating need for RWE-informed prognostic models.
**Key Details:**
- Real-world cohorts older, more comorbid
- 13-year outcome gaps documented
- Treatment pattern variability
- Regulatory utility of RWE increasing

#### 8.5 Nordic Registry Analysis of MM Outcomes (2024)
**Citation:** Critical Reviews in Oncology/Hematology 2024
**Core Claim:** Linked Danish/Finnish/Swedish registries show treatment pattern heterogeneity and outcome variation, emphasizing need for adaptable prognostic models.
**Key Details:**
- Population-level outcome data
- 2009-2013 treatment cohorts
- Treatment evolution tracking

---

### Domain 9: Clinical AI Survey & Methodology Review

#### 9.1 Artificial Intelligence in Healthcare: 2024 Year in Review (2025)
**Citation:** medRxiv 2025.02.26
**Core Claim:** 2024 saw AI/ML advancement from internal validation focus to external validation and implementation trials, with foundation models driving healthcare NLP and imaging applications.
**Key Details:**
- Maturation toward clinical deployment
- LLM application expansion
- Ambient documentation AI adoption >50% of health systems
- Implementation challenges emerging

#### 9.2 Deep Learning in Healthcare: Disease Diagnosis & Treatment (2024)
**Citation:** World Journal of Advanced Research and Reviews 2024
**Core Claim:** Deep learning (CNN, RNN, Transformers) enhances disease diagnosis, personalized treatment, and clinical decision-making, with transformer architectures gaining prominence.
**Key Details:**
- Imaging, time-series, NLP applications
- Multi-modal data integration
- Personalized medicine emphasis

#### 9.3 Adoption of AI in Health Systems: Survey of Priorities & Challenges (2024)
**Citation:** PMC 2024
**Core Claim:** Leading US health systems report high adoption in ambient clinical documentation but variable success in other AI applications, with barriers including validation, integration, and trust.
**Key Details:**
- Documentation AI: high adoption
- Prediction models: lower adoption rates
- Regulatory and trust barriers
- Implementation infrastructure needs

#### 9.4 Transformers and Large Language Models for ECG Diagnosis (2025)
**Citation:** Artificial Intelligence Review 2025
**Core Claim:** Survey of transformer and LLM applications to ECG diagnosis demonstrates state-of-the-art performance with interpretability advantages, setting precedent for time-series healthcare AI.
**Key Details:**
- Transformer architecture advantages for sequences
- Multi-task learning capabilities
- Interpretability improvements
- Clinical integration challenges

---

### Domain 10: Explainability & Interpretability in Clinical AI

#### 10.1 SHAP and LIME for Model Interpretability (2025)
**Citation:** Advanced Intelligent Systems 2025
**Core Claim:** SHAP (game-theory-based) and LIME (local surrogate) are dominant XAI methods; SHAP provides global+local explanations while LIME is local-only, with SHAP more reliable for feature attribution.
**Key Details:**
- SHAP: Shapley value-based attribution
- LIME: local linear approximation
- SHAP advantages in consistency, completeness
- LIME advantages in computation speed
- Both limited in causal inference

#### 10.2 Interpreting AI Models in Alzheimer's Detection (2024)
**Citation:** PMC 2024
**Core Claim:** SHAP and LIME enable clinician-friendly interpretation of black-box AI models in neurodegeneration diagnosis, critical for clinical adoption.
**Key Details:**
- Feature importance transparency
- Disease mechanism insights
- Clinical decision support
- Trust building

#### 10.3 Explainable AI in Healthcare Review (2024)
**Citation:** Computational Pathology 2024
**Core Claim:** XAI is essential for clinical acceptance, but limitations in uncertainty quantification, generalization, and causal inference remain open challenges.
**Key Details:**
- Clinical acceptance driver
- Trust and transparency requirements
- Limitations in uncertainty estimation
- Non-causal attribution methods

#### 10.4 SHAP & LIME in Depression/Stroke Comorbidity Prediction (2025)
**Citation:** PMC 2025
**Core Claim:** Applied XAI methods to nutritional epidemiology reveal feature interactions driving comorbidity risk, exemplifying practical interpretability for personalized medicine.
**Key Details:**
- Nutrient-gene-disease interactions
- Population-level insights
- Clinical actionability

---

## CLUSTER ANALYSIS: Shared Assumptions & Contradictions

### Cluster A: "AI Outperforms Traditional Staging"
**Members:**
- Joint AI-driven Event Prediction (2024)
- Genomic Individualized Prediction (2024)
- Gene Expression-Based Models (2024)
- Routine Blood Work Model (2025)

**Shared Assumption:** Machine learning and deep learning models on clinical/genomic data inherently outperform rule-based staging systems (ISS/R-ISS) in survival prediction.

**Mechanism:** Integration of multivariate data, non-linear relationships, genomic complexity captured by algorithms not encoded in staging rules.

**Confidence:** HIGH - Consistently replicated across domains

---

### Cluster B: "Foundation Models Superior for Small Data"
**Members:**
- TabPFN (Nature 2024)
- TabPFN Architecture (2022)

**Shared Assumption:** Pre-trained foundation models achieve state-of-the-art performance on small-to-medium tabular datasets without fine-tuning.

**Mechanism:** Prior meta-learning, in-context learning paradigm, transformer architecture advantages.

**Confidence:** MEDIUM-HIGH - Benchmarked but challenged by 2026 clinical comparison

---

### Cluster C: "Temporal Models Enhance Long-Horizon Prediction"
**Members:**
- TFT for Intraoperative BP (2024)
- TFT for ICU Vital Signs (2025)
- Multivariate Cascaded Fine-Tuning (2025)
- Longitudinal Disease Progression Models (2023-24)

**Shared Assumption:** Attention-based temporal models (Transformers, TFT) outperform static/RNN baselines for multi-horizon clinical forecasting, especially with patient-level adaptation.

**Mechanism:** Attention captures complex temporal dependencies, handles irregular sampling, integrates static+dynamic covariates, probabilistic uncertainty quantification.

**Confidence:** HIGH - Consistent across multiple clinical domains

---

### Cluster D: "Digital Twins Enable Personalized Simulation"
**Members:**
- Digital Twins for Oncology (2025)
- Paradigm Shift for Precision Care (2021)
- Exploring Twins in Cancer (2025)
- Precision Treatment via Twins (2024)

**Shared Assumption:** Virtual patient models integrating multi-dimensional data enable in silico simulation of treatment response and personalized medicine, despite current clinical immaturity.

**Mechanism:** Patient-specific data integration (genomics, imaging, labs, outcomes), mechanistic/AI modeling of treatment effects, real-time adaptation capability.

**Confidence:** MEDIUM - Promising but not clinically validated; gap between concept and implementation

---

### Cluster E: "Routine Labs Sufficient for Risk Prediction"
**Members:**
- Routine Blood Work for MM Progression (2025)
- SCORPIO for Immunotherapy Response (2025)
- AI for Primary Care Cancer Risk (2021)
- Lung Cancer Screening via Labs (2025)
- Cervical Cancer from Hematology (2025)

**Shared Assumption:** Readily accessible routine laboratory parameters (CBC, metabolic panel, enzymes) enable accurate disease risk and treatment response prediction without genomic or imaging data.

**Mechanism:** Statistical/ML capture of indirect disease biomarkers; cost-effectiveness; accessibility; temporal change in labs reflects disease state.

**Confidence:** HIGH - Demonstrated across multiple cancer types and conditions

---

### Cluster F: "External Validation Critical for Generalization"
**Members:**
- GMMG-MM5 External Validation (2025)
- External Validation Framework (2024)
- Clinical Trial vs RWE Outcomes (2024)
- Data Drift & Calibration (2023-24)
- Real-World Registries (2024)

**Shared Assumption:** Models developed on one cohort (e.g., CoMMpass NDMM) must be externally validated on independent cohorts (e.g., GMMG RRMM, real-world registry) to assess generalization; performance often substantially declines.

**Mechanism:** Data distribution shift (patient selection, treatment patterns, era effects), unmeasured confounding, measurement differences across sites.

**Confidence:** VERY HIGH - Consensus across clinical prediction literature; backed by regulatory agencies (FDA)

---

### Cluster G: "MRD Negativity Predicts Durable Remission"
**Members:**
- MRD Assessment Review (2023)
- MRD-Guided Treatment Decisions (2021)
- MRD Dynamics & AI (2024)
- MRD-Guided Therapy RCT (2024)

**Shared Assumption:** Achievement and maintenance of MRD negativity (detected via flow cytometry or NGS) is the strongest independent prognostic factor for progression-free and overall survival in MM.

**Mechanism:** MRD reflects residual clonal burden; sustained negativity indicates deep response and durable remission likelihood.

**Confidence:** VERY HIGH - Randomized trials and real-world evidence consistent; IMWG consensus

---

## DOCUMENTED CONTRADICTIONS

### Contradiction 1: TabPFN Clinical Superiority
**Position A (Nature 2024):** "TabPFN outperforms all previous methods on small-to-medium datasets, achieving state-of-the-art across diverse domains."
**Source:** Schlag et al., Nature (2024) on TabPFN foundation model

**Position B (medRxiv 2026):** "Established ML matches or exceeds TabPFN on clinical tasks; TabPFN exceeded best baseline in only 16.7% of 12 tasks, with clinically insignificant AUROC differences (±0.01)."
**Source:** Clinical benchmark study, medRxiv 2026

**Explanation:**
- **Methodological:** Nature paper used synthetic/benchmark datasets; clinical study used real clinical tasks with different data characteristics (sparse features, missing data, class imbalance)
- **Domain specificity:** TabPFN may excel on curated benchmarks but struggle with clinical data complexities
- **Hardware:** Clinical study noted GPU requirement makes practical deployment 5.5× slower, contradicting speed advantage
- **Implication:** TabPFN promising but not yet a game-changer for clinical tabular prediction vs traditional ensemble methods

---

### Contradiction 2: AI Model Complexity vs Routine Labs Sufficiency
**Position A (AI Outperforms Traditional Staging):** "Genomic sequencing, transformer models on multi-modal data, and sophisticated ensemble methods are necessary to exceed ISS performance."
**Sources:** Joint AI-driven prediction papers (2024), Genomic Classification (2024)

**Position B (Routine Labs Sufficient):** "Simple routine blood work + basic ML achieves AUROC 0.87-0.88 for MM progression prediction, matching or exceeding complex genomic models."
**Sources:** Routine Blood Work Model (2025, npj Digital Medicine), SCORPIO (2025)

**Explanation:**
- **Data dimensionality trade-off:** Routine labs are lower-dimensional but highly predictive; genomics add complexity with marginal gains in some tasks
- **Temporal dynamics:** Routine labs capture disease state changes over time; static genomics don't evolve post-treatment
- **Clinical practicality:** Cost, accessibility, standardization favor routine labs; genomics require specialized equipment and infrastructure
- **Task-dependent:** Genomics superior for baseline risk stratification; routine labs superior for progression prediction post-treatment
- **Implication:** Complementary, not contradictory—different use cases warrant different data modalities

---

### Contradiction 3: Foundation Models for Healthcare
**Position A (Maturation Narrative):** "2024 saw healthcare AI mature toward external validation and clinical implementation with foundation models (LLMs, TabPFN) driving advances."
**Sources:** AI in Healthcare 2024 Review (2025), TabPFN Nature (2024)

**Position B (Implementation Barriers):** "Health system AI adoption remains limited outside ambient documentation; prediction models face validation, integration, and trust barriers; implementation infrastructure inadequate."
**Sources:** Health Systems Adoption Survey (2024)

**Explanation:**
- **Publication bias:** Top-tier papers report successes; health system surveys capture real-world friction
- **Translation gap:** Academic achievements ≠ clinical deployment; many advances remain in research settings
- **Era/timeline:** Nature (2024) discusses promise; Health Systems survey (2024) documents current slow adoption rates
- **Implication:** Foundation models are advancing but clinical translation slower than academic publications suggest; implementation is the true bottleneck

---

### Contradiction 4: Real-World Evidence vs Clinical Trial Generalizability
**Position A (External Validation Philosophy):** "Model validation on independent cohorts is essential; clinical trial populations and real-world cohorts differ fundamentally (age, comorbidity, treatment intensity)."
**Sources:** External Validation Framework (2024), Clinical Prediction Evaluation (2024)

**Position B (Paradoxical RWE Gap):** "Real-world outcomes for MM are 75% worse than clinical trials, yet real-world registries are now informing regulatory approvals, suggesting RWE cannot be simply 'generalized' from trial data—requires distinct models."
**Sources:** Real-World Outcomes Gap (2024), Regulatory Use of RWE (2025)

**Explanation:**
- **Causal heterogeneity:** The gap isn't sampling error; it reflects fundamental differences in patient populations, treatment delivery, and selection effects
- **Regulatory evolution:** FDA increasingly requires RWE not as validation of trial findings, but as distinct evidence for real-world effectiveness
- **Model strategy:** Implications suggest prognostic models should be developed on RWE data from the start, not post-hoc validated; CoMMpass may be more representative than selected clinical trial cohorts
- **Implication:** Traditional validation paradigm (trial→real-world) insufficient; parallel or RWE-first development may be needed for MM

---

## INTERDEPENDENCIES & TENSIONS

### Tension 1: Genomics Complexity vs Lab Simplicity
- **Routine labs model (0.87 AUROC)** uses ~30-40 accessible parameters
- **Genomic models** use 100+ driver genes but show marginal improvement
- **Cost-benefit:** Routine labs 1000× cheaper, more accessible, temporal tracking possible
- **Unresolved:** Under what conditions does genomic complexity justify cost? Baseline risk? Treatment planning?

### Tension 2: Static vs Dynamic Prediction
- **ISS, genomic subtypes** are baseline (static) predictors with fixed risk stratification
- **Routine labs + temporal fusion** enable dynamic prediction, adjusting risk as patient evolves
- **Unresolved:** For clinical MM management, is baseline stratification + adaptive intensity enough? Or does dynamic tracking require continuous monitoring?

### Tension 3: Interpretability vs Performance
- **Tree-based models** (Random Forest, XGBoost) are interpretable, competitive with deep learning on tabular data
- **Transformers, TabPFN** claim better performance but add opacity (mitigated by SHAP/LIME)
- **Clinical adoption:** Gradient boosting dominates clinical predictions despite transformer hype
- **Unresolved:** Is interpretability cost worth the modest performance gain in transformer models?

---

## KEY GAPS IN LITERATURE

1. **No consensus on optimal data modality combination** — Should MM progression models use routine labs, genomics, both? When to add imaging/MRD?
2. **Limited prospective external validation** — Most models tested on historical/retrospective data; few prospective trials
3. **Treatment heterogeneity not addressed** — Models don't account for evolving treatment options (CAR-T, bispecific antibodies)
4. **Temporal generalization unclear** — Models trained on 2015-2020 data; performance on 2024+ patients unknown
5. **Cost-effectiveness not evaluated** — No studies comparing cost/benefit of routine labs vs genomics for MM models

---

## REFERENCE SUMMARY BY TYPE

| **Type** | **Count** | **Quality** | **Clinical Relevance** |
|----------|-----------|-----------|--------|
| Deep Learning/Transformer Models | 8 | HIGH | Very relevant (main innovation) |
| Genomic/Molecular Biology | 6 | HIGH | Core to MM pathobiology |
| Classical ML Benchmarks | 5 | MEDIUM | Comparative baseline |
| Clinical Trials/Outcomes | 8 | VERY HIGH | Gold standard evidence |
| Real-World Evidence/Registries | 5 | HIGH | Implementation relevant |
| Foundation Models (TabPFN) | 3 | MEDIUM | Emerging, clinically unproven |
| Temporal Modeling/TFT | 5 | MEDIUM-HIGH | Promising, limited MM-specific data |
| Interpretability/XAI | 4 | MEDIUM | Important for adoption |
| **TOTAL** | **44** papers/resources | - | - |

