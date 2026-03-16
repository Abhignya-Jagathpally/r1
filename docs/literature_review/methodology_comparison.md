# Methodology Comparison: Research Designs in MM Clinical AI Literature

## Executive Summary

Comprehensive analysis of 44 papers across 5 methodological categories reveals:
- **Dominant methodology:** Observational retrospective ML/DL model development on existing datasets
- **Underused methodology:** Prospective clinical intervention trials, health economic analysis, implementation science
- **Quality concern:** External validation increasingly reported but rigorous prospective validation (RCT) absent in MM

---

## Methodological Classification

### Category 1: Model Development & Internal/External Validation (N=18 papers)
**Definition:** Develop prediction model on historical data; validate on separate cohort(s); report performance metrics (AUROC, c-index, calibration).

#### Subcategories & Examples:

**1a. Internal Validation Only (N=5)**
- Papers: Gene expression GEP models, genomic classification (90 genes)
- Approach: Train/test on same cohort (or k-fold CV); no external validation
- Quality issue: High risk of overfitting; limited generalization evidence
- **Example:** Genomic Individualized Prediction (2024) — identified 90 driver genes; no external validation mentioned

**1b. Single External Validation (N=8)**
- Papers: Joint AI-driven event prediction, Routine blood work prediction
- Approach: Train on discovery cohort; validate on single independent cohort
- Quality: Improved over 1a; generates external AUROC
- **Example:** Routine Blood Work (npj DM 2025) — trained on CoMMpass (N=1,186), externally validated on GMMG-MM5 (N=504); AUROC 0.87 ± 0.01
- **Strength:** Demonstrates generalization to different patient population
- **Weakness:** Single external cohort; may not generalize to different eras/treatments

**1c. Multiple External Validation (N=5)**
- Papers: R2-ISS (revised staging system), International Staging System papers
- Approach: Validate on 2+ independent cohorts; report performance across populations
- Quality: Highest standard for non-RCT evidence
- **Example:** R2-ISS validation on multiple myeloma cohorts across continents
- **Strength:** Multiple population validation; geographic generalizability shown
- **Weakness:** Still not prospective; can't show if implementation changes outcomes

#### Statistical Rigor Assessment:
| Metric | 1a (Internal) | 1b (Single External) | 1c (Multiple External) |
|--------|----------|----------|---------|
| Overfitting risk | VERY HIGH | Medium | Low |
| Generalization evidence | None | One cohort | Multiple populations |
| Clinical readiness | Low | Medium | High |
| Papers in MM literature | 5 | 8 | 5 |

---

### Category 2: Benchmarking & Comparative Studies (N=10 papers)
**Definition:** Compare multiple algorithms/approaches on same dataset(s); evaluate relative performance.

#### Subcategories:

**2a. Algorithm comparison on standard benchmarks (N=4)**
- Papers: ML methodology comparisons (XGBoost vs. RF vs. neural nets), TabPFN benchmarks
- Approach: Run multiple algorithms on curated datasets; compare AUROC/AUC
- **Example:** Nature 2024 TabPFN paper (Schlag et al.) — compared TabPFN vs. 10+ baselines on Penn ML Benchmarks; showed TabPFN wins
- **Example:** Comprehensive Benchmark 2024-25 — compared 20 models on 111 tabular datasets
- **Strength:** Systematic, reproducible, identifies best-in-class methods
- **Weakness:** Benchmarks curated and clean; may not reflect clinical data complexity; don't necessarily show clinical utility

**2b. Clinical algorithm comparison (N=4)**
- Papers: Established ML vs. TabPFN in clinical tasks, tree vs. neural network clinical prediction
- Approach: Compare algorithms on real clinical data
- **Example:** medRxiv 2026 clinical benchmark — tested TabPFN vs. 12 established ML methods on 12 binary clinical prediction tasks; found TabPFN not consistently superior
- **Example:** Acute kidney injury prediction — compared logistic regression, RF, SVM, gradient boosting on clinical cohort
- **Strength:** Relevant to actual clinical deployment
- **Weakness:** Small number of clinical tasks tested; algorithm-specific optimization varies

**2c. Conceptual framework comparison (N=2)**
- Papers: Survey of AI in healthcare, systematic review of digital twins
- Approach: Narrative/systematic review; compare existing approaches, summarize evidence
- **Example:** "Application of Digital Twins for Personalized Oncology" (Nature Reviews 2025)
- **Strength:** Broad synthesis, identifies trends
- **Weakness:** Qualitative; limited quantitative comparison

#### Methodological Strength:
- **Strong:** Benchmarking studies provide concrete performance comparisons; informative for algorithm selection
- **Weakness:** Benchmarks may not transfer to clinical practice; TabPFN dominates benchmarks but underperforms on clinical data (Gap #2 in research gaps)

---

### Category 3: Mechanistic & Biological Studies (N=8 papers)
**Definition:** Focus on disease biology, genomic mechanisms, biomarker identification; less focus on prediction model performance.

#### Subcategories:

**3a. Genomic/mutational analysis (N=5)**
- Papers: "Genomic landscape highlights prognostic markers," "Genomic classification," "Whole-exome sequencing mutational signatures"
- Approach: NGS data → identify driver genes, mutations, copy number variants; associate with outcomes
- **Method:** Univariate/multivariate Cox regression; survival curves by mutational status
- **Example:** Analysis of genomic landscape (Leukemia 2018) — identified 90 driver genes and 12 molecular subtypes using NGS data from MM cohort
- **Strength:** Uncovers biological heterogeneity; identifies actionable mutations (TP53, KRAS)
- **Limitation:** Typically univariate analysis; doesn't address prediction model generalization; external validation limited

**3b. Biomarker discovery/validation (N=3)**
- Papers: MRD assessment reviews, CAR-T biomarker prediction, ferroptosis-related genes
- Approach: Identify candidate biomarker; validate association with outcomes
- **Example:** MRD assessment (Haematologica 2023) — compared detection methods (flow vs. NGS), established thresholds, validated prognostic value
- **Example:** CAR-T biomarker paper — identified candidate biomarkers (ALC, endothelial markers, immune markers) associated with response
- **Strength:** Clinically actionable; identifies treatment response markers
- **Limitation:** Often don't address utility for individual prediction; focus on group-level associations

#### Methodological Quality:
- **Strong:** Rigorous molecular biology, hypothesis-driven, mechanistically informative
- **Weakness:** Limited prediction accuracy emphasis; survival curves more common than AUROC/discrimination metrics

---

### Category 4: Clinical Trials & Prospective Intervention Studies (N=5 papers)
**Definition:** Prospective cohort studies or randomized trials; measure outcomes in real-time.

#### Subcategories:

**4a. RCTs of outcome-modifying interventions (N=2)**
- Papers: MRD-guided therapy RCT (NEJM 2024), CAR-T response prediction studies
- Approach: Randomize patients to two treatment strategies; measure PFS/OS
- **Example:** MRD-Guided Therapy RCT (NEJM 2024, Sonneveld et al.) — randomized to MRD-guided vs. fixed-duration therapy; showed PFS improvement with MRD-guided approach
- **Strength:** Gold standard evidence; directly measures clinical outcome improvement; prospective
- **Limitation:** Only 2 RCTs in MM AI/prediction space; limited to MRD-guided therapy, not AI predictor validation

**4b. Prospective observational cohorts (N=3)**
- Papers: Connect MM Registry, HUMANS registry (Nordic linked data), INSIGHT MM
- Approach: Prospectively enroll patients; follow longitudinally; document outcomes
- **Example:** Connect MM Registry — US multicenter prospective cohort of 500+ NDMM patients; tracks treatment patterns, outcomes, QoL annually for 10 years
- **Strength:** Real-world data; unselected populations; long follow-up; outcomes directly observed
- **Limitation:** No randomization; observational (confounding by indication); limited to single-arm design

**4c. Real-world outcomes analysis (N=0 true RCT of AI predictor)**
- Papers: "Real-world treatment patterns and outcomes" studies
- Approach: Retrospective analysis of treatment patterns in unselected cohorts; outcomes vs. clinical trials
- **Example:** RWE vs. clinical trials (2024) — documented 75% higher mortality in real-world vs. trials
- **Strength:** Identifies real-world outcome gaps; informs model selection/adjustment
- **Limitation:** Retrospective; can't determine if intervention (AI, treatment modification) improves outcomes

#### Prospective Trial Status in MM AI:
| Trial Type | Count | Example | Outcome |
|-----------|-------|---------|---------|
| **AI predictor RCT** | 0 | None | **Critical gap** |
| **MRD-guided RCT** | 1 | Sonneveld 2024 | Positive; PFS benefit |
| **CAR-T selection RCT** | 0 (in progress) | CARTITUDE trials | Treatment-specific; not predictor-focused |
| **Prospective cohort** | 3+ | Connect MM | Ongoing; outcomes improving |

**Major Gap:** No prospective RCT of AI progression predictor in MM. This is critical gap #4 in research gaps document.

---

### Category 5: Health Economic Analysis & Implementation Science (N=3 papers)
**Definition:** Cost-effectiveness, implementation barriers, workflow integration, implementation readiness.

#### Subcategories:

**5a. Health economic studies (N=1)**
- Papers: Limited explicit cost-effectiveness analyses in MM AI literature
- Approach: Calculate cost per QALY, cost per AUROC point, ROI
- **Current state:** Minimal; most papers don't report costs
- **Example:** TabPFN medRxiv 2026 — noted GPU computational cost 5.5× higher than traditional ML (implicit health economic finding)
- **Implication:** Cost-benefit tradeoff between genomic vs. routine labs unclear; suggests need for formal economic analysis

**5b. Implementation science & adoption surveys (N=1)**
- Papers: "Adoption of AI in health systems" survey (2024)
- Approach: Survey health system leaders on AI adoption, barriers, successes
- **Findings:**
  - Ambient documentation AI: >50% adoption
  - Prediction models: ~20% high-success adoption
  - Barriers: Validation requirements, integration friction, regulatory uncertainty, clinician trust
- **Strength:** Documents real-world implementation gap; identifies barriers
- **Weakness:** Descriptive; doesn't propose solutions; doesn't measure outcome improvements

**5c. Systematic reviews of implementation barriers (N=1)**
- Papers: "Explainable AI in healthcare review," "Digital twins barriers review"
- Approach: Synthesize literature on barriers to clinical deployment
- **Barriers identified:** Black-box models, validation burden, regulatory uncertainty, workflow friction, liability concerns
- **Strength:** Comprehensive overview; useful for program planning
- **Weakness:** Qualitative; doesn't quantify frequency/severity of barriers

#### Implementation Science Status:
- **Critical gap:** Very few health economic analyses; implementation science studies minimal
- **Implication:** Unknown whether AI improvements in AUROC translate to clinically worthwhile cost-effectiveness
- **Recommendation:** Future work should include formal health economic analysis alongside clinical validation

---

## Methodological Dominance Analysis

### By Frequency:

```
Internal/External Validation Studies     18 papers (41%)  [DOMINANT]
Benchmarking & Comparative              10 papers (23%)
Mechanistic/Biological Studies           8 papers (18%)
Clinical Trials & Prospective            5 papers (11%)
Health Economics & Implementation        3 papers (7%)
```

### Dominant Methodology in MM: Retrospective Model Development + Single External Validation

**Typical workflow:**
1. Access historical data (CoMMpass, clinical trial, registry)
2. Develop algorithm (gradient boosting, neural net, ensemble)
3. Internal validation (k-fold CV or holdout test set)
4. External validation (apply to GMMG-MM5 or similar)
5. Report AUROC, compare to baseline (ISS)
6. Publish in top journal (npj Digital Medicine, Journal of Clinical Oncology)
7. **Stop.** No prospective clinical trial; no outcome study.

**Advantages:**
- Fast (12-18 months from data access to publication)
- Low cost (~$100-200K total)
- Can leverage existing data; ethical (no randomization needed)
- Publishable; impacts research community

**Disadvantages:**
- Validation limited to research populations; unknown real-world performance
- Can't show if implementation changes outcomes
- Doesn't address clinician adoption; workflow integration; cost-effectiveness
- Regulatory pathway unclear (is external validation sufficient for FDA approval?)

---

## Methodological Underuse Analysis

### Severely Underused: Prospective Clinical Validation

**Evidence of underuse:**
- **Internal validation:** 18 papers in literature
- **External validation:** 8 papers in literature
- **Prospective RCT of AI intervention:** 0 papers in MM

**Why underused:**
1. **Cost:** RCT budget ~$2-5M vs. retrospective study ~$100-300K
2. **Time:** RCT 3-5 years vs. retrospective 1-2 years
3. **Risk:** RCT may show negative results (AI doesn't improve outcomes)
4. **Regulatory clarity:** FDA guidance on AI in clinical trials evolving; unclear what's required

**Clinical consequences:**
- Unknown if AUROC improvements (0.78→0.87) translate to outcome benefits
- Unknown if clinicians will adopt and trust AI predictions
- Unknown optimal workflow for integrating AI into clinical decision-making
- Regulatory pathway for clinical adoption unclear

**Recommendation:** Prospective validation essential before clinical deployment. At minimum: single-arm prospective cohort (outcomes tracked with AI in use). Ideally: RCT (AI vs. standard care).

---

### Severely Underused: Health Economic Analysis

**Evidence of underuse:**
- **Papers with formal cost-effectiveness analysis:** ~1-2 in literature
- **Papers discussing cost-benefit of genomics vs. routine labs:** 0
- **Papers comparing cost per unit accuracy improvement:** 0

**Examples of missing analysis:**
- Genomics cost ~$2,000-5,000 per patient vs. routine labs ~$100
- Does 0.05 AUROC improvement from adding genomics justify 50× cost increase?
- What is acceptable cost per 1% improvement in PFS prediction?
- What is willingness-to-pay threshold for health systems?

**Clinical consequence:** Resource allocation decisions made without economic evidence. Health systems can't determine if investing in genomic sequencing for MM prognostication is cost-effective vs. routine labs + clinical judgment.

**Recommendation:** Formal health economic analysis should accompany all methodological comparison studies.

---

### Moderately Underused: Implementation Science

**Evidence:**
- Survey papers: 1 main adoption study (health systems survey 2024)
- Implementation barriers identified but not quantified
- Workflow integration studies: minimal
- Clinician decision-making impact: 0 formal studies

**What's missing:**
1. **Workflow integration studies:** How to integrate AI prediction into routine clinical practice? (e.g., EHR alert vs. standalone tool vs. clinical note recommendation?)
2. **Clinician perception studies:** What information improves trust? How to present predictions to increase adoption?
3. **Decision-making impact:** Does AI recommendation change treatment decisions? By how much? In which direction?
4. **Optimization of AI presentation:** What visualization/framing maximizes clinical utility?

**Clinical consequence:** Unknown if models improving AUROC will actually be used by clinicians. Risk of "shelf-ware" — excellent model that clinicians ignore.

**Recommendation:** Parallel implementation science track alongside model development. User research (interviews, usability testing) should inform model deployment.

---

## Quality Assessment by Methodology

### Grading Framework: Methodological Rigor & Clinical Relevance

```
STRENGTH                    METHODOLOGY TYPE                    PAPERS  CLINICAL RELEVANCE
1. GOLD STANDARD            RCT of AI intervention              0       HIGHEST (if positive)
2. STRONG                   Prospective cohort + outcomes       3       VERY HIGH
3. STRONG                   Multiple external validation        5       HIGH
4. MODERATE-HIGH            Single external validation          8       MODERATE-HIGH
5. MODERATE                 Benchmarking on clinical data       4       MODERATE
6. MODERATE                 Mechanistic/biology studies         8       MODERATE
7. MODERATE                 Internal validation only            5       MODERATE (overfitting risk)
8. MODERATE                 Benchmarking on synthetic data       4       LOW (generalization unknown)
9. WEAK-MODERATE            Real-world outcome analysis         1       MODERATE (observational)
10. WEAK                    Implementation science survey       2       MODERATE (no quantification)
```

### Specific Weakness Assessment: MM Literature

| Weakness | Affected Papers | Severity | Impact |
|----------|----------|--------|--------|
| **No prospective RCT of AI predictor** | All model development papers (18) | CRITICAL | Can't prove clinical benefit |
| **Limited cost-effectiveness analysis** | All methodology papers | HIGH | Resource allocation decisions uninformed |
| **Minimal implementation science** | All deployment-focused papers | HIGH | Unknown clinician adoption barriers |
| **Limited ablation studies** | Multi-modal model papers (5-6) | MEDIUM | Unclear which data modalities essential |
| **Internal validation only** | Genomic classification papers (5) | MEDIUM | Overfitting risk; generalization unclear |
| **Single external validation** | Most model papers (8) | LOW-MEDIUM | Good but single cohort; limited generalization |
| **Lack of calibration analysis** | Some clinical prediction papers | LOW | Model miscalibration in real-world populations unknown |

---

## Methodological Recommendations for MM Digital Twin Program

### For Model Development (Phase 1-2):

1. **Standard: Retrospective Model Development + Multiple External Validation**
   - Train on CoMMpass (primary) or other large cohort
   - Validate on ≥2 independent external cohorts (GMMG-MM5, Connect MM, GMMG registry)
   - Report: AUROC/c-index with 95% CI, calibration plots, decision curves
   - Perform ablation study: contribution of each data modality
   - Timeframe: 18-24 months

2. **Health Economics (concurrent):**
   - Cost per patient by data modality (labs vs. genomics vs. MRD)
   - Cost per unit accuracy improvement (cost/AUROC point)
   - Comparison to standard ISS (cost-effectiveness threshold)
   - Timeframe: 6-12 months

### For Implementation (Phase 3-4):

3. **Prospective Cohort (preliminary clinical validation):**
   - Single-arm prospective enrollment of 100-200 patients
   - Deploy AI prediction; track outcomes, treatment decisions, clinician adherence
   - Measure: Prediction accuracy in real-world setting, clinician trust/adoption, decision impact
   - Timeframe: 12-24 months

4. **Randomized Clinical Trial (definitive validation):**
   - Multi-center RCT: AI-guided treatment vs. standard care
   - Primary outcome: PFS/OS improvement
   - Secondary: Toxicity, QoL, cost-effectiveness
   - Timeframe: 36-48 months
   - Funding: $2-3.5M estimated

5. **Implementation Science (concurrent with trials):**
   - User research: clinician perceptions, workflow integration, decision-making
   - Optimize presentation: which visualization/framing maximizes utility?
   - Identify barriers: regulatory, technical, organizational
   - Timeframe: 12-36 months

### Methodology Hierarchy for MM Digital Twin:

```
PRIORITY  PHASE       METHODOLOGY                          RIGOR    TIMELINE    COST        DECISION POINT
1.        Dev         Retrospective model + external val   High     18-24 mo    $150-300K   Proceed to Impl?
2.        Dev         Health economics analysis            Medium   6-12 mo     $50-100K    Cost-effective?
3.        Impl        Prospective cohort (real-world)      High     12-24 mo    $300-500K   Clinician adoption?
4.        Impl        RCT of AI intervention               VERY HIGH 36-48 mo  $2-3.5M     Outcome benefit?
5.        Impl        Implementation science track         Medium   12-36 mo    $200-400K   Workflow optimal?
```

---

## Conclusion: Dominant Methodology is Insufficient for Clinical Adoption

**Current state (2025):** MM clinical AI literature dominated by retrospective model development + external validation. This has generated multiple AUROC 0.78-0.87 models.

**Methodological gap:** No prospective RCT evidence that implementation of these models improves patient outcomes.

**Implication for digital twin program:** Retrospective validation will support academic publication and researcher interest, but **insufficient for clinical adoption**. Health systems and regulators require:
1. Prospective evidence that model predicts outcomes in real-world setting
2. RCT evidence that model-guided care improves outcomes vs. standard
3. Health economic evidence that improvements justify costs
4. Implementation science evidence that model adoptable by clinicians

**Recommendation:** Plan digital twin program with both research track (models) and clinical validation track (prospective cohort → RCT). Don't assume retrospective external validation sufficient.

