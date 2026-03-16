# Research Gaps in Multiple Myeloma Clinical AI & Digital Twin Prediction

## Executive Summary
Five major unanswered questions limit MM digital twin development and clinical implementation. Each gap represents a 2-5 year research project opportunity.

---

## Gap 1: Optimal Data Modality Integration for MM Progression

### The Question
**Which combination of data modalities (genomics, routine labs, MRD, imaging, transcriptomics) maximizes progression prediction accuracy?**

### Why The Gap Exists

Current literature presents **incomplete comparisons**:
- Genomic models: c-index ~0.78-0.82 on discovery cohorts
- Routine labs model: AUROC 0.87 ± 0.01 on external GMMG-MM5
- **Direct head-to-head comparison on same population: MISSING**

Sources of confusion:
1. Different prediction tasks (baseline OS vs. progression events)
2. Different populations (NDMM vs. RRMM)
3. Different timeframes (3-month vs. 12-month horizons)
4. Different statistical metrics (c-index vs. AUROC)
5. No study tests additive value (genomics + labs + MRD together)

### Which Paper Came Closest

**Closest:** Joint AI-driven Event Prediction (npj DM 2024)
- Integrated clinical, genomic, and treatment data
- Trained on CoMMpass (NDMM) + externally validated on RRMM cohort
- Achieved superior performance to ISS
- **Still missing:** Direct comparison to routine labs-only model; ablation study showing contribution of each modality

**Also relevant:** Genomic Classification (JCO 2024)
- Shows genomic subtypes capture prognostic information
- **Missing:** Marginal benefit analysis vs. simpler clinical staging

### Methodology to Close Gap

**Design: Prospective multi-modal comparison study**

**Phase 1: Retrospective harmonization (12 months)**
- Collect CoMMpass + GMMG-MM5 + Connect MM registry data
- Standardize variable definitions (labs, genomics, MRD definitions)
- Create harmonized cohort with 5-year follow-up data
- Target: N ≥ 2,000 patients with complete multi-modal data

**Phase 2: Model development (6 months)**
- Develop separate prediction models for each modality:
  * Model A: Routine labs only (CBC, chemistry, LDH, Hgb, Cr) — ~30 features
  * Model B: Genomics only (R2-ISS factors + driver mutations) — ~20 features
  * Model C: MRD dynamics + labs — ~50 features
  * Model D: Combined (A+B+C) — ~100 features
- Use consistent methodology (gradient boosting, 5-fold CV)
- Hyperparameter tuning on discovery set (70%)

**Phase 3: Independent external validation (6 months)**
- Validate all models on held-out test set (30%)
- Report AUROC, Brier score, calibration curves
- Compare 3-month, 6-month, 12-month progression prediction
- Ablation study: Calculate marginal contribution of each modality

**Phase 4: Health economic analysis (3 months)**
- Cost per patient for each modality:
  * Labs: ~$100 per panel
  * Genomics: ~$2,000-5,000 per patient (one-time)
  * MRD: ~$300-500 per test
- Calculate cost per unit accuracy improvement ($/AUROC point)
- Evaluate clinical feasibility in different care settings (academic, community)

**Expected Outcome:**
- Quantify marginal benefit of adding genomics to routine labs: ±0.05 AUROC?
- Cost-effectiveness threshold: Is 0.05 AUROC improvement worth $2,000?
- Clinically actionable recommendation: Which combination optimal for baseline vs. progression monitoring?

### Potential Insight
**Hypothesis:** Routine labs sufficient for progression prediction (AUROC 0.85+); genomics add 0.02-0.05 at 20× cost. Optimal strategy: Labs for post-treatment monitoring; genomics for baseline risk + treatment selection.

---

## Gap 2: Treatment Heterogeneity in Prognostic Model Generalization

### The Question
**How much do MM prognostic models degrade in performance when treatment landscape changes (new therapies, dosing intensity, management strategies)?**

### Why The Gap Exists

Critical observation from real-world evidence:
- Clinical trial outcomes: Median OS ~8-10 years
- Real-world outcomes: Median OS ~5-6 years (~30% worse)
- **Root cause:** Not random variation; reflects treatment differences
  - Trial patients: Young (<70), fit, intensive induction + transplant
  - Real-world: Older (median 72), comorbidities, variable intensity
  - Changing treatments: CAR-T approval 2021 improved RRMM outcomes; not in CoMMpass training data (2009-2019)

**No study has prospectively tracked how MM prognostic models degrade with treatment evolution.**

### Which Paper Came Closest

**Close:** Real-World Evidence vs. Clinical Trials (2024)
- Documents 75% higher mortality in RWE vs. trials
- **Missing:** Mechanistic decomposition—what % of gap is:
  * Patient selection differences (age, comorbidity)?
  * Treatment intensity differences (induction, transplant, maintenance)?
  * Treatment modality changes (CAR-T, bispecific antibodies not in trial)?

**Also relevant:** CoMMpass Data Era Effects
- Data collected 2009-2019
- Treatments evolved dramatically (carfilzomib approval 2012, daratumumab 2015, CAR-T 2021)
- **Missing:** Subgroup analysis by era to show how model performance changes as treatment changes

### Methodology to Close Gap

**Design: Temporal generalization study (3-year prospective)**

**Phase 1: Retrospective cohort stratification (6 months)**
- Reanalyze CoMMpass cohort by treatment era:
  * Era 1 (2009-2012): Pre-carfilzomib, traditional IMiD/bortezomib regimens
  * Era 2 (2013-2015): Carfilzomib, pomalidomide introduced
  * Era 3 (2016-2019): Monoclonal antibodies (daratumumab), longer follow-up
  * Era 4 (2020+): CAR-T for RRMM, real-world data
- Target: 250-500 patients per era with complete follow-up

**Phase 2: Model stability analysis (6 months)**
- Train progression prediction model on Era 1-3 combined (knowledge cutoff 2019)
- Test on held-out Era 1-3 test set → baseline performance (AUROC X)
- Retrospectively test on early Era 4 patients (2020-2022) → degradation (AUROC Y)
- Quantify: ΔPerformance = X - Y

**Phase 3: Real-world prospective validation (24 months)**
- Prospectively enroll new MM patients (2026-2028) in health systems using digital twin
- Capture: Treatment choices (CAR-T vs. traditional, maintenance strategies)
- Compare predicted vs. actual progression rates by treatment received
- Measure: Does model maintain calibration as treatment patterns shift?

**Phase 4: Mechanistic attribution (6 months)**
- Analyze degradation sources:
  * Compare patients receiving "old regimen" (Era 3) vs. "new regimen" (CAR-T)
  * Measure: Does same patient cohort stratified by treatment received show different outcomes?
  * Estimate: Contribution of unmeasured treatment effects to model degradation

**Expected Outcome:**
- Quantify: How much does 5-year treatment evolution degrade model performance?
- Hypothesis: 0.05-0.10 AUROC decline per era due to unmeasured treatment changes
- Actionable: How often should models be retrained? (annually? biannually?)
- Regulatory: What level of temporal validation required for FDA approval of AI predictor?

### Potential Insight
**Hypothesis:** MM models require continuous retraining as treatment landscape evolves. One-time development insufficient. Recommend annual retraining with new treatment cohorts.

---

## Gap 3: MRD-Guided Adaptive Therapy Optimization Algorithm

### The Question
**What is the optimal timing, intensity, and treatment algorithm for MRD-guided adaptive management, and how to predict which MRD+ patients will achieve delayed MRD negativity vs. those destined to relapse?**

### Why The Gap Exists

Strong evidence for MRD-negativity prognostic value, BUT:
1. **No consensus on "when to escalate":** MRD-negative at month 3 vs. month 6 vs. month 12?
2. **No prediction of response trajectory:** Which MRD+ patients will eventually clear vs. persist?
3. **No optimization of escalation intensity:** When MRD+ detected, should escalation be:
   * Immediate vs. wait-and-see?
   * Dose intensification vs. drug switch?
   * Single-agent addition vs. combination therapy?
4. **Limited RCTs on MRD-guided escalation:** Only Larocca et al. (2021) with modest sample

### Which Paper Came Closest

**Closest:** MRD-Guided Therapy RCT (2024, NEJM)
- Prospective RCT of MRD-guided vs. fixed-duration therapy
- Showed benefit of MRD-based treatment stopping
- **Missing:** Inverse problem—what to do when MRD+ (escalation algorithm)
- **Missing:** Prediction of response trajectory from baseline MRD kinetics

**Also relevant:** MRD Dynamics + AI (Blood Cancer Journal 2024)
- Serial MRD measurements can predict relapse
- **Missing:** Quantitative algorithm for escalation timing/intensity

### Methodology to Close Gap

**Design: Machine learning on MRD kinetics + prospective validation (3-year)**

**Phase 1: Retrospective MRD kinetics analysis (6 months)**
- CoMMpass subset with serial MRD measurements (N=200-300)
- Extract features from MRD trajectories:
  * MRD at months 3, 6, 12
  * MRD slope (rate of decline)
  * MRD volatility (measurement variability)
  * Time to MRD negativity
  * Time to sustained MRD negativity (≥3 consecutive measures)
- Clinical outcomes:
  * Time to progression (event definition: biochemical relapse or clinical progression)
  * Progression-free survival
  * Respond to escalation (yes/no if escalated when MRD+)

**Phase 2: Predictive model development (6 months)**
- Develop machine learning model on MRD kinetics to predict:
  * Outcome 1: Will this MRD+ patient achieve MRD- within 6 months with continued therapy?
  * Outcome 2: If MRD remains positive, what is risk of progression within 12 months?
  * Outcome 3: If escalated, will patient convert to MRD-?
- Use random forest/gradient boosting with early MRD measurements (months 0-6) to predict late outcomes (months 6-24)

**Phase 3: Escalation algorithm design (6 months)**
- For MRD+ patients at month 4-6, construct decision tree:
  ```
  IF MRD ≥ 10^-4 AND slope > threshold:
      → Escalate immediately (add agent X)
  ELIF MRD 10^-5 to 10^-4 AND declining:
      → Continue current therapy, retest month 8
  ELIF MRD declining to <10^-5 AND approaching negativity:
      → Maintain therapy, retest month 10
  ELIF MRD plateau/rising OR >10^-4:
      → Escalate + switch therapy
  ```
- Parameterize thresholds using cohort data

**Phase 4: Prospective validation in real-world setting (24 months)**
- Enroll MM patients in partner health systems with MRD monitoring every 2-3 months
- Randomize to:
  * Arm A: Standard care (clinician decides escalation)
  * Arm B: Algorithm-guided escalation (computer recommends, clinician decides)
- Primary outcome: Proportion MRD-negative at 12 months
- Secondary: PFS, OS, treatment toxicity

**Expected Outcome:**
- Quantitative algorithm for MRD-guided escalation
- Evidence on optimal timing (immediate vs. delayed escalation)
- Prediction model for response trajectory
- RCT evidence (if positive) for clinical adoption

### Potential Insight
**Hypothesis:** MRD kinetics (slope, volatility) more predictive than single value. Early steep decline → watch; flat/rising → escalate immediately. 20-30% improvement in on-time escalation decisions vs. clinician judgment.

---

## Gap 4: Prospective Clinical Validation of AI Progression Predictor (Digital Twin)

### The Question
**Does an externally validated AI model for MM progression prediction, when deployed clinically with real-time feedback to physicians, improve patient outcomes (PFS, OS, quality of life, treatment burden)?**

### Why The Gap Exists

**Critical gap between model development and clinical implementation:**
- Internal validation: AUROC 0.78-0.88 typical
- External validation: AUROC 0.87 on GMMG-MM5 reported
- **Clinical outcome validation: ZERO prospective RCTs of AI progression predictor in MM**

Unknowns:
1. Does prediction accuracy in research setting = clinical decision support value?
2. Do physicians trust and act on AI predictions? (Adoption/adherence)
3. Does actionable AI prediction change treatment decisions? (Behavior change)
4. Do changed treatment decisions improve outcomes? (Clinical benefit)
5. What is clinical trial design for AI intervention? (FDA guidance evolving)

### Which Paper Came Closest

**Closest:** No direct equivalent in MM literature
- AI healthcare papers describe model development/external validation
- Healthcare adoption survey shows <20% health systems have >50% successful prediction model deployment
- **Missing:** Prospective RCT of AI intervention in MM or similar blood cancer

**Related:** MRD-Guided Therapy RCT (NEJM 2024)
- Shows RCT design for outcome-modifying biomarker
- Could use similar framework for AI predictor validation

### Methodology to Close Gap

**Design: Multi-center prospective RCT of AI digital twin deployment (3-year)**

**Phase 1: AI model finalization & validation (12 months, parallel to other gaps)**
- Complete external validation on multiple cohorts (CoMMpass, GMMG-MM5, Connect MM, international registries)
- Achieve consensus on "production-ready" thresholds (minimum AUROC 0.82)
- Prepare regulatory submission (510k or De Novo to FDA if intended as medical device)
- Integrate into clinical software platform (EHR interface, real-time prediction)

**Phase 2: Clinical implementation in pilot sites (6 months)**
- Partner with 5-10 MM treatment centers (mix of academic/community)
- Install digital twin system; train clinicians on interpretation
- Pilot period: 50-100 NDMM patients, refine workflows, resolve technical issues
- Measure: System uptime, prediction turnaround, clinician engagement

**Phase 3: Prospective RCT (24 months)**

**Study design:** Open-label, multi-center, randomized, parallel-group

**Population:** N=600 newly diagnosed MM patients across 10-15 sites

**Inclusion:**
- Age ≥18, newly diagnosed symptomatic MM
- Planned for induction therapy ± transplant
- Baseline labs, imaging, bone marrow cytogenetics available

**Randomization (1:1):**
- **Arm A (n=300):** Standard care + AI predictor (clinician sees prediction every 2 months)
- **Arm B (n=300):** Standard care alone (no AI prediction visible)

**Intervention:**
- AI system generates progression risk score (3-month, 6-month, 12-month horizons) every 2 months
- Score: Low risk (AUROC <0.3), Medium (0.3-0.7), High (>0.7)
- Arm A: Prediction + "suggested action" (e.g., "escalate therapy if MRD+ at month 4")
- Arm B: Blinded to predictions; clinician manages by standard judgment

**Primary Outcome:**
- Progression-free survival at 18 months (composite: progression event, death, or treatment switch)

**Secondary Outcomes:**
- Overall survival
- Time to progression event
- Proportion achieving MRD negativity
- Quality of life (EORTC QLQ-MM24)
- Treatment toxicity grade ≥3 (compare over-treatment in Arm A vs. under-treatment in Arm B)
- Clinician decision-making patterns (treatment changes triggered by prediction vs. clinical judgment)

**Exploratory Outcomes:**
- Biomarker trajectory (MRD, M-spike) vs. predicted trajectory
- Health economic: Cost-effectiveness (QALY gained per AI prediction intervention)
- Adoption metrics: Clinician engagement (% of predictions acted upon), trust scores

**Sample Size:**
- 18-month PFS difference between arms: Assume Arm B = 75%, Arm A = 82% (clinically meaningful 7% absolute benefit from AI)
- Power 80%, α 0.05, need N=280 per arm, 600 total with 7% dropout

**Duration:** 24 months enrollment + 18 months follow-up = 3-year total

**Phase 4: Implementation & regulatory pathway (12 months)**
- If positive, prepare FDA submission (510k or De Novo)
- Develop clinical implementation toolkit for broader adoption
- Train additional sites; scale deployment

**Expected Outcome:**
- **Positive result:** First evidence that MM progression prediction improves outcomes
- **Negative result:** Identifies barriers to implementation; informs next-generation AI design
- **Either way:** Establishes best practices for AI clinical validation in MM and blood cancers

### Potential Insight
**Hypothesis:** AI predictor improves outcomes by enabling earlier escalation (high-risk patients get intensified therapy sooner), reducing delayed interventions. 5-10% PFS improvement expected if algorithm-guided escalation superior to clinician pattern.

---

## Gap 5: Digital Twin Personalized Treatment Simulation for MM

### The Question
**Can a personalized digital twin, trained on patient-specific baseline characteristics and updated with serial measurements (MRD, labs, imaging), simulate treatment response and guide individual treatment selection vs. standard protocols?**

### Why The Gap Exists

Conceptual maturity high (digital twins described in Nature Reviews Cancer 2025), **clinical implementation essentially zero** in MM:
1. **No MM-specific digital twin system in clinical use**
2. **No RCT comparing simulated vs. standard treatment recommendations**
3. **Key unknowns:**
   - What patient data needed for predictive twin? (Baseline only vs. serial)
   - How accurate are response predictions? (Treatment outcome prediction, not just progression risk)
   - Does per-patient simulation improve on population-average models? (Personalization value)
   - Integration with genomics + MRD dynamics + imaging?
   - How to handle treatment heterogeneity (CAR-T vs. conventional)?

### Which Paper Came Closest

**Closest:** Digital Twins in Oncology Reviews (2025)
- Conceptual framework for in silico treatment simulation
- Applications: Radiotherapy planning, drug development, surgical planning
- **Missing:** MM-specific instantiation; clinical validation; treatment selection application

**Also relevant:** Joint AI-driven Event Prediction (2024)
- Incorporates treatment effects ("assessment of effect of different treatment strategies")
- **Missing:** Explicit per-patient simulation; treatment selection optimization

### Methodology to Close Gap

**Design: Machine learning framework for patient-specific treatment simulation (2-3 year development + 3-year validation)**

**Phase 1: Conceptual framework & data requirements (6 months)**

**Question:** For a patient with known baseline (age, ISS, genomics), which features predict response to:
- Induction regimen A vs. B vs. C?
- Continuation vs. discontinuation of maintenance?
- Conventional therapy vs. CAR-T vs. bispecific antibodies?

**Data sources:**
- CoMMpass (treatment effects by subgroup)
- Clinical trials (CASSIOPEIA, ELOQUENT-3, CARTITUDE-4, etc. — treatment-specific cohorts)
- Real-world registries (treatment patterns, outcomes)

**Key variables to capture:**
- Baseline: Age, ISS, cytogenetics, comorbidities
- Treatment: Drug classes, doses, schedule, duration
- Serial: MRD dynamics, lab trends, imaging response
- Outcome: PFS, OS, toxicity, QoL

**Phase 2: Predictive model development (12 months)**

**Approach A: Subgroup analysis models**
- Stratify by treatment received
- Build separate outcome prediction models for each arm:
  * Model A1: IMiD-based induction
  * Model A2: Proteasome inhibitor-based induction
  * Model B1: High-dose melphalan ± transplant
  * Model B2: Non-transplant approach
  * Model C1: CAR-T for RRMM
  * Model C2: Bispecific antibodies for RRMM
- For each, predict: PFS, OS, MRD trajectory, toxicity

**Approach B: Treatment-outcome interaction model**
- Single model predicting outcome given patient features + treatment assignment
- Estimate treatment effect heterogeneity: Does patient X benefit from therapy Y better than therapy Z?
- Use: Causal forests or Bayesian additive regression trees to estimate CATE (conditional average treatment effect)

**Approach C: Mechanistic simulation**
- Hybrid AI + mechanistic model:
  * Mechanistic: Tumor burden (M-spike, free light chains) dynamics based on treatment pharmacodynamics
  * AI: Patient-specific parameters (treatment sensitivity, toxicity risk) estimated from baseline data
  * Simulation: Forward predict trajectory under different treatments

**Phase 3: Validation & personalization evaluation (12 months)**

**Validation:** External test set (25% CoMMpass + independent GMMG-MM5)
- For each patient, predict outcomes under different treatment options (A1, A2, B1, B2)
- Measure: Calibration, discrimination (can model predict which treatment will be best for this patient?)
- Key metric: "Personalization accuracy" — Does model correctly predict that treatment X better for patient Y than treatment Z?

**Comparison:** Population-average vs. personalized
- Population model: All patients with ISS stage 3 → induction X
- Personalized model: Patient Y with ISS-3 + feature Z → induction X; Patient W with ISS-3 without feature Z → induction Y
- Measure: How often does personalization change treatment recommendation vs. population average?

**Phase 4: Prospective clinical validation (24-36 months)**

**Design:** Pragmatic RCT of treatment selection

**Population:** N=300-400 newly diagnosed MM patients

**Arms:**
- **Arm A (n=150):** Treatment selection via clinician + standard guidelines (NCCN)
- **Arm B (n=150):** Treatment selection via clinician + digital twin recommendation

**Procedure:**
- Baseline assessment (labs, imaging, cytogenetics, MRD if available)
- Digital twin generates predicted outcomes for 2-3 standard induction options:
  * Option 1: Bortezomib + lenalidomide + dexamethasone (VRd)
  * Option 2: Carfilzomib + lenalidomide + dexamethasone (KRd)
  * Option 3: Daratumumab + bortezomib + melphalan + prednisone (D-VMP, for transplant ineligible)
- For each option, twin predicts:
  * Probability of achieving MRD- by month 4
  * Predicted PFS at 18 months
  * Expected grade ≥3 toxicity rate
  * Quality of life trajectory
- Arm B clinician sees this comparison; Arm A clinician makes standard decision

**Primary Outcome:**
- Proportion achieving MRD negativity at month 4

**Secondary:**
- PFS, OS
- Toxicity burden
- Treatment adherence
- Clinician and patient satisfaction with decision process

**Data Collection (serial):**
- Months 0, 1, 2, 4, 6, 8, 12: Labs, MRD, imaging, QoL
- Monthly toxicity assessment
- Real-time model updates: Digital twin re-predicts as new data arrives; tracks prediction accuracy

**Expected Outcome:**
- Validated framework for per-patient treatment simulation
- Evidence on personalization value: Does sim-guided care beat population-average protocols?
- Hypothesis: 10-15% improvement in achieving target biomarkers (MRD-) with personalized guidance
- Regulatory: Establish model for FDA approval of AI treatment selection aids

### Potential Insight
**Hypothesis:** Personalization provides 5-10% outcome improvement over population average for 20-30% of patients (those with atypical biology). Remaining 70-80% follow population patterns; personalization adds little.

---

## Summary: Research Gap Priority & Timeline

| Gap # | **Title** | **Priority** | **Complexity** | **Time to Close** | **Key Outcome** | **Funding Need** |
|-------|----------|----------|----------|--------|--------|---------|
| 1 | Data modality integration | CRITICAL | Medium | 18-24 months | Cost-effectiveness ratio (genomics vs. labs) | $500K-1M |
| 2 | Treatment heterogeneity/temporal drift | CRITICAL | Medium-High | 24-36 months | Retraining frequency recommendation | $800K-1.2M |
| 3 | MRD-guided escalation optimization | HIGH | Medium | 18-24 months | Quantitative escalation algorithm | $400K-600K |
| 4 | Prospective clinical validation RCT | CRITICAL | High | 36-48 months | Outcome improvement evidence | $2M-3.5M |
| 5 | Digital twin treatment simulation | HIGH | Very High | 36-48 months | Personalization framework | $1.5M-2.5M |

---

## Recommended Parallel Pursuit Strategy

1. **Start immediately:** Gaps 1-3 (data modality integration, treatment heterogeneity, MRD escalation)
   - Can leverage existing data (CoMMpass, GMMG-MM5, registries)
   - 18-24 month timeline to first results
   - Low cost (~$1.5M total), high clinical impact

2. **Begin in parallel:** Gap 4 (prospective RCT)
   - Longer timeline (36-48 months) but critical for clinical adoption
   - Start regulatory/IRB work while other gaps progress
   - Multi-center coordination now; patient enrollment year 2-3

3. **Start after Gap 1 resolved:** Gap 5 (digital twin)
   - Depends on Gap 1 insights (what data to include)
   - Depends on Gap 3 insights (how to guide treatment)
   - Could start conceptual framework development year 2

