# Hidden Assumptions in MM Clinical AI

## Assumption 1: Longitudinal Blood Work Captures Disease Burden Adequately

**Stated in:** "Predicting progression events in multiple myeloma from routine blood work" (Nature Digital Medicine, 2025); SCOPE model papers treating M-spike, free light chains, and CBC as sufficient proxies for tumor biology.

**The assumption:** Lab parameters (M-spike, free light chains, creatinine, hemoglobin, LDH, corrected calcium) reflect true tumor burden, clonal composition, and bone disease sufficiently to predict major clinical events without genomic or imaging data.

**Consequences if wrong:** Models trained solely on blood work will systematically under-predict progression in patients with high tumor burden but low M-spike (non-secretory myeloma, ~2–3% of cases), or miss bone marrow infiltration-driven complications in patients with early-stage lab derangements. Treatment response predictions may conflate M-spike clearance with durable remission, missing clonal evolution and minimal residual disease persistence. Real-world deployment would then require post-prediction imaging/bone marrow confirmation, doubling healthcare costs and delaying intervention.

---

## Assumption 2: Genomic Subtypes Are Temporally Stable Across Treatment

**Stated in:** IRMMa and CoMMpass subtype literature (12-cluster classification from baseline sequencing).

**The assumption:** The 12 myeloma subtypes (hyperdiploidy, t(11;14), t(4;14), t(14;16), t(14;20), gain(1q), del(1p), del(17p), TP53 mutation, etc.) inferred at diagnosis remain the dominant clonal populations and prognostic drivers throughout treatment, relapse, and progression.

**Consequences if wrong:** Subtype-matched treatment selection at diagnosis may become obsolete by relapse if clonal selection pressure or hypermutation during therapy shifts the dominant subtype or induces new driver mutations. Models treating subtype as a fixed covariate would then misallocate high-risk treatments, potentially overexposing lower-risk emerging clones while under-treating clonal escape variants. This could explain why some patients follow predicted trajectories while others "surprise" clinicians with atypical progression patterns.

---

## Assumption 3: Historical Cohorts Capture Modern Treatment Effects Sufficiently

**Stated in:** TOURMALINE data (2015–2020), CoMMpass data (2015–2018), and survival curves derived from these.

**The assumption:** Survival estimates and treatment response patterns learned from patients receiving older regimens (bortezomib induction, lenalidomide maintenance) generalize to contemporary cohorts receiving carfilzomib, pomalidomide, venetoclax, teclistamab, or other newer agents.

**Consequences if wrong:** A model trained on TOURMALINE data predicting "median OS 8 years for standard-risk" patients may over-pessimistically forecast outcomes for identical patients in 2025 who receive dual targeted therapy (e.g., teclistamab + CFZX). Conversely, older cohorts likely under-represent RRMM patient recovery rates post-CAR-T or bispecific antibodies. External validation on modern cohorts is thus essential and often underperformed; domain shift introduces silent failures.

---

## Assumption 4: Prediction Model Uncertainty Translates to Clinical Decision Uncertainty

**Stated in:** Most uncertainty quantification papers (Bayesian NN, ensemble methods, conformal prediction).

**The assumption:** A model outputting "80% confidence in PFS >12 months" meaningfully communicates the clinician's epistemic uncertainty; calibration implies that in cohorts where the model expresses 80% confidence, ~80% achieve the outcome.

**Consequences if wrong:** Clinical confidence is driven by individual patient narrative, imaging visibility, and side-effect burden, not model confidence scores. A patient with visibly improving bone lesions and stable labs may be refractory to model predictions of progression. Conversely, a patient with stable imaging but rising M-spike may cause alarm despite model-predicted low risk. Physicians are trained to distrust probabilistic outputs that contradict perceived patient status. Deployment without explicit physician retraining or interface design may lead to alarm fatigue or, worse, dismissal of valid early-warning signals embedded in model confidence.

---

## Assumption 5: Accessible Data Equals Causal Drivers

**Stated in:** All CoMMpass-derived models; feature importance ranking (SHAP, tree-split frequency).

**The assumption:** The top predictive features—M-spike trajectory, del(17p), LDH, creatinine—are causal drivers of progression, not proxies for unmeasured confounders (e.g., disease burden in bone marrow not reflected in serum biomarkers, immune microenvironment composition, evolving clonal heterogeneity).

**Consequences if wrong:** Interventions targeting "high M-spike" without addressing underlying clonal dynamics may fail if M-spike is an effect, not a cause, of aggressive biology. A feature appearing "predictive" simply because it correlates with tumor burden may offer no actionable lever. Patient stratification by M-spike kinetics alone could misidentify truly treatment-sensitive vs. -resistant populations, leading to futile escalations or inappropriate de-escalations.
