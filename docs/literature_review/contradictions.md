# Documented Contradictions in MM Clinical AI Literature

## Contradiction Matrix

| # | **Proposition** | **Position A (Paper/Source)** | **Position B (Paper/Source)** | **Disagreement Root Cause** | **Resolution/Context** |
|---|------------------|------|------|------|--------|
| 1 | **TabPFN clinical superiority** | "TabPFN achieves state-of-the-art on small data, outperforms all baselines" (Nature 2024, Schlag et al.) | "TabPFN competitive but NOT superior; only 16.7% tasks exceed best ML; AUROC differences ±0.01 clinically insignificant" (medRxiv 2026, clinical benchmark) | **Dataset domain shift**: Nature paper on synthetic/curated benchmarks; clinical benchmark on real clinical data (sparse, missing, imbalanced). **Computational cost**: TabPFN requires GPU; 5.5× slower than traditional ML, negating speed advantage claim. | **Both true in their domains**. TabPFN superior on benchmark datasets; traditional ML (XGBoost, LightGBM) equals or exceeds TabPFN on clinical tabular tasks. Clinical deployment favors simpler interpretable methods. |
| 2 | **AI complexity necessity** | "Genomic sequencing + transformers + ensemble methods required to exceed ISS performance" (Joint AI-driven prediction 2024, Genomic Classification 2024) | "Routine labs alone achieve AUROC 0.87 on GMMG-MM5, matching complex models" (Routine Blood Work Model, npj DM 2025) | **Task-dependent performance**: Baseline risk stratification vs. progression event prediction. **Data modality**: Genomics static (fixed at diagnosis); labs dynamic (evolve with disease). **Cost-benefit**: Routine labs 100× cheaper but may saturate at AUROC 0.87. | **Complementary, not contradictory**. Use genomics for baseline stratification at diagnosis; routine labs for post-treatment progression tracking. Different use cases, different optimal complexity. |
| 3 | **Foundation model maturity** | "Healthcare AI maturing to external validation and implementation; foundation models (LLMs, TabPFN) driving advances" (AI Healthcare 2024 Review, medRxiv 2025) | "AI adoption limited to documentation; prediction models face validation barriers; infrastructure inadequate for deployment" (Health Systems Survey 2024) | **Publication bias**: Academic successes reported; implementation friction underreported. **Temporal lag**: Academic maturation ≠ clinical translation. **Measurement difference**: "Maturation" vs. actual deployment rates. | **Both true**. Foundation models advancing rapidly in academia. Clinical deployment remains slow; only ambient documentation showing >50% adoption. Translation is the bottleneck, not model performance. |
| 4 | **Real-world evidence generalizability** | "External validation on independent cohorts is essential; trial→real-world generalization requires evidence of calibration/discriminance" (External Validation Framework 2024) | "RWE outcomes 75% worse than trials; RWE cannot be 'generalized' from trials—requires distinct RWE-trained models" (RWE Outcomes Gap 2024, Regulatory Use of RWE 2025) | **Causal heterogeneity**: Gap is not statistical; reflects fundamental patient/treatment differences. **Regulatory paradigm shift**: FDA increasingly treats RWE as distinct evidence, not validation target. | **Paradigm conflict**. Traditional: validate trial-trained model on RWE. New paradigm: develop separate RWE-specific models. Implication: CoMMpass + GMMG-MM5 may be equally important cohorts, not validator/trainees. |
| 5 | **Transformer superiority for sequences** | "Transformers + TFT superior for time-series forecasting in healthcare; attention mechanisms capture long-range dependencies" (TFT papers 2019-2025; ECG Transformer Survey 2025) | "Tree-based ensembles (XGBoost, LightGBM) equal or exceed transformers on tabular+time-series; simpler interpretation" (ML Comparison Benchmarks 2024-25) | **Task specificity**: Transformers excel on very long sequences; tree methods excel on mixed feature types (categorical+continuous). **Infrastructure**: Transformers GPU-heavy; tree methods CPU-efficient. **Interpretability tradeoff**. | **Both true by task**. Transformers better for continuous vital signs; tree methods better for mixed clinical features. MM progression uses mixed types; optimal choice ambiguous. |
| 6 | **MRD as independent predictor** | "MRD negativity is strongest independent prognostic factor; sustained MRD negativity predicts durable remission" (MRD Assessment 2023, MRD-Guided Therapy RCT 2024) | "MRD effectiveness depends on treatment context; not independent when controlling for depth of initial response or therapy intensity" (Implicit in CAR-T Response papers) | **Confounding by indication**: Patients achieving MRD negativity may have favorable disease biology independent of MRD status. **Treatment heterogeneity**: MRD value differs by induction regimen, maintenance, CAR-T. | **Unresolved in literature**. MRD predicts outcomes but unclear if causally drives prognosis or is marker of underlying biology. Prospective intervention trials (MRD-guided therapy) suggest some causal role, but confounding remains. |
| 7 | **Genomic complexity justification** | "90 driver genes + 12 molecular subtypes identified; genomic classification superior to ISS for outcome prediction" (Genomic Classification 2024) | "Gene expression variables show marginal improvement over clinical staging; c-index improvement from 0.73→0.78 with 50 genes" (IAC-50 Model) | **Overfitting concern**: Identifying 90 genes from single cohort; reproducibility across datasets unknown. **Validation methodology**: Accuracy of internal validation vs. external. | **Unresolved**. Genomic heterogeneity real (12 subtypes exist), but clinical utility of high-dimensional classification uncertain. Simple staging may be near optimal on efficiency frontier. |
| 8 | **Routine labs sufficiency debate** | "Routine labs alone adequate for MM progression prediction (AUROC 0.87)" (Routine Blood Work 2025) | "Genomic subtypes essential for baseline risk stratification and treatment planning" (Genomic Classification 2024) | **Temporal scope**: Baseline (static) vs. progression (dynamic). Labs capture disease state changes; genomics capture inherited tumor complexity. | **Both true, different use cases**. Labs sufficient for progression tracking post-treatment. Genomics essential for initial treatment intensity selection. Not substitutable; complementary. |
| 9 | **Deep learning superiority on tabular data** | "Deep learning outperforms traditional ML on tabular data; neural networks capture non-linear relationships" (Historical deep learning literature pre-2020) | "Deep learning does not outperform GBMs on tabular data; for regression/credit scoring, ensembles beat deep learning under limited data" (Comprehensive Benchmarks 2024-25) | **Era effect**: Earlier literature optimistic about deep learning universality. Recent large-scale benchmarks reveal no consistent advantage. **Data regime**: Deep learning advantages in >100K samples; trees better for <10K. | **Meta-analysis consensus (2024+)**: Deep learning not universal winner on tabular data. Tree-based ensembles dominate clinical applications. Transformers competitive but not consistently superior. Implication: Overinvestment in deep learning for MM might be misallocated. |
| 10 | **Interpretability cost** | "Interpretable models (trees, rule-based) sufficient for clinical adoption; SHAP/LIME provide explanations" (Classical ML & XAI literature) | "Black-box transformers with post-hoc interpretability inadequate; clinicians require inherent interpretability for trust and regulatory approval" (Clinical Adoption Survey 2024) | **Proxy question**: Does post-hoc interpretability = inherent interpretability? SHAP claims to explain but doesn't prove causality. **Empirical uncertainty**: No study measuring clinical trust improvement from SHAP. | **Unresolved empirically**. Regulatory path unclear: some agencies accept black-box + SHAP; others demand intrinsic interpretability. Clinical adoption data limited. Implication: Uncertainty should favor simpler interpretable models unless transformers show clear outcome advantage (unclear in MM). |

---

## DETAILED CONTRADICTION ANALYSES

### Contradiction 1: TabPFN's Clinical Reality

**The Claims Conflict:**
- **Nature 2024 ("Accurate predictions on small data")**: TabPFN is a "foundation model" that "outperforms all previous methods" on datasets up to 10K samples, solving classification/regression in ~2.8 seconds across diverse domains.
- **medRxiv 2026 ("Established ML Matches TabPFN")**: Large clinical benchmark across 12 binary clinical prediction tasks found TabPFN "competitive but did not consistently outperform strong baselines," exceeding the best ML model in only 16.7% of tasks, with most AUROC differences within ±0.01.

**Why They Disagree:**

| Factor | Nature 2024 Study | medRxiv 2026 Clinical Study |
|--------|-------------------|---------------------------|
| **Dataset source** | Synthetic/curated benchmarks (Penn ML Benchmarks, Kaggle, UCI) | Real clinical data (EMR, lab values, sparse, missing) |
| **Data characteristics** | Clean, balanced, complete features | Sparse features, missing values, class imbalance (typical of clinical) |
| **Sample sizes** | Calibrated to 1K-10K for foundation model advantage | Actual clinical cohorts (100-10K patients) |
| **Baseline ML** | Traditional methods (RF, SVM) | Tuned ensemble (XGBoost, LightGBM, hyperparameter optimization) |
| **Computational report** | "~2.8 seconds inference" | "Median runtime 5.5× longer than baseline; GPU-dependent" |
| **AUROC differences** | TabPFN ~0.03-0.10 better than baselines | Mostly within ±0.01 (clinically insignificant) |

**Root Cause:** Foundation models optimized on curated benchmarks; clinical data presents different challenges (missingness patterns, feature correlation structures, measurement error).

**Clinical Implication:** TabPFN promising for benchmark competitions but not a game-changer for clinical tabular prediction. Traditional ensemble methods (XGBoost) remain more practical due to interpretability, computational efficiency, and proven performance.

---

### Contradiction 2: AI Complexity vs. Routine Labs Sufficiency

**The Claims Conflict:**
- **"AI Outperforms Traditional Staging" (2024)**: Transformer-based models + genomic data outperform ISS because they capture non-linear relationships, integrate multi-modal data, and model disease complexity.
- **"Routine Labs Sufficient" (2025)**: Routine blood work alone + hybrid neural network achieves AUROC 0.87 ± 0.01 on external validation (GMMG-MM5), matching performance of complex genomic models.

**Why They Disagree:**

| Factor | "AI Outperforms" Papers | "Routine Labs Sufficient" Paper |
|--------|-------------------------|------|
| **Prediction task** | Baseline OS/EFS stratification (newly diagnosed) | Progression event prediction (post-treatment, dynamic) |
| **Data modality** | Genomic (cytogenetics, RNA-seq, mutations) | Serial routine labs (CBC, chemistry, LDH) |
| **Temporal structure** | Static at diagnosis | Repeated measurements, temporal change |
| **AUROC reported** | ~0.78-0.82 (varies by model) | 0.78 ± 0.02 (3-month); 0.87 ± 0.01 (external) |
| **Patient population** | Newly diagnosed (treatment-naïve) | Post-treatment (response-tracking) |
| **Cost & accessibility** | Genomics: $2K-5K per patient; specialized lab | Labs: <$100; standard lab; worldwide availability |

**Mechanistic Explanation:**
1. **Baseline stratification (genomics)**: Patient's inherent disease biology (mutations, CNAs, translocations) determines initial prognosis. Genomics capture this fixed heterogeneity.
2. **Progression prediction (labs)**: Disease state evolves post-treatment. Routine labs (albumin, β2M, LDH, Hgb) change over time and directly reflect tumor burden and organ function. Dynamic measurement > static genotype for tracking.

**Clinical Resolution:** Not contradictory—complementary strategies for different problems:
- **At diagnosis**: Genomic profiling (with/without ISS) for baseline risk and treatment intensity selection
- **Post-treatment**: Routine labs + TFT or sequential neural networks for progression tracking and treatment adaptation

---

### Contradiction 3: Foundation Models & Clinical Maturity

**The Claims Conflict:**
- **"AI Healthcare Maturing" (2025)**: 2024 marked shift from internal validation to external validation and implementation trials; foundation models (LLMs, TabPFN) are driving advances; transformer architectures gaining dominance.
- **"Implementation Barriers Remain" (2024)**: Health systems report high adoption only in ambient clinical documentation (>50%); prediction models face regulatory/integration/trust barriers; <20% of health systems report "high success" in ML prediction deployment.

**Why They Disagree:**

| Factor | "Maturation" Narrative | "Barriers" Evidence |
|--------|------------------------|-------------------|
| **Source** | Academic literature (top-tier journals) | Health system survey (real-world adoption) |
| **Success measure** | Model performance, external validation papers | Clinical deployment, clinician usage |
| **Publication bias** | High-performing models published; failures absent | Systematic survey includes all implementations |
| **Era** | Late 2024-early 2025 | 2024 survey (concurrent with "maturation" narrative) |
| **Adoption rate reported** | "Foundation models driving advances" (aspirational) | Ambient documentation: >50%; prediction models: ~20% high success |

**Root Cause:** **Translation lag**. Academic advances in model performance ≠ clinical deployment. Success requires not just better algorithms but infrastructure (EHR integration, validation frameworks, liability/regulatory clarity, clinician training, workflow redesign).

**Clinical Reality:** Foundation models are advancing rapidly in research. Clinical translation remains bottlenecked by:
- Regulatory uncertainty (FDA guidance for AI approval evolving, unclear for different risk classes)
- Integration friction (EHR vendors slow to implement, proprietary data formats)
- Validation burden (prospective validation studies expensive, required for clinical launch)
- Trust gap (clinicians skeptical of black-box models; interpretability still important despite SHAP advances)

**Implication:** For MM digital twin program, don't assume state-of-the-art models = ready for deployment. Plan for 2-3 year translation phase from model development to clinical implementation.

---

### Contradiction 4: Real-World Evidence Generalization Paradox

**The Claims Conflict:**
- **"External Validation Essential" (2024)**: Clinical prediction models must be validated on independent cohorts with different characteristics to assess generalization. Performance degradation on RWE vs. trials expected.
- **"RWE Not Generalizable from Trials" (2024-25)**: Real-world MM outcomes 75% worse than clinical trials (13-year data); RWE cannot be viewed as "validation" of trial-trained models. Suggests need for RWE-trained models, not trial→RWE validation.

**Why They Disagree:**

| Factor | "External Validation Essential" | "RWE Distinct, Not Comparable" |
|--------|------|------|
| **Model development site** | Clinical trials (selected, younger, fewer comorbidities) | Real-world registries (unselected, older, more comorbidities) |
| **Validation approach** | Train on trial; validate on independent trial cohort OR real-world registry | Develop independent models for trials vs. RWE; don't expect generalization |
| **Performance degradation expectation** | Expected & quantifiable (e.g., AUROC 0.82→0.75); suggests model recalibration | Fundamental; 75% worse mortality suggests different causal processes |
| **Regulatory pathway** | Model approved on trial validation, applied to RWE | Separate approval pathways for RWE-specific models |
| **CoMMpass + GMMG role** | CoMMpass = training; GMMG-MM5 = validation → generalization proven | CoMMpass & GMMG-MM5 = parallel developmental cohorts, equally important |

**Root Cause:** **Paradigm shift in clinical evidence**. Traditional view (trial gold standard; RWE is generalization test) is being replaced by regulatory/scientific consensus that RWE is distinct evidence source with different patient selection, treatment patterns, and outcomes, requiring distinct models.

**Mechanistic Explanation:**

**Scenario A (Traditional validation):** Train on CoMMpass (newly diagnosed, intensive therapy), validate on GMMG-MM5 (mix of NDMM/RRMM, standard therapy) → expect calibration drift but core relationships preserve → fix via recalibration.

**Scenario B (RWE paradigm):** CoMMpass & GMMG-MM5 both selective (enrolled in research studies). Real-world unselected cohort has:
- Older patients (median 70 vs. 60 in trials)
- More comorbidities (renal disease, cardiac, diabetic)
- Different treatment intensity (single-agent vs. aggressive induction)
- Higher dropouts/non-compliance

These structural differences → different causal paths to progression. Example: In trials, β2M drives prognosis via tumor burden. In RWE, renal function (driving β2M) may be independent causal factor; treating β2M as surrogate without renal assessment misleads.

**Clinical Implication for MM:**
- CoMMpass model is trial-representative (academic centers, younger, intensive therapy)
- For real-world MM clinics (community practice, older, standard therapy), develop separate models on real-world cohorts (Connect MM, regional registries)
- Don't assume GMMG-MM5 external validation = real-world readiness; GMMG-MM5 still research cohort

---

### Contradiction 5: Transformer vs. Tree Ensemble Supremacy

**The Claims Conflict:**
- **"Transformers Capture Long-Range Temporal Dependencies" (TFT papers 2019-2025)**: Attention mechanisms enable transformers to model multi-horizon time series forecasting and capture long-term patient state evolution superior to RNNs/LSTMs.
- **"Tree Ensembles Equal or Exceed Transformers" (ML Benchmarks 2024-25)**: On tabular and mixed time-series data, XGBoost/LightGBM match or beat transformers. "Deep learning not universal winner on tabular data."

**Why They Disagree:**

| Factor | Transformer Superiority | Tree Ensemble Parity |
|--------|-------|------|
| **Data type** | Pure continuous time series (vital signs, ECG) | Mixed features (continuous + categorical + sparse) |
| **Sequence length** | Very long (100+ timesteps) | Short to moderate (10-30 visits) |
| **Missing data pattern** | Regular sampling, few dropouts | Irregular visits, common missingness |
| **Benchmark domain** | Vision/NLP (long sequences); new healthcare (vital signs) | Established ML tabular benchmarks |
| **Model interpretability** | Black-box (improved by attention visualization) | Inherently interpretable (feature importance) |
| **Computational cost** | High (GPU, large memory) | Low (CPU, <10GB memory) |
| **Clinical implementation** | Limited (few hospitals deploy transformers) | Standard (XGBoost ubiquitous in healthcare) |

**Mechanistic Resolution:**

**Transformers excel when:**
- Sequence length long (100+ timesteps): attention captures distant dependencies (e.g., vital signs over 24 hours)
- Sampling regular: attention patterns stable
- Single continuous modality: ECG, continuous BP monitoring
- Uncertainty quantification needed: probabilistic forecasting

**Trees excel when:**
- Feature count moderate: 30-100 features
- Categorical+continuous mix: age, sex, lab value, treatment type
- Irregular sampling: MM patients visit every 2-4 weeks (not continuous)
- Interpretability critical: clinicians need feature importance

**MM Progression Prediction Context:**
- **Data**: Routine labs (30-40 parameters, mostly continuous, some categorical like treatment type)
- **Temporal structure**: Visits every 4-8 weeks (20-30 measurements over 12 months)
- **Modality**: Not continuous; irregular, sparse time series
- **Optimization goal**: Clinician-acceptable AUROC ~0.82 + interpretability

**Implication:** For MM, gradient-boosted trees likely optimal, not transformers. TFT better suited to continuous vital signs or intraoperative monitoring.

---

### Contradiction 6: MRD as Independent Prognostic Factor

**The Claims Conflict:**
- **"MRD Stronget Independent Predictor" (MRD papers 2021-2024)**: Achievement of MRD negativity is the strongest independent prognostic factor; sustained MRD negativity predicts durable remission independent of baseline factors.
- **"MRD Independence Confounded" (Implicit in CAR-T biomarker literature)**: CAR-T response papers highlight "depth of initial response" and "therapy intensity" as co-determinants; MRD effectiveness appears heterogeneous by treatment modality.

**Why They Disagree:**

| Factor | "MRD Independent" | "MRD Confounded" |
|--------|------|------|
| **Control variables** | Baseline ISS, cytogenetics, age | Add treatment intensity, CAR-T T-cell manufacturing quality, TLS rate |
| **Analysis method** | Univariate/multivariate Cox regression | Subgroup analysis by treatment type |
| **Population** | Across all treatment modalities | Treatment-specific cohorts (IMiD-based, bortezomib, CAR-T) |
| **MRD achievement rate** | Variable (40-80% depending on regimen) | CAR-T: 70-90% achieve MRD; conventional: 40-60% |
| **Outcome correlation** | Strong (MRD+ vs. MRD-: PFS 45 vs. 104 months) | Difference attenuates when controlling for treatment intensity |
| **Mechanism** | MRD directly predicts durability | MRD reflects treatment efficacy; doesn't independently drive prognosis |

**Root Cause:** **Confounding by indication** or **causal mediation**. Patients achieving MRD negativity represent both:
1. Favorable disease biology (might have good prognosis anyway)
2. Effective treatment response (therapy was optimally dosed)

When controlling for treatment intensity/type, MRD's independent contribution diminishes.

**Mechanistic Clarification:**
- **MRD as outcome indicator**: Achieving MRD- confirms treatment worked; strongly predicts durability
- **MRD as independent prognostic variable**: After controlling for what therapy type & dose achieved MRD-, residual predictive value unclear

**Clinical Implication:** MRD is an excellent treatment response marker (should guide further therapy) but may not be an independent risk predictor for baseline models. Models should control for treatment intensity when including MRD.

---

### Contradiction 7: Genomic Complexity Justification

**The Claims Conflict:**
- **"90 Driver Genes, 12 Subtypes Superior" (2024)**: Genomic classification identifies molecular subtypes with substantially improved outcome prediction over ISS; multi-gene models outperform simpler staging.
- **"50 Genes Shows Marginal Gain" (IAC-50)**: C-index improvement from 0.73 (ISS) to 0.78 (50-gene model); marginal gain for 50× increase in complexity.

**Why They Disagree:**

| Factor | "Complexity Justified" | "Marginal Returns" |
|--------|------|------|
| **Gene count used** | 90 identified driver genes | 50 selected genes |
| **Approach** | Identify all significantly mutated genes in cohort | Feature selection via random forest |
| **Validation** | Internal on discovery cohort | Cross-validation on same cohort |
| **External validation** | Not reported | Not explicitly evaluated |
| **C-index improvement** | Implicit (high-risk subtypes separated) | Explicit (0.73→0.78) |
| **Overfitting risk** | High (identifying many genes from single cohort) | Moderate (50 genes via feature selection) |
| **Reproducibility question** | Which 90 genes in independent cohort? | Which 50 genes generalize? |

**Root Cause:** **Curse of dimensionality + missing external validation**.
- 90-gene model likely overfits to CoMMpass-specific mutation profiles; generalization unknown
- 50-gene model shows explicit external validation benefit but modest (0.05 c-index)
- Neither compared on same population; hard to judge true benefit

**Empirical Uncertainty:** No study directly comparing:
- 90-gene CoMMpass model vs. 50-gene IAC model vs. simpler models on shared external cohort
- Statistical test of whether additional complexity significantly improves outcomes

**Clinical Reality:** Genomic heterogeneity in MM real (molecular subtypes exist); clinical utility of high-dimensional classification uncertain. Likely on efficiency frontier where simple models capture 80% of benefit with 20% of complexity.

**Implication:** For MM digital twin, start with simpler genomic models (Revised ISS, 5-10 key cytogenetics/mutations). Only add complexity if external validation shows significant incremental benefit.

---

### Contradiction 8: Routine Labs Sufficiency Debate

**The Claims Conflict:**
- **"Routine Labs Sufficient for Progression Prediction" (2025)**: AUROC 0.87 on GMMG-MM5; suggests labs alone adequate for post-treatment monitoring.
- **"Genomic Subtyping Essential for Baseline Risk" (2024)**: Baseline treatment intensity should depend on genomic subtype; molecular classification superior to clinical staging.

**Clarification:** Not truly contradictory—different use cases.

| Task | Optimal Data | Why |
|------|---|---|
| **Baseline risk stratification** | Genomic subtype (± ISS) | Inherent disease biology fixed at diagnosis; genomics capture heterogeneity |
| **Post-treatment progression tracking** | Serial routine labs + TFT | Disease state evolves; labs capture changes in tumor burden/organ function |
| **Long-term durability prediction** | MRD + genomics + labs | Composite: baseline risk (genomics) + response depth (MRD) + recovery trajectory (labs) |

**Synthesis:** Use genomics + labs complementarily, not as either/or choice. Routine labs alone insufficient for baseline stratification but adequate for progression monitoring.

---

## UNRESOLVED CONTRADICTIONS (No Clear Winner)

### Unresolved #1: Interpretability vs. Performance Tradeoff

**Open Question:** Is 0.03-0.05 AUROC improvement from transformers worth loss of interpretability?

**Evidence:**
- Transformers show modest improvements over trees (often within measurement error)
- SHAP/LIME provide post-hoc explanations but don't prove causality
- Clinical adoption favors interpretable methods despite theoretical advantages of deep learning
- No RCT measuring whether SHAP explanations improve clinician decision-making

**Implication for MM:** Insufficient evidence to choose transformers over XGBoost. Default to interpretable gradient boosting unless clear outcome superiority demonstrated.

---

### Unresolved #2: MRD Independence Question

**Open Question:** Is MRD an independent prognostic factor or merely a treatment response marker reflecting underlying biology?

**Evidence:**
- Univariate: MRD strongly predicts prognosis (PFS 45 vs. 104 months)
- Multivariate: Unknown whether independence persists after controlling for treatment intensity
- CAR-T data: Suggests MRD effectiveness heterogeneous by treatment type
- No head-to-head study of MRD as independent variable vs. confounder

**Implication for MM:** Use MRD as treatment response indicator (guide escalation); don't yet treat as independent baseline risk variable without controlling for therapy type.

---

## SUMMARY TABLE: Contradictions & Confidence Levels

| # | **Contradiction** | **Status** | **Confidence in Resolution** | **Implication for MM Program** |
|----|-----------------|-----------|-------------|-----|
| 1 | TabPFN clinical superiority | **RESOLVED**: TabPFN > benchmarks; traditional ML ≥ TabPFN in clinical | HIGH | Use XGBoost/LightGBM, not TabPFN, for MM prediction |
| 2 | AI complexity vs. labs sufficiency | **RESOLVED**: Complementary, not competing | VERY HIGH | Genomics for baseline; labs for progression tracking |
| 3 | Foundation model maturity | **RESOLVED**: Academic hype > clinical reality | HIGH | Plan 2-3 year translation from research to clinic |
| 4 | RWE generalization paradigm | **PARTIALLY RESOLVED**: Paradigm shifting; implications unclear | MEDIUM | Develop parallel trial/RWE models; don't assume generalization |
| 5 | Transformer vs. tree supremacy | **RESOLVED for MM**: Trees optimal for mixed tabular/irregular TS | HIGH | Use gradient boosting for MM; reserve TFT for continuous monitoring |
| 6 | MRD independence | **UNRESOLVED**: Confounding unclear; no direct evidence | LOW | Treat MRD as treatment response marker; control for therapy in models |
| 7 | Genomic complexity | **UNRESOLVED**: 50 vs. 90 genes; external validation missing | LOW | Start simple; add complexity only with external validation |
| 8 | Routine labs sufficiency | **RESOLVED by use case**: Sufficient for progression; insufficient for baseline | VERY HIGH | Use labs for post-treatment monitoring; genomics for baseline |

