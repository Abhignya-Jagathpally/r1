# Field Synthesis: Multiple Myeloma Clinical AI

## What the Field Collectively Believes

The clinical AI community agrees on three core convictions: (1) myeloma patient heterogeneity demands individualized prediction systems that integrate genomics, imaging, clinical parameters, and longitudinal blood work rather than static biomarkers alone; (2) transformer architectures and attention mechanisms can capture temporal dependencies in disease progression more effectively than Cox models or traditional survival methods; (3) large, longitudinally tracked datasets (CoMMpass, TOURMALINE) with rich genomic annotation enable both discovery of molecular subtypes and development of treatment-response predictors.

Consensus also emerges around the utility of multi-modal fusion—combining genomic risk stratification (MMRF 12-subtype classification), real-time lab trajectory prediction, and imaging phenotypes—as a pathway toward precision medicine rather than one-size-fits-all induction protocols.

## What Remains Contested

Two critical debates persist: First, **foundation model promise vs. clinical practicality**. While TabPFN, vision-language models (MUSK), and large language models show benchmark performance on curated datasets, recent evidence suggests established ML baselines (random forests, gradient boosting) remain competitive or superior in actual clinical cohorts, and foundation models incur 5–10× computational cost without consistent performance gains. Second, **blood-work-alone sufficiency**. Recent work predicting progression from routine labs (CBC, chemistry, M-spike) shows promise, yet genomic subtypes and imaging burden (lytic lesions, tumor mass) still contribute independent prognostic signal. The field hasn't resolved whether a model trained on non-invasive blood parameters can match genomic-integrated approaches without significant performance loss.

## What Is Proven

(1) Deep learning survival models (DeepHit, DeepSurv, random survival forests) outperform Cox proportional hazards on myeloma cohorts (C-index: 0.80+ vs. 0.77). (2) Transformer-based joint outcome prediction (SCOPE model) simultaneously forecasts PFS, OS, adverse events, and biomarker trajectories with superior calibration over single-endpoint models. (3) Genomic subtypes (12 CoMMpass clusters) stratify risk more sharply than ISS/RISS alone and are now actionable in clinical practice. (4) Multi-horizon forecasting of vital signs and lab trajectories is feasible in ICU/hospital settings and improves clinician alerting for deterioration. (5) Digital twin simulation frameworks can model counterfactual treatment responses before deployment.

## Single Most Important Unanswered Question

**How do we prospectively validate AI-driven treatment selection (e.g., "patient A benefits most from IRd; patient B from VRd") in a way that informs clinical trial design and regulatory approval?** The field has strong retrospective models but lacks a gold-standard prospective trial demonstrating that AI-guided treatment assignment improves OS or PFS compared to standard-of-care allocation. This gap prevents adoption despite technical readiness. Resolution requires either pragmatic randomized trials in relapsed/refractory cohorts or real-world evidence frameworks that can handle unmeasured confounding and treatment selection bias.
