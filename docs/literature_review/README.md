# Multiple Myeloma Clinical AI & Digital Twin Literature Review

## Complete Literature Review Deliverables

This directory contains a comprehensive structured literature review of 44+ scientific papers on Multiple Myeloma progression prediction, clinical AI, and digital twins. The review is organized into 5 complementary documents designed to support a digital twin research program.

**Prepared:** March 2026 | **Scope:** MM progression prediction, clinical AI, CoMMpass dataset, survival prediction, TabPFN, temporal fusion transformers, digital twins, routine blood work cancer prediction

---

## Document Guide

### 1. **papers_inventory.md** (633 lines, 32 KB)
**Purpose:** Complete catalog of all research papers organized by domain with structured summaries and concept clustering.

**Contents:**
- 10 research domains with 44+ papers
- Each paper: Author, year, core claim, key details
- Concept clustering (7 major thematic clusters)
- Contradictions flagged between papers
- Reference summary table (by type, quality, relevance)

**Key findings:**
- **Benchmark:** npj Digital Medicine 2025 MM progression AUROC 0.78 ± 0.02 (3-month), external validation AUROC 0.87 ± 0.01 (GMMG-MM5)
- **Paper count by domain:**
  - MM progression prediction: 6 papers
  - Digital twins: 4 papers
  - TabPFN: 3 papers
  - Temporal fusion transformers: 5 papers
  - Prognostic markers: 3 papers
  - MRD: 4 papers
  - Routine blood work: 5 papers
  - CoMMpass: 4 papers
  - Clinical AI surveys: 4 papers
  - Interpretability/XAI: 4 papers

**Use when:** Searching for which papers address specific topics; understanding landscape of MM AI research.

---

### 2. **contradictions.md** (303 lines, 29 KB)
**Purpose:** Systematic catalog of points where 2+ papers directly contradict each other with explanations of disagreement sources.

**Contents:**
- 10 documented contradictions with detailed analysis
- Contradiction matrix (table format for quick reference)
- For each contradiction:
  * Position A vs. Position B (with paper citations)
  * Root cause of disagreement
  * Clinical resolution/implications
  * Confidence in resolution
- 2 unresolved contradictions (open questions)

**Key contradictions:**
1. TabPFN clinical superiority (benchmark hype vs. clinical reality)
2. AI complexity vs. routine labs sufficiency (complementary, not competing)
3. Foundation model maturity (academic vs. clinical translation lag)
4. Real-world evidence generalization (paradigm shift in evidence)
5. Transformer supremacy (task-dependent, not universal)

**Use when:** Understanding disagreements in field; evaluating conflicting claims in literature; decision-making on methodology.

---

### 3. **concept_lineage.md** (355 lines, 17 KB)
**Purpose:** Intellectual history and genealogy of the 3 most-cited foundational concepts across MM AI literature.

**Contents:**
- 3 concept family trees with historical evolution
- For each concept:
  * Who introduced it (founding paper)
  * Who challenged it (critical papers)
  * Who refined it (modern developments)
  * Current consensus (2024-2025)
  * Unresolved tensions

**The 3 concepts:**
1. **International Staging System (ISS) & Genomic Classification**
   - Evolution: ISS (2005) → R-ISS (2015) → R2-ISS (2022) → Genomic subtypes (2024)
   - Unresolved: Complexity vs. simplicity tradeoff

2. **Measurable Residual Disease (MRD)**
   - Evolution: MRD detection (2004) → IMWG consensus (2015) → Treatment-guided (2021) → AI dynamics (2024)
   - Unresolved: MRD independence vs. confounding by therapy type

3. **Machine Learning for Multi-Modal Risk Stratification**
   - Evolution: Univariate (1990s) → Gene expression (2007) → ML boom (2015) → Deep learning (2020) → Reality check (2024)
   - Key insight: Tree-based ensembles dominate, not transformers

**Use when:** Understanding historical context of concepts; identifying unresolved tensions; planning research strategy.

---

### 4. **research_gaps.md** (505 lines, 24 KB)
**Purpose:** Identify 5 major unanswered research questions blocking MM digital twin development, with methodologies to close each gap.

**Contents:**
- 5 research gaps, each with:
  * The specific question
  * Why the gap exists
  * Which paper came closest to answering it
  * Detailed methodology to close gap (phases, timeline, expected outcomes)
  * Potential insights/hypotheses

**The 5 gaps:**
1. **Optimal data modality integration** (Genomics vs. labs vs. MRD? Cost-benefit?)
   - Timeline: 18-24 months | Cost: $500K-1M
   - Expected insight: Labs sufficient for progression; genomics for baseline

2. **Treatment heterogeneity in generalization** (How much do models degrade as treatments evolve?)
   - Timeline: 24-36 months | Cost: $800K-1.2M
   - Expected insight: Retraining needed annually as treatment landscape shifts

3. **MRD-guided escalation algorithm** (When/how to intensify based on MRD dynamics?)
   - Timeline: 18-24 months | Cost: $400K-600K
   - Expected insight: MRD kinetics (slope, volatility) more predictive than static values

4. **Prospective clinical validation RCT** (Does AI improve outcomes vs. standard care?)
   - Timeline: 36-48 months | Cost: $2M-3.5M
   - Critical gap: 0 RCTs in MM AI prediction

5. **Digital twin treatment simulation** (Can personalized simulation guide individual treatment selection?)
   - Timeline: 36-48 months | Cost: $1.5M-2.5M
   - Expected insight: Personalization provides 5-10% benefit for 20-30% of patients

**Recommended parallel pursuit strategy:** Start gaps 1-3 immediately (low cost, 18-24 mo), begin gap 4 coordination in parallel, start gap 5 after gap 1 resolved.

**Use when:** Planning research program; identifying next-generation research priorities; understanding program-relevant open questions.

---

### 5. **methodology_comparison.md** (383 lines, 21 KB)
**Purpose:** Comprehensive analysis of research methodologies across all papers; identify strengths, weaknesses, gaps, quality patterns.

**Contents:**
- 5 methodological categories:
  1. Model development & validation (18 papers)
  2. Benchmarking & comparative studies (10 papers)
  3. Mechanistic & biological studies (8 papers)
  4. Clinical trials & prospective studies (5 papers, but 0 AI predictor RCTs)
  5. Health economics & implementation science (3 papers)
- For each category: Definition, subcategories, examples, strengths, weaknesses
- Methodological dominance analysis (frequency, rigor, clinical relevance)
- Underuse analysis (prospective validation, cost-effectiveness, implementation science)
- Quality grading framework
- Recommendations for digital twin program

**Key findings:**
- **Dominant:** Retrospective model development + external validation (41% of papers)
- **Underused:** Prospective RCTs (0% for AI prediction), cost-effectiveness (7%), implementation science (7%)
- **Critical gap:** No prospective RCT validating AI predictor in MM; this is methodological ceiling preventing clinical translation
- **Recommendation:** Plan digital twin with both research track (models) + clinical validation track (prospective cohort → RCT)

**Methodology hierarchy recommended:**
1. Retrospective model + external validation (18-24 mo)
2. Health economic analysis (6-12 mo, concurrent)
3. Prospective real-world cohort (12-24 mo)
4. RCT (36-48 mo)
5. Implementation science (12-36 mo, concurrent with trials)

**Use when:** Understanding methodological quality of literature; planning research phases; evaluating what evidence is sufficient for clinical adoption.

---

## Cross-References & Quick Lookups

### By Research Question:
- **"Should we use genomics or labs?"** → gaps.md Gap #1, contradictions.md #2, papers_inventory.md Domain 1, methodology_comparison.md Category 3
- **"How accurate are current MM prediction models?"** → papers_inventory.md Benchmark section, papers_inventory.md Domain 1
- **"What are the most important concepts?"** → concept_lineage.md (3 major concepts)
- **"Which papers disagree with each other?"** → contradictions.md (10 contradictions with details)
- **"What research should we fund next?"** → research_gaps.md (5 priorities with timelines/costs)
- **"Is the methodology sound?"** → methodology_comparison.md (quality grading, underused methods)

### By Paper Type:
- **Deep learning models:** papers_inventory.md Domains 1, 2, 4; methodology_comparison.md Category 1
- **Prognostic factors:** papers_inventory.md Domains 5, 6, 7; concept_lineage.md Concept 1, 2
- **Registries/real-world:** papers_inventory.md Domain 8; contradictions.md #4; methodology_comparison.md Category 4
- **Benchmarks/comparisons:** methodology_comparison.md Category 2; contradictions.md #1, #5

### By Clinical Application:
- **Baseline risk stratification:** papers_inventory.md Genomic classification, ISS; concept_lineage.md Concept 1
- **Post-treatment progression tracking:** papers_inventory.md Routine blood work, MRD; contradictions.md #2
- **Treatment response prediction:** papers_inventory.md CAR-T, treatment effects; concept_lineage.md Concept 2
- **Personalized medicine:** papers_inventory.md Digital twins; research_gaps.md Gap #5

---

## Statistical Benchmark Summary

**Reference benchmark (npj Digital Medicine 2025):**
- **Task:** 3-month MM progression event prediction from routine blood work
- **Training:** CoMMpass (N=1,186)
- **Internal validation:** AUROC 0.78 ± 0.02 (3-month horizon), declines to 0.65 ± 0.01 at 12-month
- **External validation:** GMMG-MM5 (N=504) → AUROC 0.87 ± 0.01
- **Clinical application:** Virtual human twin for resource optimization and patient outcomes

**Comparison methodologies:**
- ISS (clinical staging): ~0.65-0.70 AUROC (from literature)
- Genomic models: ~0.78-0.82 c-index (from papers_inventory.md)
- Routine labs alone: 0.87 AUROC on external validation (matches reference benchmark)
- Combined multi-modal (theoretical): Unknown; not directly compared

---

## Key Insights for Digital Twin Program

1. **Start simple:** Gradient boosting on routine labs + R2-ISS clinical features will likely achieve 0.85+ AUROC with high interpretability.

2. **Add modalities sequentially:** Validate each addition (genomics, MRD, imaging) before including; demonstrate external validation benefit before adding complexity.

3. **Don't chase hype:** TabPFN, transformers promising in benchmarks but not yet proven superior on clinical MM data. Tree-based ensembles dominate real-world deployment.

4. **Temporal generalization critical:** Treatment landscape evolves; plan for annual model retraining as new therapies adopted (CAR-T, bispecific antibodies).

5. **Prospective validation essential:** Retrospective external validation insufficient for clinical adoption. Plan for prospective cohort study (12-24 mo) and RCT (36-48 mo) to demonstrate outcome benefit.

6. **Complementary data modalities:** Labs sufficient for progression tracking; genomics essential for baseline stratification. Use both together, not as competing approaches.

7. **MRD as treatment response marker:** Very strong evidence; use to guide escalation/de-escalation. May not be independent baseline risk variable; control for treatment intensity in models.

8. **Implementation is bottleneck:** Model performance improving rapidly, but clinical translation slower than academia suggests. Plan implementation science track parallel to model development.

---

## Document Quality & Limitations

**Strengths:**
- Comprehensive coverage (44+ papers across 10 domains)
- Systematic contradiction identification and resolution
- Methodological rigor assessment
- Actionable gaps with research methodologies
- Cross-referenced for easy navigation

**Limitations:**
- **Knowledge cutoff:** March 2026. Newer papers may supersede findings.
- **MM-specific focus:** Some concepts (transformers, digital twins) well-developed in other cancers; MM-specific evidence sparse.
- **Gray literature:** Focuses on peer-reviewed publications. Guidelines, white papers, conference presentations not systematically included.
- **No meta-analysis:** Contradictions analyzed qualitatively, not quantitatively.

**Recommended use:**
- **Primary reference:** Domain-specific searches (papers_inventory.md)
- **Conflict resolution:** When papers disagree (contradictions.md)
- **Research planning:** Identifying gaps and methodologies (research_gaps.md, methodology_comparison.md)
- **Concept understanding:** Historical context and intellectual lineage (concept_lineage.md)

---

## How to Use This Review for Your Program

### Phase 1: Foundation (Weeks 1-4)
1. Read papers_inventory.md sections 1-3 (MM progression prediction, digital twins, CoMMpass)
2. Skim contradictions.md to understand field tensions
3. Review concept_lineage.md to understand key foundational concepts

### Phase 2: Research Planning (Weeks 5-8)
1. Carefully read research_gaps.md to prioritize next-generation research
2. Review methodology_comparison.md to understand what evidence is sufficient for each decision
3. Map research gaps to your program roadmap; identify which gaps to address first

### Phase 3: Model Development (Months 2+)
1. Reference papers_inventory.md Domain 1 for similar model papers; review their methodologies
2. Check contradictions.md #2 (complexity vs. simplicity) and #4 (generalization) to inform design decisions
3. Use methodology_comparison.md to ensure adequate external validation and plan prospective cohort

### Phase 4: Clinical Validation (Months 12+)
1. Deep-read methodology_comparison.md Category 4 on clinical trials
2. Reference research_gaps.md Gap #4 for prospective RCT design
3. Use papers_inventory.md Domain 8 (registries) and Domain 9 (AI surveys) to understand implementation context

---

## Document Metadata

| Document | Size | Lines | Key Sections | Domains Covered |
|----------|------|-------|--------------|----------|
| papers_inventory.md | 32 KB | 633 | 10 domains, 44+ papers, 7 clusters, contradictions, reference table | All |
| contradictions.md | 29 KB | 303 | 10 contradictions, matrix table, detailed analysis, unresolved questions | Cross-domain |
| concept_lineage.md | 17 KB | 355 | 3 concept families, evolution trees, introduction→challenge→refinement | Conceptual |
| research_gaps.md | 24 KB | 505 | 5 research gaps, methodologies, timeline/cost, pursuit strategy | Forward-looking |
| methodology_comparison.md | 21 KB | 383 | 5 method categories, dominance analysis, underuse analysis, grading | Methodological |
| **TOTAL** | **123 KB** | **2179** | - | - |

---

## Questions Answered by This Review

✓ What is the current state of MM clinical AI research?
✓ Which papers address my specific question?
✓ What do papers say about [topic]?
✓ Which papers contradict each other and why?
✓ What are the most important concepts in the field?
✓ What hasn't been studied yet?
✓ How should I design my research?
✓ What methodology should I use?
✓ Is retrospective validation sufficient or do I need prospective evidence?
✓ How does cost-effectiveness inform my decisions?
✓ What are the barriers to clinical implementation?

---

**Last Updated:** March 15, 2026 | **Version:** 1.0 | **Contact:** PhD Researcher 1, Clinical AI Research Program

