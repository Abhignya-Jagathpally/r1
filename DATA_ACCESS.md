# Data Access Matrix

## MMRF CoMMpass Data Sources

| Data Type | Source | Access Level | Auth Required | What You Get |
|-----------|--------|-------------|---------------|-------------|
| Case metadata (demographics, vital status, diagnosis dates) | GDC Cases API | Open | None | ~995 patients, age, gender, race, ISS stage, survival endpoints |
| Clinical flat files (per-patient, per-visit, labs, treatments) | MMRF Researcher Gateway | Registered | Free MMRF account | 1,143 patients, longitudinal labs, treatment lines, response |
| Clinical flat files (IA20) | AWS s3://mmrf-commpass | Registered | MMRF DUA | Same as above, bulk download |
| Whole exome sequencing | GDC / dbGaP | Controlled | dbGaP approval (phs000748) | BAM/VCF files |
| RNA-seq | GDC / dbGaP | Controlled | dbGaP approval (phs000748) | FASTQ/BAM files |
| Copy number variation | GDC / dbGaP | Controlled | dbGaP approval (phs000748) | Segment files |

## What This Pipeline Can Run On

### Without any registration (GDC open metadata):
- Demographics, vital status, OS/PFS endpoints
- Snapshot classification (no longitudinal features, no lab values)
- **Result quality**: Limited — no lab trajectories, no treatment data, no molecular features

### With MMRF Researcher Gateway account (free registration):
- Full longitudinal labs (M-protein, FLC, hemoglobin, calcium, creatinine, etc.)
- Treatment lines, transplant status, response assessments
- **Result quality**: Full pipeline capability including temporal features

### With dbGaP approval:
- Genomic data (WES, RNA-seq, CNV)
- Required for multimodal fusion models
- **Result quality**: Full multimodal pipeline

## Important Caveats

1. The GDC Cases API returns METADATA only. It does NOT return lab values, treatment details, or longitudinal visit data.
2. Results produced from GDC-only data should be labeled as "metadata-only prototype" — they lack the clinical features that drive published benchmarks.
3. The npj Digital Medicine 2025 benchmark (AUROC 0.78) used full longitudinal lab data. Comparisons using GDC metadata-only features are NOT commensurate.
