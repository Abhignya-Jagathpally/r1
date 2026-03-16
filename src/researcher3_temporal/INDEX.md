# Module Index: MM Temporal & Fusion Models

## Quick Navigation

### Getting Started
- **README.md**: Full documentation, installation, quick start examples
- **IMPLEMENTATION_SUMMARY.md**: Detailed implementation notes and architecture
- **example_usage.py**: Runnable examples for all 4 models

### Core Training Infrastructure
- **model_base.py** (408 LOC)
  - `BaseTemporalModel`: Base class with training loop
  - `TrainingConfig`: Configuration dataclass
  - `CheckpointManager`: Model checkpoint management

### Data Loading
- **datasets.py** (335 LOC)
  - `LongitudinalDataset`: Variable-length sequences
  - `SurvivalDataset`: With event times and censoring
  - `MultimodalDataset`: Temporal + clinical + genomic + imaging
  - `pad_sequence_batch()`: Sequence padding helper
  - `create_survival_collate_fn()`: Custom batch collate function

### Model 1: Temporal Fusion Transformer
- **temporal_fusion_transformer.py** (352 LOC)
  - `TFTConfig`: TFT configuration
  - `TemporalFusionTransformer`: Multi-horizon forecasting
  - `VariableSelectionNetwork`: Gated feature selection
  - `TemporalAttentionLayer`: Multi-head attention

**Use for**: Predicting MM progression 3, 6, 12 months ahead

### Model 2: DeepHit (Competing Risks)
- **deephit.py** (328 LOC)
  - `DeepHitConfig`: DeepHit configuration
  - `DeepHit`: Competing risks model
  - `CauseSpecificNetwork`: Per-cause subnetwork

**Use for**: Predicting progression, death, or relapse risk

### Model 3: Dynamic Survival (Landmarks)
- **dynamic_survival.py** (397 LOC)
  - `DynamicSurvivalConfig`: Dynamic model configuration
  - `DynamicSurvivalModel`: Landmark-based conditional survival
  - `LandmarkWindow`: History window extraction
  - `ConditionalRiskHead`: Risk estimation head

**Use for**: P(T > t+Δ | T>t, H(t)) conditional predictions

### Model 4: Multimodal Fusion
- **multimodal_fusion.py** (524 LOC)
  - `MultimodalFusionConfig`: Fusion configuration
  - `MultimodalFusionNet`: Late fusion architecture
  - `TemporalBranch`: LSTM + attention temporal encoder
  - `ClinicalBranch`: Static feature encoder
  - `GenomicBranch`: Gene expression encoder
  - `ImagingBranch`: Radiomics encoder
  - `AttentionFusion`: Modality weighting and fusion

**Use for**: Combining temporal, clinical, genomic, imaging data

### Package Initialization
- **__init__.py** (76 LOC)
  - Public API exports
  - Package metadata

### Configuration & Dependencies
- **requirements.txt**: PyTorch, numpy, pandas, mlflow
- **IMPLEMENTATION_SUMMARY.md**: Detailed technical summary

---

## Class Hierarchy

```
nn.Module
├── BaseTemporalModel
│   ├── TemporalFusionTransformer
│   ├── DeepHit
│   ├── DynamicSurvivalModel
│   └── MultimodalFusionNet
└── [Other layers]
    ├── VariableSelectionNetwork
    ├── TemporalAttentionLayer
    ├── CauseSpecificNetwork
    ├── LandmarkWindow
    ├── ConditionalRiskHead
    ├── TemporalBranch
    ├── ClinicalBranch
    ├── GenomicBranch
    ├── ImagingBranch
    └── AttentionFusion

Dataset
├── LongitudinalDataset
├── SurvivalDataset
└── MultimodalDataset
```

---

## Configuration Dataclasses

```
TrainingConfig (base)
├── TFTConfig
├── DeepHitConfig
├── DynamicSurvivalConfig
└── MultimodalFusionConfig
```

---

## Import Examples

### All Models
```python
from researcher3_temporal import (
    TemporalFusionTransformer, TFTConfig,
    DeepHit, DeepHitConfig,
    DynamicSurvivalModel, DynamicSurvivalConfig,
    MultimodalFusionNet, MultimodalFusionConfig,
)
```

### Datasets Only
```python
from researcher3_temporal import (
    LongitudinalDataset,
    SurvivalDataset,
    MultimodalDataset,
    create_survival_collate_fn,
)
```

### Base Infrastructure
```python
from researcher3_temporal import (
    BaseTemporalModel,
    TrainingConfig,
    CheckpointManager,
)
```

---

## Key Methods by Model

### TemporalFusionTransformer
- `forward(x, times, seq_lengths, static)` → predictions (batch, horizons)
- `fit(train_loader, val_loader)` → training history
- `compute_loss(y_pred, batch)` → loss tensor

### DeepHit
- `forward(x, seq_lengths)` → (batch, causes, time_steps)
- `get_survival_curves(x, seq_lengths)` → survival probs
- `get_risk_scores(x, seq_lengths, time_horizon)` → cumulative risk

### DynamicSurvivalModel
- `forward(x, times, seq_lengths, landmark_times)` → survival probs
- `get_conditional_survival(x, times, seq_lengths, landmarks, horizons)` → dict

### MultimodalFusionNet
- `forward(x, seq_lengths, static, genomic, imaging)` → predictions
- `get_modality_weights()` → importance weights
- `ablate_modality(name)` → disable modality
- `enable_modality(name)` → re-enable modality

---

## File Statistics

| File | Lines | Classes | Functions |
|------|-------|---------|-----------|
| model_base.py | 408 | 2 | 12 |
| datasets.py | 335 | 4 | 8 |
| temporal_fusion_transformer.py | 352 | 3 | 6 |
| deephit.py | 328 | 2 | 7 |
| dynamic_survival.py | 397 | 4 | 8 |
| multimodal_fusion.py | 524 | 6 | 8 |
| **TOTAL** | **2,344** | **21** | **49** |

(Plus 1,331 LOC of documentation and examples)

---

## Type Annotations

All code includes full PEP 484 type hints:
- Function arguments and returns
- Tensor shape documentation in docstrings
- Optional and Union types
- Generic types (List, Dict, Tuple)

---

## Documentation Features

Each module includes:
- ASCII architecture diagrams
- Detailed docstrings
- Parameter documentation
- Usage examples
- Method signatures with types

---

## References

- **TFT**: Lim et al. (2021) - Temporal Fusion Transformers
- **DeepHit**: Lee et al. (2018) - DeepHit Competing Risks
- **Landmarks**: van Houwelingen & Putter (2007) - Dynamic Prediction
- **PyTorch**: torch.nn, torch.optim, torch.cuda.amp

---

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Review README.md for quick start
3. Run example_usage.py to see all models in action
4. Adapt configurations for your specific MM dataset
5. Train models with your data

---

Ready for integration into MM digital twin pipeline!
