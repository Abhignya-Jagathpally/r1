"""
Multiple Myeloma Temporal & Fusion Models
==========================================

Advanced PyTorch implementations for:
  - Temporal Fusion Transformer (TFT) for MM progression forecasting
  - DeepHit for competing risks (progression, death, relapse)
  - Dynamic survival models with landmarking
  - Multimodal fusion (clinical + temporal + genomic)

All models support:
  - Variable-length sequences with irregular time intervals
  - Mixed precision training
  - Early stopping and LR scheduling
  - MLflow experiment tracking
  - Interpretability via attention weights

Author: PhD Researcher 4
Version: 1.0.0
"""

from .model_base import (
    BaseTemporalModel,
    TrainingConfig,
    CheckpointManager,
)

from .temporal_fusion_transformer import (
    TemporalFusionTransformer,
    TFTConfig,
)

from .deephit import (
    DeepHit,
    DeepHitConfig,
)

from .dynamic_survival import (
    DynamicSurvivalModel,
    DynamicSurvivalConfig,
)

from .multimodal_fusion import (
    MultimodalFusionNet,
    MultimodalFusionConfig,
)

from .datasets import (
    LongitudinalDataset,
    SurvivalDataset,
    MultimodalDataset,
    pad_sequence_batch,
    create_survival_collate_fn,
)

__all__ = [
    "BaseTemporalModel",
    "TrainingConfig",
    "CheckpointManager",
    "TemporalFusionTransformer",
    "TFTConfig",
    "DeepHit",
    "DeepHitConfig",
    "DynamicSurvivalModel",
    "DynamicSurvivalConfig",
    "MultimodalFusionNet",
    "MultimodalFusionConfig",
    "LongitudinalDataset",
    "SurvivalDataset",
    "MultimodalDataset",
    "pad_sequence_batch",
    "create_survival_collate_fn",
]

__version__ = "1.0.0"
__author__ = "PhD Researcher 4"
