"""
Multiple Myeloma Baseline Models - Production Package
PhD Researcher 3: Classical baseline modeling for digital twin pipeline
"""

__version__ = "1.0.0"
__author__ = "PhD Researcher 3"

from .baselines import (
    BaselineModel,
    LOCFBaseline,
    MovingAverageBaseline,
    CoxPHBaseline,
    RandomSurvivalForestBaseline,
    XGBoostSnapshotBaseline,
    CatBoostSnapshotBaseline,
    LogisticRegressionBaseline,
    TabPFNBaseline,
)
from .model_registry import ModelRegistry
from .training import BaselineTrainer
from .evaluation import BaselineEvaluator

__all__ = [
    "BaselineModel",
    "LOCFBaseline",
    "MovingAverageBaseline",
    "CoxPHBaseline",
    "RandomSurvivalForestBaseline",
    "XGBoostSnapshotBaseline",
    "CatBoostSnapshotBaseline",
    "LogisticRegressionBaseline",
    "TabPFNBaseline",
    "ModelRegistry",
    "BaselineTrainer",
    "BaselineEvaluator",
]
