"""
Multiple Myeloma Digital Twin - Evaluation Framework & MLOps Infrastructure
 
PhD Researcher 5: MLOps and Evaluation Expert
 
Modules:
  - splits: Patient-level stratified temporal cross-validation with leakage detection
  - calibration: Probabilistic calibration methods (Platt, isotonic, temperature)
  - metrics: Time-dependent survival metrics (Uno's AUROC, IBS, concordance)
  - mlflow_tracking: Experiment tracking, model registry, artifact management
  - autoresearch: Automated hyperparameter search (random, Bayesian, population-based)
  - reproducibility: Docker, DVC, seed management, environment snapshots
  - reporting: Automated markdown reports, dashboards, statistical tests
"""
 
__version__ = "0.1.0"
__author__ = "PhD Researcher 5 - MLOps & Evaluation"
 
from .splits import (
    PatientLevelSplit,
    TemporalCrossValidator,
    StratifiedGroupKFold,
    LeakageDetector,
)
from .calibration import (
    PlattScaler,
    IsotonicCalibrator,
    TemperatureScaler,
    CalibrationAnalyzer,
)
from .metrics import (
    SurvivalMetrics,
    TimeDependent,
)
try:
    from .mlflow_tracking import (
        ExperimentTracker,
        ModelRegistry,
    )
except ImportError:
    ExperimentTracker = None
    ModelRegistry = None
from .autoresearch import (
    AutoresearchHarness,
    ConfigurationSpace,
)
from .reproducibility import (
    EnvironmentSnapshot,
    DockerfileGenerator,
    DVCPipelineBuilder,
)
from .reporting import (
    ExperimentReporter,
    DeLongTest,
)
 
__all__ = [
    "PatientLevelSplit",
    "TemporalCrossValidator",
    "StratifiedGroupKFold",
    "LeakageDetector",
    "PlattScaler",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "CalibrationAnalyzer",
    "SurvivalMetrics",
    "TimeDependent",
    "ExperimentTracker",
    "ModelRegistry",
    "AutoresearchHarness",
    "ConfigurationSpace",
    "EnvironmentSnapshot",
    "DockerfileGenerator",
    "DVCPipelineBuilder",
    "ExperimentReporter",
    "DeLongTest",
]
 
