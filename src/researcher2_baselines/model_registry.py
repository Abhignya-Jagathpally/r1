"""
Model Registry and Factory for Baseline Models

Provides centralized configuration, hyperparameter defaults, and search spaces
for Bayesian optimization / hyperparameter tuning.
"""

from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
import warnings

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


@dataclass
class HyperparameterSpace:
    """Defines hyperparameter search space for tuning."""

    param_name: str
    param_type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    categories: Optional[List[Any]] = None

    def __post_init__(self):
        """Validate hyperparameter configuration."""
        if self.param_type == "int" or self.param_type == "float":
            if self.low is None or self.high is None:
                raise ValueError(f"Numeric params must have 'low' and 'high': {self.param_name}")
        elif self.param_type == "categorical":
            if self.categories is None or len(self.categories) == 0:
                raise ValueError(f"Categorical params must have 'categories': {self.param_name}")


@dataclass
class ModelConfig:
    """Configuration for a baseline model."""

    name: str
    model_class: Callable
    default_params: Dict[str, Any] = field(default_factory=dict)
    search_space: List[HyperparameterSpace] = field(default_factory=list)
    description: str = ""


class ModelRegistry:
    """
    Factory pattern for baseline models with hyperparameter defaults
    and search spaces for tuning.
    """

    def __init__(self):
        """Initialize model registry with all baseline configurations."""
        self._registry: Dict[str, ModelConfig] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all baseline models with default configurations."""

        # LOCF Baseline
        self._registry["LOCF"] = ModelConfig(
            name="LOCF",
            model_class=LOCFBaseline,
            default_params={"name": "LOCF", "seed": 42},
            search_space=[],
            description="Last Observation Carried Forward - naive temporal baseline",
        )

        # Moving Average Baseline
        self._registry["MovingAverage"] = ModelConfig(
            name="MovingAverage",
            model_class=MovingAverageBaseline,
            default_params={"window": 3, "name": "MA", "seed": 42},
            search_space=[
                HyperparameterSpace(
                    param_name="window",
                    param_type="int",
                    low=1,
                    high=10,
                    log_scale=False,
                )
            ],
            description="Moving Average baseline with configurable window",
        )

        # Cox PH
        self._registry["CoxPH"] = ModelConfig(
            name="CoxPH",
            model_class=CoxPHBaseline,
            default_params={"name": "CoxPH", "seed": 42},
            search_space=[],
            description="Cox Proportional Hazards model from lifelines",
        )

        # Random Survival Forest
        self._registry["RandomSurvivalForest"] = ModelConfig(
            name="RandomSurvivalForest",
            model_class=RandomSurvivalForestBaseline,
            default_params={
                "n_estimators": 100,
                "max_depth": 10,
                "name": "RSF",
                "seed": 42,
            },
            search_space=[
                HyperparameterSpace(
                    param_name="n_estimators",
                    param_type="int",
                    low=50,
                    high=300,
                    log_scale=False,
                ),
                HyperparameterSpace(
                    param_name="max_depth",
                    param_type="int",
                    low=5,
                    high=20,
                    log_scale=False,
                ),
            ],
            description="Random Survival Forest from scikit-survival",
        )

        # XGBoost Snapshot
        self._registry["XGBoost"] = ModelConfig(
            name="XGBoost",
            model_class=XGBoostSnapshotBaseline,
            default_params={
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "name": "XGBoost",
                "seed": 42,
            },
            search_space=[
                HyperparameterSpace(
                    param_name="max_depth",
                    param_type="int",
                    low=3,
                    high=12,
                    log_scale=False,
                ),
                HyperparameterSpace(
                    param_name="learning_rate",
                    param_type="float",
                    low=0.01,
                    high=0.3,
                    log_scale=True,
                ),
                HyperparameterSpace(
                    param_name="n_estimators",
                    param_type="int",
                    low=50,
                    high=300,
                    log_scale=False,
                ),
            ],
            description="XGBoost for binary progression prediction",
        )

        # CatBoost Snapshot
        self._registry["CatBoost"] = ModelConfig(
            name="CatBoost",
            model_class=CatBoostSnapshotBaseline,
            default_params={
                "depth": 6,
                "learning_rate": 0.1,
                "iterations": 100,
                "name": "CatBoost",
                "seed": 42,
            },
            search_space=[
                HyperparameterSpace(
                    param_name="depth",
                    param_type="int",
                    low=3,
                    high=12,
                    log_scale=False,
                ),
                HyperparameterSpace(
                    param_name="learning_rate",
                    param_type="float",
                    low=0.01,
                    high=0.3,
                    log_scale=True,
                ),
                HyperparameterSpace(
                    param_name="iterations",
                    param_type="int",
                    low=50,
                    high=300,
                    log_scale=False,
                ),
            ],
            description="CatBoost for binary progression prediction",
        )

        # Logistic Regression
        self._registry["LogisticRegression"] = ModelConfig(
            name="LogisticRegression",
            model_class=LogisticRegressionBaseline,
            default_params={"name": "LogReg", "seed": 42},
            search_space=[],
            description="Logistic Regression for 3-month progression",
        )

        # TabPFN
        self._registry["TabPFN"] = ModelConfig(
            name="TabPFN",
            model_class=TabPFNBaseline,
            default_params={"name": "TabPFN", "seed": 42},
            search_space=[],
            description="TabPFN for tabular visit-level risk prediction",
        )

    def get_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a registered model.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig with defaults and search space

        Raises:
            ValueError: If model not found in registry
        """
        if model_name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"Model '{model_name}' not registered. Available: {available}")
        return self._registry[model_name]

    def create(
        self,
        model_name: str,
        **kwargs
    ) -> BaselineModel:
        """
        Factory method to instantiate a baseline model.

        Args:
            model_name: Name of the model to create
            **kwargs: Override default hyperparameters

        Returns:
            Instantiated BaselineModel

        Raises:
            ValueError: If model not found in registry
        """
        config = self.get_config(model_name)
        params = config.default_params.copy()
        params.update(kwargs)

        # Remove 'seed' from kwargs if present to avoid duplication
        seed = params.pop("seed", 42)

        model = config.model_class(**params, seed=seed)
        return model

    def list_models(self) -> Dict[str, str]:
        """
        List all registered models with descriptions.

        Returns:
            Dictionary mapping model names to descriptions
        """
        return {name: config.description for name, config in self._registry.items()}

    def get_search_space(self, model_name: str) -> List[HyperparameterSpace]:
        """
        Get hyperparameter search space for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of HyperparameterSpace objects

        Raises:
            ValueError: If model not found in registry
        """
        config = self.get_config(model_name)
        return config.search_space

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of default parameters

        Raises:
            ValueError: If model not found in registry
        """
        config = self.get_config(model_name)
        return config.default_params.copy()

    def register(
        self,
        model_name: str,
        model_class: Callable,
        default_params: Dict[str, Any],
        search_space: Optional[List[HyperparameterSpace]] = None,
        description: str = "",
    ) -> None:
        """
        Register a custom model.

        Args:
            model_name: Unique name for the model
            model_class: Model class (must inherit from BaselineModel)
            default_params: Dictionary of default hyperparameters
            search_space: List of HyperparameterSpace objects for tuning
            description: Model description

        Raises:
            ValueError: If model name already registered
        """
        if model_name in self._registry:
            warnings.warn(f"Model '{model_name}' already registered. Overwriting.", stacklevel=2)

        self._registry[model_name] = ModelConfig(
            name=model_name,
            model_class=model_class,
            default_params=default_params,
            search_space=search_space or [],
            description=description,
        )
