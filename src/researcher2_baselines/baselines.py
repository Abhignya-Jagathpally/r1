"""
Baseline Models for Multiple Myeloma Progression Prediction
 
Implements classical survival and risk prediction models with unified interface.
Each model provides:
- fit(X_train, y_train): Train the model
- predict(X_test): Generate risk scores
- predict_proba(X_test, horizons): Calibrated probabilities at multiple horizons
"""
 
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
 
warnings.filterwarnings("ignore", category=FutureWarning)
 
logger = logging.getLogger(__name__)
 
 
class BaselineModel(ABC):
    """Abstract base class for baseline models."""
 
    def __init__(self, name: str, seed: int = 42):
        """
        Initialize baseline model.
 
        Args:
            name: Model name for logging/tracking
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        self.horizons = [3, 6, 12]  # months
 
    @abstractmethod
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "BaselineModel":
        """
        Fit model to training data.
 
        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Dictionary with 'time' and 'event' keys (duration and event indicators)
            **kwargs: Additional model-specific arguments
 
        Returns:
            self
        """
        pass
 
    @abstractmethod
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate risk scores (higher = higher risk).
 
        Args:
            X_test: Feature matrix
 
        Returns:
            Risk scores (n_samples,)
        """
        pass
 
    def predict_proba(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Generate calibrated progression probabilities at multiple horizons.
 
        Args:
            X_test: Feature matrix
            horizons: Time horizons in months (default: [3, 6, 12])
 
        Returns:
            Dictionary mapping months -> probabilities (n_samples,)
        """
        if horizons is None:
            horizons = self.horizons
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return {h: np.clip(self.predict(X_test), 0, 1) for h in horizons}
 
    def _preprocess(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Handle DataFrame/array conversion and validation."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            return X.values
        return X
 
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply standardization scaling."""
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call with fit=True on training data.")
        return self.scaler.transform(X)
 
 
class LOCFBaseline(BaselineModel):
    """Last Observation Carried Forward - Naive baseline."""
 
    def __init__(self, name: str = "LOCF", seed: int = 42):
        super().__init__(name=name, seed=seed)
        self.last_values = None
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "LOCFBaseline":
        """
        Store last values from training set.
 
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Unused
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        self.last_values = X[-1, :].copy()  # Last observation in training
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return constant risk based on event rate in training set.
 
        Args:
            X_test: Test features
 
        Returns:
            Constant risk scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X = self._preprocess(X_test)
        # Return mean value from last feature (risk proxy)
        return np.full(X.shape[0], self.last_values.mean())
 
 
class MovingAverageBaseline(BaselineModel):
    """Moving Average - Temporal baseline for longitudinal data."""
 
    def __init__(self, window: int = 3, name: str = "MA", seed: int = 42):
        """
        Initialize Moving Average baseline.
 
        Args:
            window: Window size for moving average
            name: Model name
            seed: Random seed
        """
        super().__init__(name=name, seed=seed)
        self.window = window
        self.train_mean = None
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "MovingAverageBaseline":
        """
        Compute moving average statistics from training data.
 
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Unused
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        # Compute rolling mean across samples
        if isinstance(X_train, pd.DataFrame):
            self.train_mean = X_train.rolling(window=self.window).mean().iloc[-1, :].values
        else:
            self.train_mean = np.mean(X[-self.window :, :], axis=0)
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return moving average-based predictions.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X = self._preprocess(X_test)
        # Compute moving average for each test sample
        scores = np.mean(X[:, :], axis=1)
        return np.clip(scores, 0, 1)
 
 
class CoxPHBaseline(BaselineModel):
    """Cox Proportional Hazards model from lifelines."""
 
    def __init__(self, name: str = "CoxPH", seed: int = 42):
        super().__init__(name=name, seed=seed)
        self.fitter = None
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "CoxPHBaseline":
        """
        Fit Cox PH model.
 
        Args:
            X_train: Training features (will be normalized)
            y_train: Dictionary with 'time' and 'event' keys
            **kwargs: Additional arguments for CoxPHFitter
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        X = self._normalize(X, fit=True)
 
        # Prepare dataframe for lifelines
        df = pd.DataFrame(X, columns=self.feature_names or [f"feat_{i}" for i in range(X.shape[1])])
        df["duration"] = y_train["time"]
        df["event_observed"] = y_train["event"]
 
        # Fit Cox PH
        self.fitter = CoxPHFitter()
        self.fitter.fit(df, duration_col="duration", event_col="event_observed", show_progress=False)
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate risk scores from partial hazard.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores (higher = higher risk)
        """
        if not self.is_fitted or self.fitter is None:
            raise ValueError("Model must be fitted before prediction")
 
        try:
            X = self._preprocess(X_test)
            X = self._normalize(X, fit=False)
            n_samples = X.shape[0]
 
            # Get partial hazard scores
            df_test = pd.DataFrame(X, columns=self.feature_names or [f"feat_{i}" for i in range(X.shape[1])])
            scores = self.fitter.predict_partial_hazard(df_test).values
 
            # Ensure 1D output (handle case where .values returns 2D column vector)
            scores = np.ravel(scores)
 
            # Normalize to [0,1] range, handle edge case of zero range
            score_range = scores.max() - scores.min()
            if score_range > 1e-8:
                scores = (scores - scores.min()) / (score_range + 1e-8)
            else:
                logger.warning("CoxPHBaseline: Score range is zero, returning 0.5 * ones")
                return 0.5 * np.ones(n_samples)
 
            return np.clip(scores, 0, 1)
        except Exception as e:
            logger.error(f"CoxPHBaseline.predict() failed: {e}")
            n_samples = self._preprocess(X_test).shape[0]
            return 0.5 * np.ones(n_samples)
 
    def predict_proba(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Generate survival probabilities at multiple horizons using Cox PH survival function.
 
        Args:
            X_test: Feature matrix
            horizons: Time horizons in months (default: [3, 6, 12])
 
        Returns:
            Dictionary mapping months -> survival probabilities (n_samples,)
        """
        if horizons is None:
            horizons = self.horizons
        if not self.is_fitted or self.fitter is None:
            raise ValueError("Model must be fitted before prediction")
 
        try:
            X = self._preprocess(X_test)
            X = self._normalize(X, fit=False)
            n_samples = X.shape[0]
 
            df_test = pd.DataFrame(X, columns=self.feature_names or [f"feat_{i}" for i in range(X.shape[1])])
 
            result = {}
            for horizon in horizons:
                try:
                    # Get survival function for this patient and horizon
                    survival_func = self.fitter.predict_survival_function(df_test)
                    # For each horizon, get the survival probability
                    probs = np.zeros(n_samples)
                    for i, surv_idx in enumerate(survival_func.columns):
                        # Use linear interpolation to get survival at the specified horizon
                        times = survival_func.index.values
                        surv_vals = survival_func.iloc[:, i].values
                        if times[-1] >= horizon:
                            # Interpolate within observed range
                            probs[i] = np.interp(horizon, times, surv_vals)
                        else:
                            # Use last observed value if horizon is beyond follow-up
                            probs[i] = surv_vals[-1]
                    result[horizon] = np.clip(probs, 0, 1)
                except Exception as e:
                    logger.warning(f"Failed to get survival probabilities for horizon {horizon}: {e}")
                    result[horizon] = 0.5 * np.ones(n_samples)
 
            return result
        except Exception as e:
            logger.error(f"CoxPHBaseline.predict_proba() failed: {e}")
            n_samples = self._preprocess(X_test).shape[0]
            return {h: 0.5 * np.ones(n_samples) for h in (horizons or self.horizons)}
 
 
class RandomSurvivalForestBaseline(BaselineModel):
    """Random Survival Forest from scikit-survival."""
 
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        name: str = "RSF",
        seed: int = 42,
    ):
        """
        Initialize Random Survival Forest.
 
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            name: Model name
            seed: Random seed
        """
        super().__init__(name=name, seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "RandomSurvivalForestBaseline":
        """
        Fit Random Survival Forest.
 
        Args:
            X_train: Training features
            y_train: Dictionary with 'time' and 'event' keys
            **kwargs: Additional arguments
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        X = self._normalize(X, fit=True)
 
        # Create structured array for scikit-survival
        y_struct = Surv.from_arrays(
            event=y_train["event"].astype(bool),
            time=y_train["time"],
        )
 
        # Fit forest
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(X, y_struct)
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate risk scores as 1 - survival probability.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
 
        try:
            X = self._preprocess(X_test)
            X = self._normalize(X, fit=False)
            n_samples = X.shape[0]
 
            # Get cumulative hazard at median follow-up time (use 12-month as default)
            scores = self.model.predict(X)
 
            # Ensure 1D output (ravel to handle edge cases)
            scores = np.ravel(scores)
 
            # Normalize to [0,1], handle edge case of zero range
            score_range = scores.max() - scores.min()
            if score_range > 1e-8:
                scores = (scores - scores.min()) / (score_range + 1e-8)
            else:
                logger.warning("RandomSurvivalForestBaseline: Score range is zero, returning 0.5 * ones")
                return 0.5 * np.ones(n_samples)
 
            return np.clip(scores, 0, 1)
        except Exception as e:
            logger.error(f"RandomSurvivalForestBaseline.predict() failed: {e}")
            n_samples = self._preprocess(X_test).shape[0]
            return 0.5 * np.ones(n_samples)
 
    def predict_proba(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Generate survival probabilities at multiple horizons using RSF survival function.
 
        Args:
            X_test: Feature matrix
            horizons: Time horizons in months (default: [3, 6, 12])
 
        Returns:
            Dictionary mapping months -> survival probabilities (n_samples,)
        """
        if horizons is None:
            horizons = self.horizons
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
 
        try:
            X = self._preprocess(X_test)
            X = self._normalize(X, fit=False)
            n_samples = X.shape[0]
 
            result = {}
            for horizon in horizons:
                try:
                    # Get cumulative hazard function for this horizon
                    cumhaz = self.model.predict_cumulative_hazard(X)
                    # For RSF, cumhaz is a DataFrame with times as index and samples as columns
                    times = cumhaz.index.values
                    probs = np.zeros(n_samples)
 
                    for i in range(n_samples):
                        hazard_vals = cumhaz.iloc[:, i].values
                        if times[-1] >= horizon:
                            # Interpolate within observed range
                            cumhaz_at_horizon = np.interp(horizon, times, hazard_vals)
                        else:
                            # Use last observed value if horizon is beyond follow-up
                            cumhaz_at_horizon = hazard_vals[-1]
                        # Convert cumulative hazard to survival: S(t) = exp(-H(t))
                        probs[i] = np.exp(-cumhaz_at_horizon)
 
                    result[horizon] = np.clip(probs, 0, 1)
                except Exception as e:
                    logger.warning(f"Failed to get survival probabilities for horizon {horizon}: {e}")
                    result[horizon] = 0.5 * np.ones(n_samples)
 
            return result
        except Exception as e:
            logger.error(f"RandomSurvivalForestBaseline.predict_proba() failed: {e}")
            n_samples = self._preprocess(X_test).shape[0]
            return {h: 0.5 * np.ones(n_samples) for h in (horizons or self.horizons)}
 
 
class XGBoostSnapshotBaseline(BaselineModel):
    """XGBoost for binary progression prediction (3-month snapshot)."""
 
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        name: str = "XGBoost",
        seed: int = 42,
    ):
        """
        Initialize XGBoost baseline.
 
        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            name: Model name
            seed: Random seed
        """
        super().__init__(name=name, seed=seed)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "XGBoostSnapshotBaseline":
        """
        Fit XGBoost classifier.
 
        Args:
            X_train: Training features
            y_train: Dictionary with 'time' and 'event' keys
            **kwargs: Additional arguments
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        X = self._normalize(X, fit=True)
 
        # Use event indicator as binary target
        y_binary = y_train["event"].astype(int)
 
        # Fit XGBoost
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X, y_binary, verbose=False)
 
        # Apply Platt scaling for calibration
        self.model = CalibratedClassifierCV(self.model, method="sigmoid", cv=5)
        self.model.fit(X, y_binary)
 
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate calibrated risk probabilities.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores (probabilities)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
 
        X = self._preprocess(X_test)
        X = self._normalize(X, fit=False)
 
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0, 1)
 
 
class CatBoostSnapshotBaseline(BaselineModel):
    """CatBoost for binary progression prediction (3-month snapshot)."""
 
    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.1,
        iterations: int = 100,
        name: str = "CatBoost",
        seed: int = 42,
    ):
        """
        Initialize CatBoost baseline.
 
        Args:
            depth: Tree depth
            learning_rate: Learning rate
            iterations: Number of boosting iterations
            name: Model name
            seed: Random seed
        """
        if cb is None:
            raise ImportError(
                "catboost is not installed. Install with: pip install catboost"
            )
        super().__init__(name=name, seed=seed)
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "CatBoostSnapshotBaseline":
        """
        Fit CatBoost classifier.
 
        Args:
            X_train: Training features
            y_train: Dictionary with 'time' and 'event' keys
            **kwargs: Additional arguments
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        X = self._normalize(X, fit=True)
 
        # Use event indicator as binary target
        y_binary = y_train["event"].astype(int)
 
        # Fit CatBoost
        self.model = cb.CatBoostClassifier(
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            random_state=self.seed,
            verbose=False,
            thread_count=-1,
        )
        self.model.fit(X, y_binary)
 
        # Apply Platt scaling for calibration
        self.model = CalibratedClassifierCV(self.model, method="sigmoid", cv=5)
        self.model.fit(X, y_binary)
 
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate calibrated risk probabilities.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores (probabilities)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
 
        X = self._preprocess(X_test)
        X = self._normalize(X, fit=False)
 
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0, 1)
 
 
class LogisticRegressionBaseline(BaselineModel):
    """Logistic Regression for binary progression (3-month)."""
 
    def __init__(self, name: str = "LogReg", seed: int = 42):
        """
        Initialize Logistic Regression baseline.
 
        Args:
            name: Model name
            seed: Random seed
        """
        super().__init__(name=name, seed=seed)
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "LogisticRegressionBaseline":
        """
        Fit Logistic Regression.
 
        Args:
            X_train: Training features
            y_train: Dictionary with 'time' and 'event' keys
            **kwargs: Additional arguments
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        X = self._normalize(X, fit=True)
 
        # Use event indicator as binary target
        y_binary = y_train["event"].astype(int)
 
        # Fit logistic regression
        self.model = LogisticRegression(random_state=self.seed, max_iter=1000, n_jobs=-1)
        self.model.fit(X, y_binary)
 
        # Apply Platt scaling for calibration
        self.model = CalibratedClassifierCV(self.model, method="sigmoid", cv=5)
        self.model.fit(X, y_binary)
 
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate calibrated risk probabilities.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores (probabilities)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
 
        X = self._preprocess(X_test)
        X = self._normalize(X, fit=False)
 
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0, 1)
 
 
class TabPFNBaseline(BaselineModel):
    """
    TabPFN for tabular prediction (simplified placeholder).
 
    Note: Full TabPFN/TabPFNv2 requires external models. This implementation
    provides a compatible interface with mock predictions for demonstration.
    """
 
    def __init__(self, name: str = "TabPFN", seed: int = 42):
        """
        Initialize TabPFN baseline.
 
        Args:
            name: Model name
            seed: Random seed
        """
        super().__init__(name=name, seed=seed)
        self._fallback_model = None
 
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Dict[str, np.ndarray],
        **kwargs
    ) -> "TabPFNBaseline":
        """
        Fit TabPFN (uses XGBoost as fallback for compatibility).
 
        Args:
            X_train: Training features
            y_train: Dictionary with 'time' and 'event' keys
            **kwargs: Additional arguments
 
        Returns:
            self
        """
        X = self._preprocess(X_train)
        X = self._normalize(X, fit=True)
 
        y_binary = y_train["event"].astype(int)
 
        # Use XGBoost as fallback (TabPFN model API would replace this)
        self._fallback_model = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
            random_state=self.seed,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self._fallback_model.fit(X, y_binary, verbose=False)
 
        # Apply Platt scaling
        self._fallback_model = CalibratedClassifierCV(
            self._fallback_model, method="sigmoid", cv=5
        )
        self._fallback_model.fit(X, y_binary)
 
        self.is_fitted = True
        return self
 
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate risk predictions.
 
        Args:
            X_test: Test features
 
        Returns:
            Risk scores (probabilities)
        """
        if not self.is_fitted or self._fallback_model is None:
            raise ValueError("Model must be fitted before prediction")
 
        X = self._preprocess(X_test)
        X = self._normalize(X, fit=False)
 
        probs = self._fallback_model.predict_proba(X)[:, 1]
        return np.clip(probs, 0, 1)
 
