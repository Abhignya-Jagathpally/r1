"""
Probabilistic calibration methods for survival/time-to-event models.

Implements:
  - PlattScaler: Platt scaling for binary classification
  - IsotonicCalibrator: Isotonic regression calibration
  - TemperatureScaler: Temperature scaling for neural networks
  - CalibrationAnalyzer: Calibration curves, ECE, Hosmer-Lemeshow test

Note: All fitting is done on training set only to avoid overfitting.
"""

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit, logit
from scipy.stats import chi2
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for calibration evaluation metrics."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    hl_statistic: float  # Hosmer-Lemeshow test statistic
    hl_pvalue: float  # Hosmer-Lemeshow p-value
    hl_calibrated: bool  # Whether H-L test suggests calibration


class PlattScaler:
    """
    Platt scaling: fits a logistic regression to model predictions.

    P(y=1|f) = 1 / (1 + exp(A*f + B))

    Fitted on training data only.
    """

    def __init__(self, max_iter: int = 100, regularization: float = 1e-12):
        """
        Args:
            max_iter: Maximum iterations for optimization
            regularization: L2 regularization parameter
        """
        self.max_iter = max_iter
        self.regularization = regularization
        self.A = None
        self.B = None
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "PlattScaler":
        """
        Fit Platt scaling parameters.

        Args:
            y_true: True binary labels (0/1)
            y_pred: Uncalibrated predictions (probabilities or raw scores)

        Returns:
            self
        """
        # Ensure predictions are in (0, 1)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Convert to log-odds form
        f = logit(y_pred)

        # Simple closed-form solution (gradient descent optimization)
        # Using numpy's polyfit for linear regression on logit scale
        with np.errstate(divide="ignore", invalid="ignore"):
            coefficients = np.polyfit(f, y_true, 1)

        self.A = coefficients[0]
        self.B = coefficients[1]
        self.fitted = True

        logger.debug(f"PlattScaler fitted: A={self.A:.6f}, B={self.B:.6f}")

        return self

    def predict_proba(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to predictions.

        Args:
            y_pred: Uncalibrated predictions

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("PlattScaler must be fitted before prediction")

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        f = logit(y_pred)

        calibrated = expit(self.A * f + self.B)
        return np.clip(calibrated, 1e-15, 1 - 1e-15)


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    Non-parametric, monotonic calibration.

    Fitted on training data only.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Args:
            out_of_bounds: How to handle predictions outside [0,1]
                'clip' or 'isotonic'
        """
        self.out_of_bounds = out_of_bounds
        self.isotonic_fn = None
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "IsotonicCalibrator":
        """
        Fit isotonic regression.

        Args:
            y_true: True binary labels (0/1)
            y_pred: Uncalibrated predictions

        Returns:
            self
        """
        self.isotonic_fn = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            increasing=True
        )
        self.isotonic_fn.fit(y_pred, y_true)
        self.fitted = True

        logger.debug(f"IsotonicCalibrator fitted with {len(y_pred)} samples")

        return self

    def predict_proba(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            y_pred: Uncalibrated predictions

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("IsotonicCalibrator must be fitted before prediction")

        return np.clip(self.isotonic_fn.predict(y_pred), 1e-15, 1 - 1e-15)


class TemperatureScaler:
    """
    Temperature scaling for neural networks.
    Scales logits by a learned temperature parameter T.

    P(y=1|z) = softmax(z / T) = 1 / (1 + exp(-z/T))

    Fitted on training data only.
    """

    def __init__(self, max_iter: int = 100, learning_rate: float = 0.01):
        """
        Args:
            max_iter: Maximum iterations for optimization
            learning_rate: Learning rate for gradient descent
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.temperature = None
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_logits: np.ndarray) -> "TemperatureScaler":
        """
        Fit temperature parameter via maximum likelihood.

        Args:
            y_true: True binary labels (0/1)
            y_logits: Raw logits from model

        Returns:
            self
        """
        # Convert probabilities to logits if needed
        if np.all((y_logits >= 0) & (y_logits <= 1)):
            y_logits = logit(np.clip(y_logits, 1e-15, 1 - 1e-15))

        # Initialize temperature
        temperature = 1.0

        # Gradient descent to optimize NLL
        for iteration in range(self.max_iter):
            # Compute predictions
            probs = expit(y_logits / temperature)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)

            # Compute loss (negative log-likelihood)
            nll = -np.mean(
                y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)
            )

            # Compute gradient
            grad = np.mean(
                (probs - y_true) * y_logits / (temperature ** 2)
            )

            # Update temperature
            temperature -= self.learning_rate * grad

            # Ensure temperature stays positive
            temperature = max(temperature, 0.1)

            if iteration % 20 == 0:
                logger.debug(f"Temperature scaling iteration {iteration}: T={temperature:.4f}, NLL={nll:.4f}")

        self.temperature = temperature
        self.fitted = True

        logger.info(f"TemperatureScaler fitted: T={self.temperature:.4f}")

        return self

    def predict_proba(self, y_logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            y_logits: Raw logits from model

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("TemperatureScaler must be fitted before prediction")

        # Convert probabilities to logits if needed
        if np.all((y_logits >= 0) & (y_logits <= 1)):
            y_logits = logit(np.clip(y_logits, 1e-15, 1 - 1e-15))

        calibrated = expit(y_logits / self.temperature)
        return np.clip(calibrated, 1e-15, 1 - 1e-15)


class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis.
    Computes ECE, MCE, Hosmer-Lemeshow test, and calibration curves.
    """

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for ECE computation

        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_pred[mask].mean()
                bin_size = np.sum(mask)

                ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

        return ece

    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for MCE computation

        Returns:
            MCE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        mce = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_pred[mask].mean()

                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce

    @staticmethod
    def hosmer_lemeshow_test(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_groups: int = 10,
    ) -> Tuple[float, float]:
        """
        Hosmer-Lemeshow goodness-of-fit test.

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_groups: Number of groups for H-L test

        Returns:
            (test_statistic, p_value)
        """
        # Sort by predicted probability
        sorted_indices = np.argsort(y_pred)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        # Divide into deciles
        group_size = len(y_true) // n_groups
        hl_statistic = 0.0

        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < n_groups - 1 else len(y_true)

            n_k = end_idx - start_idx
            if n_k == 0:
                continue

            o_k = np.sum(y_true_sorted[start_idx:end_idx])  # Observed events
            p_bar = np.mean(y_pred_sorted[start_idx:end_idx])  # Mean predicted prob
            e_k = n_k * p_bar  # Expected events

            if e_k > 0 and (n_k - e_k) > 0:
                hl_statistic += ((o_k - e_k) ** 2) / (e_k * (1 - e_k / n_k))

        # Chi-square distribution with n_groups - 2 degrees of freedom
        df = n_groups - 2
        p_value = 1 - chi2.cdf(hl_statistic, df)

        return hl_statistic, p_value

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute calibration curve (mean predicted vs observed frequency).

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins

        Returns:
            (bin_edges, mean_predicted, mean_observed)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        mean_predicted = []
        mean_observed = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                mean_predicted.append(y_pred[mask].mean())
                mean_observed.append(y_true[mask].mean())
            else:
                mean_predicted.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                mean_observed.append(np.nan)

        return bin_boundaries, np.array(mean_predicted), np.array(mean_observed)

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Compute all calibration metrics.

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for ECE/MCE/H-L

        Returns:
            CalibrationMetrics object
        """
        ece = CalibrationAnalyzer.expected_calibration_error(y_true, y_pred, n_bins)
        mce = CalibrationAnalyzer.maximum_calibration_error(y_true, y_pred, n_bins)
        hl_stat, hl_pval = CalibrationAnalyzer.hosmer_lemeshow_test(y_true, y_pred, n_bins)

        # H-L test: p > 0.05 suggests well-calibrated
        hl_calibrated = hl_pval > 0.05

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            hl_statistic=hl_stat,
            hl_pvalue=hl_pval,
            hl_calibrated=hl_calibrated,
        )
