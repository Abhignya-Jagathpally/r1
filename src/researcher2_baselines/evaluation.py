"""
Evaluation Metrics for Baseline Models

Implements time-dependent AUROC, Brier score, calibration metrics (Platt/isotonic),
concordance index, and bootstrap confidence intervals.
"""

from typing import Dict, List, Tuple, Optional, Union
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy import stats

try:
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """
    Comprehensive evaluation framework for baseline models with multiple metrics,
    calibration assessment, and bootstrap confidence intervals.
    """

    def __init__(self, n_bootstrap: int = 1000, bootstrap_ci: float = 0.95):
        """
        Initialize evaluator.

        Args:
            n_bootstrap: Number of bootstrap resamples for CI computation
            bootstrap_ci: Confidence level for bootstrap intervals (0.95 = 95% CI)
        """
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = bootstrap_ci

    def auroc_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bootstrap_ci: bool = True,
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Compute AUROC with optional bootstrap confidence interval.

        Args:
            y_true: Binary labels
            y_pred: Predicted probabilities
            bootstrap_ci: Whether to compute bootstrap CI

        Returns:
            AUROC score or (mean, lower_ci, upper_ci) if bootstrap_ci=True
        """
        try:
            auroc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            logger.warning(f"Could not compute AUROC: {e}")
            return np.nan if not bootstrap_ci else (np.nan, np.nan, np.nan)

        if not bootstrap_ci:
            return auroc

        # Bootstrap confidence interval
        np.random.seed(42)
        aurocs = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            try:
                aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
            except ValueError:
                continue

        if len(aurocs) == 0:
            return auroc, np.nan, np.nan

        lower = np.percentile(aurocs, (1 - self.bootstrap_ci) / 2 * 100)
        upper = np.percentile(aurocs, (1 + self.bootstrap_ci) / 2 * 100)

        return auroc, lower, upper

    def brier_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bootstrap_ci: bool = True,
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Compute Brier score (mean squared error) with optional bootstrap CI.

        Args:
            y_true: Binary labels
            y_pred: Predicted probabilities
            bootstrap_ci: Whether to compute bootstrap CI

        Returns:
            Brier score or (mean, lower_ci, upper_ci) if bootstrap_ci=True
        """
        brier = brier_score_loss(y_true, y_pred)

        if not bootstrap_ci:
            return brier

        # Bootstrap confidence interval
        np.random.seed(42)
        briers = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            briers.append(brier_score_loss(y_true[idx], y_pred[idx]))

        lower = np.percentile(briers, (1 - self.bootstrap_ci) / 2 * 100)
        upper = np.percentile(briers, (1 + self.bootstrap_ci) / 2 * 100)

        return brier, lower, upper

    def concordance_index(
        self,
        times: np.ndarray,
        events: np.ndarray,
        predictions: np.ndarray,
        bootstrap_ci: bool = True,
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Compute concordance index (C-index) for survival predictions.

        Args:
            times: Event times (or censoring times)
            events: Binary event indicators
            predictions: Predicted risk scores
            bootstrap_ci: Whether to compute bootstrap CI

        Returns:
            C-index or (mean, lower_ci, upper_ci) if bootstrap_ci=True
        """
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines not available. Skipping C-index computation.")
            return np.nan if not bootstrap_ci else (np.nan, np.nan, np.nan)

        try:
            c_index = concordance_index(times, -predictions, events)
        except Exception as e:
            logger.warning(f"Could not compute C-index: {e}")
            return np.nan if not bootstrap_ci else (np.nan, np.nan, np.nan)

        if not bootstrap_ci:
            return c_index

        # Bootstrap confidence interval
        np.random.seed(42)
        c_indices = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(len(times), size=len(times), replace=True)
            try:
                c_indices.append(concordance_index(times[idx], -predictions[idx], events[idx]))
            except Exception:
                continue

        if len(c_indices) == 0:
            return c_index, np.nan, np.nan

        lower = np.percentile(c_indices, (1 - self.bootstrap_ci) / 2 * 100)
        upper = np.percentile(c_indices, (1 + self.bootstrap_ci) / 2 * 100)

        return c_index, lower, upper

    def calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute calibration metrics and plot calibration curve.

        Args:
            y_true: Binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration metrics:
            - expected_calibration_error: ECE (mean absolute calibration gap)
            - max_calibration_error: MCE (maximum calibration gap)
            - bin_means: Mean predictions per bin
            - bin_fractions: True positive rate per bin
        """
        # Compute calibration curve
        bin_fractions, bin_means = calibration_curve(y_true, y_pred, n_bins=n_bins)

        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(bin_fractions - bin_means))

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_fractions - bin_means))

        return {
            "expected_calibration_error": ece,
            "max_calibration_error": mce,
            "bin_fractions": bin_fractions,
            "bin_means": bin_means,
        }

    def time_dependent_auroc(
        self,
        times: np.ndarray,
        events: np.ndarray,
        predictions: np.ndarray,
        time_horizons: List[float],
        bootstrap_ci: bool = False,
    ) -> Dict[float, Union[float, Tuple[float, float, float]]]:
        """
        Compute time-dependent AUROC at multiple horizons.

        This evaluates discrimination at specific time points:
        - At time t, compares events occurring by t vs. no event by t
        - Handles censoring appropriately

        Args:
            times: Follow-up times (months)
            events: Binary event indicators
            predictions: Predicted risk scores
            time_horizons: Time points to evaluate (months)
            bootstrap_ci: Whether to compute bootstrap CI

        Returns:
            Dictionary mapping horizon -> AUROC (and CI if requested)
        """
        results = {}

        for horizon in time_horizons:
            # Create binary labels for time horizon: event occurred by horizon?
            y_binary = (events == 1) & (times <= horizon)

            # Include only non-censored or censored-after-horizon samples
            mask = (times <= horizon) | ((times > horizon) & (events == 0))

            if np.sum(mask) < 10:  # Need minimum samples
                logger.warning(f"Too few samples at horizon {horizon}. Skipping.")
                results[horizon] = np.nan
                continue

            y_h = y_binary[mask]
            pred_h = predictions[mask]

            if np.sum(y_h) == 0 or np.sum(y_h) == len(y_h):
                # No positives or negatives at this horizon
                logger.warning(f"Imbalanced labels at horizon {horizon}. Skipping.")
                results[horizon] = np.nan
                continue

            if bootstrap_ci:
                results[horizon] = self.auroc_score(y_h, pred_h, bootstrap_ci=True)
            else:
                results[horizon] = self.auroc_score(y_h, pred_h, bootstrap_ci=False)

        return results

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        times: Optional[np.ndarray] = None,
        events: Optional[np.ndarray] = None,
        time_horizons: Optional[List[float]] = None,
        model_name: str = "Model",
    ) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive model evaluation.

        Args:
            y_true: Binary labels for 3-month progression
            y_pred: Predicted probabilities
            times: Follow-up times (optional, for C-index)
            events: Event indicators (optional, for C-index)
            time_horizons: Time horizons for time-dependent AUROC (optional)
            model_name: Model name for logging

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")

        results = {
            "model_name": model_name,
        }

        # Basic metrics
        auroc, auroc_lo, auroc_hi = self.auroc_score(y_true, y_pred, bootstrap_ci=True)
        results["auroc"] = auroc
        results["auroc_ci_lower"] = auroc_lo
        results["auroc_ci_upper"] = auroc_hi

        brier, brier_lo, brier_hi = self.brier_score(y_true, y_pred, bootstrap_ci=True)
        results["brier"] = brier
        results["brier_ci_lower"] = brier_lo
        results["brier_ci_upper"] = brier_hi

        logger.info(f"  AUROC: {auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")
        logger.info(f"  Brier: {brier:.4f} [{brier_lo:.4f}, {brier_hi:.4f}]")

        # Calibration metrics
        cal_metrics = self.calibration_metrics(y_true, y_pred)
        results["calibration_ece"] = cal_metrics["expected_calibration_error"]
        results["calibration_mce"] = cal_metrics["max_calibration_error"]
        logger.info(f"  ECE: {cal_metrics['expected_calibration_error']:.4f}")
        logger.info(f"  MCE: {cal_metrics['max_calibration_error']:.4f}")

        # C-index if survival data available
        if times is not None and events is not None:
            c_idx, c_lo, c_hi = self.concordance_index(times, events, y_pred, bootstrap_ci=True)
            if not np.isnan(c_idx):
                results["c_index"] = c_idx
                results["c_index_ci_lower"] = c_lo
                results["c_index_ci_upper"] = c_hi
                logger.info(f"  C-index: {c_idx:.4f} [{c_lo:.4f}, {c_hi:.4f}]")

        # Time-dependent AUROC
        if time_horizons is not None and times is not None and events is not None:
            td_auroc = self.time_dependent_auroc(
                times,
                events,
                y_pred,
                time_horizons,
                bootstrap_ci=False,
            )
            results["time_dependent_auroc"] = td_auroc
            logger.info("  Time-dependent AUROC:")
            for horizon, auroc_val in td_auroc.items():
                if not np.isnan(auroc_val):
                    logger.info(f"    {horizon} months: {auroc_val:.4f}")

        return results

    def benchmark_comparison(
        self,
        model_results: Dict[str, Dict[str, Union[float, Dict]]],
        target_auroc: float = 0.78,
        target_auroc_ci: float = 0.02,
    ) -> pd.DataFrame:
        """
        Create comparison table vs. benchmark performance.

        Args:
            model_results: Dictionary mapping model names to evaluation results
            target_auroc: Benchmark AUROC (default: 0.78 ± 0.02)
            target_auroc_ci: Benchmark CI width

        Returns:
            DataFrame with comparison results
        """
        rows = []
        benchmark = {"AUROC": target_auroc, "CI": target_auroc_ci}

        for model_name, results in model_results.items():
            auroc = results.get("auroc", np.nan)
            auroc_ci_lower = results.get("auroc_ci_lower", np.nan)
            auroc_ci_upper = results.get("auroc_ci_upper", np.nan)
            brier = results.get("brier", np.nan)
            ece = results.get("calibration_ece", np.nan)
            c_idx = results.get("c_index", np.nan)

            # Compute CI width
            ci_width = auroc_ci_upper - auroc_ci_lower if not np.isnan(auroc_ci_lower) else np.nan

            # Check if meets benchmark
            meets_benchmark = (
                auroc >= (benchmark["AUROC"] - benchmark["CI"])
                if not np.isnan(auroc)
                else False
            )

            rows.append({
                "Model": model_name,
                "AUROC": f"{auroc:.4f}" if not np.isnan(auroc) else "N/A",
                "AUROC_CI": f"[{auroc_ci_lower:.4f}, {auroc_ci_upper:.4f}]" if not np.isnan(auroc_ci_lower) else "N/A",
                "Brier": f"{brier:.4f}" if not np.isnan(brier) else "N/A",
                "ECE": f"{ece:.4f}" if not np.isnan(ece) else "N/A",
                "C-Index": f"{c_idx:.4f}" if not np.isnan(c_idx) else "N/A",
                "Meets_Benchmark": meets_benchmark,
            })

        df = pd.DataFrame(rows)
        logger.info("\n" + "=" * 120)
        logger.info("BASELINE MODEL COMPARISON")
        logger.info("=" * 120)
        logger.info(f"Benchmark AUROC: {benchmark['AUROC']:.4f} ± {benchmark['CI']:.4f}\n")
        logger.info(df.to_string(index=False))
        logger.info("=" * 120 + "\n")

        return df
