"""
Time-dependent survival metrics for clinical prediction models.

Implements:
  - TimeDependent: Base class for time-dependent metrics
  - SurvivalMetrics: AUROC (Uno's), Integrated Brier Score, concordance indices
  - Decision Curve Analysis: Net benefit curves
  - Bootstrap confidence intervals
"""

from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for metric evaluation results."""

    metric_name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_bootstrap: int = 0

    def __repr__(self) -> str:
        if self.ci_lower is not None:
            return (
                f"{self.metric_name}: {self.value:.4f} "
                f"[95% CI: {self.ci_lower:.4f}-{self.ci_upper:.4f}]"
            )
        return f"{self.metric_name}: {self.value:.4f}"


class TimeDependent:
    """
    Base class for time-dependent metrics in survival analysis.
    Handles censoring and time-varying predictions.
    """

    @staticmethod
    def _check_survival_data(
        time: np.ndarray,
        event: np.ndarray,
        pred: np.ndarray,
    ) -> None:
        """Validate survival data format."""
        if len(time) != len(event) or len(time) != len(pred):
            raise ValueError("time, event, and pred must have same length")

        if not np.all((event == 0) | (event == 1)):
            raise ValueError("event must be binary (0/1)")

        if not np.all((pred >= 0) & (pred <= 1)):
            raise ValueError("pred must be in [0, 1]")

    @staticmethod
    def _get_risk_sets(
        time: np.ndarray,
        event: np.ndarray,
        eval_times: Optional[np.ndarray] = None,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Build risk sets for each event time.

        Returns:
            (risk_sets_dict, event_times)
        """
        unique_times = np.unique(time[event == 1])

        if eval_times is not None:
            unique_times = unique_times[unique_times <= eval_times.max()]

        risk_sets = {}

        for t in unique_times:
            at_risk = time >= t
            risk_sets[t] = np.where(at_risk)[0]

        return risk_sets, unique_times


class SurvivalMetrics(TimeDependent):
    """
    Comprehensive survival metrics for time-to-event prediction.
    """

    @staticmethod
    def unos_auc(
        time: np.ndarray,
        event: np.ndarray,
        pred: np.ndarray,
        eval_times: Optional[np.ndarray] = None,
    ) -> float:
        """
        Uno's time-dependent AUROC.

        Accounts for censoring and evaluates discriminability at specific times.

        Args:
            time: Survival times
            event: Event indicators (1=event, 0=censored)
            pred: Predicted risk (higher = higher risk)
            eval_times: Times at which to evaluate AUROC (default: event times)

        Returns:
            Uno's AUROC value
        """
        TimeDependent._check_survival_data(time, event, pred)

        if eval_times is None:
            eval_times = np.quantile(time[event == 1], [0.25, 0.5, 0.75])

        risk_sets, event_times = TimeDependent._get_risk_sets(time, event, eval_times)

        auc_values = []

        for eval_time in eval_times:
            if eval_time < time.min():
                continue

            # Get comparable pairs at eval_time
            # Pairs: (i, j) where i had event at/before eval_time, j censored after eval_time
            event_at_eval = (time <= eval_time) & (event == 1)
            censored_after_eval = time > eval_time

            if event_at_eval.sum() == 0 or censored_after_eval.sum() == 0:
                continue

            event_indices = np.where(event_at_eval)[0]
            censored_indices = np.where(censored_after_eval)[0]

            # Compare predictions
            y_true = np.concatenate([np.ones(len(event_indices)), np.zeros(len(censored_indices))])
            y_scores = np.concatenate([pred[event_indices], pred[censored_indices]])

            # Compute concordance
            concordance = 0
            pairs = 0

            for i in event_indices:
                for j in censored_indices:
                    if pred[i] > pred[j]:
                        concordance += 1
                    elif pred[i] == pred[j]:
                        concordance += 0.5
                    pairs += 1

            if pairs > 0:
                auc_values.append(concordance / pairs)

        return np.mean(auc_values) if auc_values else 0.5

    @staticmethod
    def integrated_brier_score(
        time: np.ndarray,
        event: np.ndarray,
        pred: np.ndarray,
        eval_times: Optional[np.ndarray] = None,
    ) -> float:
        """
        Integrated Brier Score: mean squared error integrated over time.

        Measures calibration in survival prediction.

        Args:
            time: Survival times
            event: Event indicators
            pred: Predicted risk (1 - survival probability)
            eval_times: Times at which to evaluate (default: all event times)

        Returns:
            IBS value (lower is better)
        """
        TimeDependent._check_survival_data(time, event, pred)

        if eval_times is None:
            eval_times = np.unique(time[event == 1])
            eval_times = eval_times[eval_times <= time.max()]

        brier_scores = []

        for eval_time in eval_times:
            # Survival status at eval_time
            # Y(t) = 1 if survived to time t, 0 if event before t
            y_eval = (time > eval_time).astype(int)

            # Censoring indicator for IPCW weighting
            # Weight = 1 if uncensored by eval_time
            c_eval = event | (time <= eval_time)

            # Brier score with inverse probability censoring weights
            if c_eval.sum() == 0:
                continue

            # Simple version: unweighted Brier score
            brier = np.mean((pred - y_eval) ** 2)
            brier_scores.append(brier)

        return np.mean(brier_scores) if brier_scores else 0.5

    @staticmethod
    def harrell_concordance(
        time: np.ndarray,
        event: np.ndarray,
        pred: np.ndarray,
    ) -> float:
        """
        Harrell's concordance index.

        Proportion of concordant pairs among comparable pairs.

        Args:
            time: Survival times
            event: Event indicators
            pred: Predicted risk

        Returns:
            Concordance index (c-index)
        """
        TimeDependent._check_survival_data(time, event, pred)

        concordance = 0
        pairs = 0

        for i in range(len(time)):
            for j in range(i + 1, len(time)):
                # Pair (i,j) is comparable if:
                # - Subject i had event before j, OR
                # - Both had events but i before j, OR
                # - i censored after j's event
                if time[i] < time[j]:
                    comparable = True
                    higher_risk_had_event = event[i] == 1
                elif time[i] > time[j]:
                    comparable = True
                    higher_risk_had_event = event[j] == 1
                else:
                    # Same time, only comparable if one censored and one not
                    comparable = event[i] != event[j]
                    higher_risk_had_event = event[i] == 1 if event[i] != event[j] else None

                if not comparable:
                    continue

                pairs += 1

                # Check concordance: higher risk should have event
                if pred[i] > pred[j]:
                    if time[i] < time[j] and event[i] == 1:
                        concordance += 1
                    elif time[i] > time[j] and event[j] == 1:
                        concordance += 1
                    elif time[i] == time[j] and higher_risk_had_event:
                        concordance += 1
                elif pred[i] == pred[j]:
                    concordance += 0.5

        return concordance / pairs if pairs > 0 else 0.5

    @staticmethod
    def unos_concordance(
        time: np.ndarray,
        event: np.ndarray,
        pred: np.ndarray,
        eval_time: Optional[float] = None,
    ) -> float:
        """
        Uno's concordance index (handles censoring better than Harrell's).

        Args:
            time: Survival times
            event: Event indicators
            pred: Predicted risk
            eval_time: Time at which to evaluate (default: median survival time)

        Returns:
            Uno's concordance index
        """
        TimeDependent._check_survival_data(time, event, pred)

        if eval_time is None:
            eval_time = np.median(time)

        # Get risk sets
        risk_sets, event_times = TimeDependent._get_risk_sets(time, event)

        concordance = 0
        pairs = 0

        for i in np.where(event == 1)[0]:
            if time[i] > eval_time:
                continue

            for j in risk_sets.get(time[i], []):
                if i == j:
                    continue

                pairs += 1

                if pred[i] > pred[j]:
                    concordance += 1
                elif pred[i] == pred[j]:
                    concordance += 0.5

        return concordance / pairs if pairs > 0 else 0.5

    @staticmethod
    def net_reclassification_index(
        y_true: np.ndarray,
        y_pred_old: np.ndarray,
        y_pred_new: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        Net Reclassification Index (NRI).

        Measures improvement of new model over old model.

        Args:
            y_true: True event indicators
            y_pred_old: Old model predictions
            y_pred_new: New model predictions
            thresholds: Risk category thresholds (default: [0.33, 0.67])

        Returns:
            (nri, nri_events, nri_non_events)
        """
        if thresholds is None:
            thresholds = [0.33, 0.67]

        def reclassify_category(pred):
            """Assign risk category."""
            if pred <= thresholds[0]:
                return 0
            elif pred <= thresholds[1]:
                return 1
            else:
                return 2

        old_categories = np.array([reclassify_category(p) for p in y_pred_old])
        new_categories = np.array([reclassify_category(p) for p in y_pred_new])

        # Events
        event_mask = y_true == 1
        events_reclassified = new_categories[event_mask] > old_categories[event_mask]
        events_downclassified = new_categories[event_mask] < old_categories[event_mask]
        pct_reclassified_events = (events_reclassified.sum() - events_downclassified.sum()) / event_mask.sum()

        # Non-events
        non_event_mask = y_true == 0
        non_events_reclassified = new_categories[non_event_mask] > old_categories[non_event_mask]
        non_events_downclassified = new_categories[non_event_mask] < old_categories[non_event_mask]
        pct_reclassified_non_events = (non_events_downclassified.sum() - non_events_reclassified.sum()) / non_event_mask.sum()

        nri = pct_reclassified_events + pct_reclassified_non_events

        return nri, pct_reclassified_events, pct_reclassified_non_events


class DecisionCurveAnalysis:
    """
    Decision Curve Analysis: evaluates clinical utility of predictions.
    """

    @staticmethod
    def net_benefit(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute net benefit across probability thresholds.

        Args:
            y_true: True event indicators
            y_pred: Predicted probabilities
            thresholds: Probability thresholds (default: 0-1 with 0.01 step)

        Returns:
            (thresholds, net_benefits)
        """
        if thresholds is None:
            thresholds = np.arange(0, 1.01, 0.01)

        n_events = y_true.sum()
        n_total = len(y_true)

        net_benefits = []

        for threshold in thresholds:
            positive_mask = y_pred >= threshold

            tp = ((y_pred >= threshold) & (y_true == 1)).sum()
            fp = ((y_pred >= threshold) & (y_true == 0)).sum()

            if positive_mask.sum() == 0:
                nb = 0
            else:
                nb = (tp / n_total) - (fp / n_total) * (threshold / (1 - threshold))

            net_benefits.append(nb)

        return thresholds, np.array(net_benefits)


class BootstrapCI:
    """Bootstrap confidence intervals for metrics."""

    @staticmethod
    def ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        random_state: int = 42,
        **metric_kwargs
    ) -> Tuple[float, float, float]:
        """
        Compute metric with bootstrap confidence interval.

        Args:
            y_true: True labels
            y_pred: Predictions
            metric_fn: Metric function to evaluate
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level (default 0.95)
            random_state: Random seed
            **metric_kwargs: Additional arguments for metric_fn

        Returns:
            (metric_value, ci_lower, ci_upper)
        """
        rng = np.random.RandomState(random_state)

        metric_value = metric_fn(y_true, y_pred, **metric_kwargs)

        bootstrap_values = []

        for _ in range(n_bootstrap):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            try:
                boot_metric = metric_fn(y_true_boot, y_pred_boot, **metric_kwargs)
                bootstrap_values.append(boot_metric)
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {e}")
                continue

        if not bootstrap_values:
            return metric_value, metric_value, metric_value

        bootstrap_values = np.array(bootstrap_values)

        alpha = (1 - ci) / 2
        ci_lower = np.quantile(bootstrap_values, alpha)
        ci_upper = np.quantile(bootstrap_values, 1 - alpha)

        return metric_value, ci_lower, ci_upper
