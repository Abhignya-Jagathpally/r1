"""
Automated reporting and visualization.
 
Provides:
  - ExperimentReporter: Markdown reports with plots
  - DeLong test: Statistical comparison of model predictions
  - Model comparison dashboards
  - LaTeX table generation
"""
 
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz
 
logger = logging.getLogger(__name__)
 
matplotlib.use("Agg")
 
 
@dataclass
class ComparisonResult:
    """Result of model comparison."""
 
    model_1_name: str
    model_2_name: str
    metric_name: str
    model_1_value: float
    model_2_value: float
    difference: float
    pvalue: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
 
 
class DeLongTest:
    """
    DeLong test for comparing AUROC of two classifiers.
    Handles dependencies between predictions (e.g., same test set).
    """
 
    @staticmethod
    def auc_error_variance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate variance of AUC estimate (for use in DeLong test).
 
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
 
        Returns:
            Variance of AUC
        """
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
 
        if n_pos == 0 or n_neg == 0:
            return 0.0
 
        # Separate positive and negative predictions
        y_pred_pos = y_pred[y_true == 1]
        y_pred_neg = y_pred[y_true == 0]
 
        # Compute pairwise comparisons
        # V10: variance for positive examples
        v10 = np.zeros(n_pos)
        for i in range(n_pos):
            # Proportion of negative examples predicted lower than this positive
            v10[i] = ((y_pred_neg < y_pred_pos[i]).sum() +
                     0.5 * (y_pred_neg == y_pred_pos[i]).sum()) / n_neg
 
        # V01: variance for negative examples
        v01 = np.zeros(n_neg)
        for j in range(n_neg):
            # Proportion of positive examples predicted higher than this negative
            v01[j] = ((y_pred_pos > y_pred_neg[j]).sum() +
                     0.5 * (y_pred_pos == y_pred_neg[j]).sum()) / n_pos
 
        # Variance components
        s10_sq = np.var(v10, ddof=1) if n_pos > 1 else 0
        s01_sq = np.var(v01, ddof=1) if n_neg > 1 else 0
 
        variance = (s10_sq / n_pos) + (s01_sq / n_neg)
 
        return max(variance, 1e-10)  # Avoid zero variance
 
    @staticmethod
    def compare(
        y_true: np.ndarray,
        y_pred_1: np.ndarray,
        y_pred_2: np.ndarray,
    ) -> Tuple[float, float]:
        """
        DeLong test comparing two classifiers on same test set.
 
        Args:
            y_true: True binary labels
            y_pred_1: Predictions from classifier 1
            y_pred_2: Predictions from classifier 2
 
        Returns:
            (z_statistic, p_value)
        """
        from sklearn.metrics import roc_auc_score
 
        auc_1 = roc_auc_score(y_true, y_pred_1)
        auc_2 = roc_auc_score(y_true, y_pred_2)
 
        var_1 = DeLongTest.auc_error_variance(y_true, y_pred_1)
        var_2 = DeLongTest.auc_error_variance(y_true, y_pred_2)
 
        # Covariance term (simplified)
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
 
        y_pred_1_pos = y_pred_1[y_true == 1]
        y_pred_1_neg = y_pred_1[y_true == 0]
        y_pred_2_pos = y_pred_2[y_true == 1]
        y_pred_2_neg = y_pred_2[y_true == 0]
 
        cov_pos = np.cov(y_pred_1_pos, y_pred_2_pos)[0, 1]
        cov_neg = np.cov(y_pred_1_neg, y_pred_2_neg)[0, 1]
 
        covariance = (cov_pos / n_pos) + (cov_neg / n_neg)
 
        # SE of difference
        se = np.sqrt(var_1 + var_2 - 2 * covariance)
 
        if se < 1e-10:
            return 0.0, 1.0
 
        z_stat = (auc_1 - auc_2) / se
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
 
        return z_stat, p_value
 
 
class ExperimentReporter:
    """
    Generate automated markdown reports and visualizations.
    """
 
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize reporter.
 
        Args:
            output_dir: Directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
 
    def generate_calibration_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Calibration Curve",
        output_name: str = "calibration_curve.png",
    ) -> Path:
        """
        Generate calibration curve plot.
 
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            title: Plot title
            output_name: Output filename
 
        Returns:
            Path to saved figure
        """
        n_bins = 10
        bin_sums = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)
 
        bin_edges = np.linspace(0, 1, n_bins + 1)
 
        for i in range(n_bins):
            in_bin = (y_pred > bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            if in_bin.sum() > 0:
                bin_sums[i] = y_pred[in_bin].mean()
                bin_true[i] = y_true[in_bin].mean()
                bin_total[i] = in_bin.sum()
 
        fig, ax = plt.subplots(figsize=(8, 6))
 
        # Only plot bins with samples
        valid_bins = bin_total > 0
        ax.scatter(bin_sums[valid_bins], bin_true[valid_bins],
                  s=bin_total[valid_bins] * 2, alpha=0.6, label="Model")
 
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.5)
 
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Empirical)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
 
        output_path = self.output_dir / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
 
        logger.info(f"Calibration plot saved to {output_path}")
        return output_path
 
    def generate_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "ROC Curve",
        output_name: str = "roc_curve.png",
    ) -> Path:
        """
        Generate ROC curve plot.
 
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            title: Plot title
            output_name: Output filename
 
        Returns:
            Path to saved figure
        """
        from sklearn.metrics import roc_curve, auc
 
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
 
        fig, ax = plt.subplots(figsize=(8, 6))
 
        ax.plot(fpr, tpr, color="b", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
 
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
 
        output_path = self.output_dir / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
 
        logger.info(f"ROC curve saved to {output_path}")
        return output_path
 
    def generate_model_comparison_report(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
        output_name: str = "model_comparison.md",
    ) -> Path:
        """
        Generate model comparison markdown report.
 
        Args:
            models_metrics: Dict mapping model names to metric dicts
            title: Report title
            output_name: Output filename
 
        Returns:
            Path to saved report
        """
        report = f"# {title}\n\n"
        report += f"Generated: {pd.Timestamp.now().isoformat()}\n\n"
 
        # Metrics table
        report += "## Performance Metrics\n\n"
        report += "| Model | "
        all_metrics = set()
        for metrics in models_metrics.values():
            all_metrics.update(metrics.keys())
 
        for metric in sorted(all_metrics):
            report += f"{metric} | "
        report += "\n"
 
        report += "| --- | " + "--- | " * len(all_metrics) + "\n"
 
        for model_name, metrics in models_metrics.items():
            report += f"| {model_name} | "
            for metric in sorted(all_metrics):
                value = metrics.get(metric, "-")
                if isinstance(value, float):
                    report += f"{value:.4f} | "
                else:
                    report += f"{value} | "
            report += "\n"
 
        report += "\n"
 
        output_path = self.output_dir / output_name
        with open(output_path, "w") as f:
            f.write(report)
 
        logger.info(f"Model comparison report saved to {output_path}")
        return output_path
 
    def generate_comprehensive_report(
        self,
        experiment_name: str,
        results_dict: Dict,
        plots: Optional[List[Path]] = None,
    ) -> Path:
        """
        Generate comprehensive experiment report.
 
        Args:
            experiment_name: Name of experiment
            results_dict: Dictionary of results
            plots: List of plot paths to include
 
        Returns:
            Path to saved report
        """
        report = f"# {experiment_name} Report\n\n"
        report += f"**Generated**: {pd.Timestamp.now().isoformat()}\n\n"
 
        report += "## Summary\n\n"
        for key, value in results_dict.items():
            if not isinstance(value, (dict, list)):
                report += f"- **{key}**: {value}\n"
 
        report += "\n## Metrics\n\n"
        if "metrics" in results_dict:
            for metric_name, metric_value in results_dict["metrics"].items():
                if isinstance(metric_value, float):
                    report += f"- **{metric_name}**: {metric_value:.4f}\n"
                else:
                    report += f"- **{metric_name}**: {metric_value}\n"
 
        report += "\n## Plots\n\n"
        if plots:
            for plot_path in plots:
                relative_path = plot_path.relative_to(self.output_dir)
                report += f"![{plot_path.stem}]({relative_path})\n\n"
 
        output_path = self.output_dir / f"{experiment_name}_report.md"
        with open(output_path, "w") as f:
            f.write(report)
 
        logger.info(f"Comprehensive report saved to {output_path}")
        return output_path
 
 
class LaTeXTableGenerator:
    """Generate LaTeX tables from results."""
 
    @staticmethod
    def metrics_table(
        models_metrics: Dict[str, Dict[str, float]],
        caption: str = "Model Performance Metrics",
        label: str = "table:metrics",
    ) -> str:
        """
        Generate LaTeX metrics table.
 
        Args:
            models_metrics: Dict mapping model names to metric dicts
            caption: Table caption
            label: Table label
 
        Returns:
            LaTeX table string
        """
        all_metrics = set()
        for metrics in models_metrics.values():
            all_metrics.update(metrics.keys())
 
        sorted_metrics = sorted(all_metrics)
        n_cols = len(sorted_metrics) + 1
 
        latex = f"\\begin{{table}}[H]\n"
        latex += f"\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += f"\\begin{{tabular}}{{{'l' + 'c' * len(sorted_metrics)}}}\n"
        latex += f"\\toprule\n"
 
        # Header
        latex += "Model"
        for metric in sorted_metrics:
            latex += f" & {metric}"
        latex += " \\\\\n"
        latex += "\\midrule\n"
 
        # Rows
        for model_name in sorted(models_metrics.keys()):
            latex += model_name
            for metric in sorted_metrics:
                value = models_metrics[model_name].get(metric, "-")
                if isinstance(value, float):
                    latex += f" & {value:.4f}"
                else:
                    latex += f" & {value}"
            latex += " \\\\\n"
 
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
 
        return latex
 
