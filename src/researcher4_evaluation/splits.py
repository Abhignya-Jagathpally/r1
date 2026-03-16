"""
Patient-level data splits with temporal awareness and leakage detection.

Implements:
  - PatientLevelSplit: Ensures entire patient records stay together
  - TemporalCrossValidator: Time-aware splits respecting clinical timelines
  - StratifiedGroupKFold: Stratified by event, grouped by patient
  - LeakageDetector: Automated detection of future-data leakage
  - SplitAuditReport: Comprehensive split analysis and validation
"""

from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class SplitAuditReport:
    """Detailed audit report for data splits."""

    split_name: str
    timestamp: str
    n_patients: int
    n_samples: int
    n_folds: int
    event_rate: float

    fold_stats: List[Dict] = field(default_factory=list)
    leakage_detected: bool = False
    leakage_details: Dict = field(default_factory=dict)
    temporal_gaps: Dict = field(default_factory=dict)
    group_balance: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save audit report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        text = f"""
Split Audit Report: {self.split_name}
Generated: {self.timestamp}

Dataset Summary:
  - Patients: {self.n_patients}
  - Total Samples: {self.n_samples}
  - Event Rate: {self.event_rate:.4f}
  - Cross-Validation Folds: {self.n_folds}

Fold Statistics:
"""
        for i, stats in enumerate(self.fold_stats):
            text += f"""
  Fold {i}:
    - Train Samples: {stats['train_size']}
    - Valid Samples: {stats['valid_size']}
    - Train Event Rate: {stats['train_event_rate']:.4f}
    - Valid Event Rate: {stats['valid_event_rate']:.4f}
    - Train Patients: {stats['train_patients']}
    - Valid Patients: {stats['valid_patients']}
"""

        if self.leakage_detected:
            text += f"\nWARNING: Data leakage detected!\n{self.leakage_details}\n"
        else:
            text += "\nLeakage Check: PASSED\n"

        if self.temporal_gaps:
            text += f"\nTemporal Gaps:\n{json.dumps(self.temporal_gaps, indent=2, default=str)}\n"

        return text


class PatientLevelSplit:
    """
    Ensures entire patient records stay together during splits.
    Prevents patient-level leakage.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            test_size: Fraction of patients for test set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_patients_train = None
        self.n_patients_test = None

    def split(
        self,
        df: pd.DataFrame,
        patient_col: str = "patient_id",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split indices such that all samples for a patient stay together.

        Args:
            df: DataFrame with patient identifiers
            patient_col: Column name for patient ID

        Returns:
            (train_indices, test_indices)
        """
        rng = np.random.RandomState(self.random_state)
        patients = df[patient_col].unique()
        n_test = max(1, int(len(patients) * self.test_size))

        test_patients = rng.choice(patients, size=n_test, replace=False)
        test_mask = df[patient_col].isin(test_patients)

        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        self.n_patients_train = len(patients) - n_test
        self.n_patients_test = n_test

        logger.info(
            f"PatientLevelSplit: {self.n_patients_train} train patients, "
            f"{self.n_patients_test} test patients"
        )

        return train_idx, test_idx


class TemporalCrossValidator:
    """
    Time-aware cross-validator respecting temporal ordering.
    Implements expanding window and time-based folds.
    """

    def __init__(
        self,
        n_splits: int = 5,
        method: str = "expanding",
        min_train_samples: int = 100,
    ):
        """
        Args:
            n_splits: Number of temporal folds
            method: 'expanding' (expanding window) or 'sliding' (fixed window)
            min_train_samples: Minimum samples required in training set
        """
        self.n_splits = n_splits
        self.method = method
        self.min_train_samples = min_train_samples

        if method not in ["expanding", "sliding"]:
            raise ValueError(f"Unknown method: {method}")

    def split(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        patient_col: str = "patient_id",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate temporal CV folds.

        Args:
            df: DataFrame with temporal column
            time_col: Column name for timestamps
            patient_col: Column name for patient ID

        Returns:
            List of (train_idx, valid_idx) tuples
        """
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        n_samples = len(df_sorted)

        folds = []

        if self.method == "expanding":
            # Expanding window: training set grows, test set slides
            fold_size = n_samples // (self.n_splits + 1)

            for i in range(1, self.n_splits + 1):
                train_end = i * fold_size
                valid_end = (i + 1) * fold_size

                if valid_end > n_samples:
                    valid_end = n_samples

                if train_end < self.min_train_samples:
                    continue

                train_idx = np.arange(train_end)
                valid_idx = np.arange(train_end, valid_end)
                folds.append((train_idx, valid_idx))

        elif self.method == "sliding":
            # Fixed window: both train and test slide forward
            window_size = n_samples // (self.n_splits + 1)

            for i in range(self.n_splits):
                train_start = i * window_size
                train_end = train_start + window_size
                valid_start = train_end
                valid_end = min(valid_start + window_size, n_samples)

                if valid_end - valid_start < 1:
                    break

                train_idx = np.arange(train_start, train_end)
                valid_idx = np.arange(valid_start, valid_end)
                folds.append((train_idx, valid_idx))

        logger.info(f"TemporalCrossValidator: Generated {len(folds)} folds using {self.method}")
        return folds


class StratifiedGroupKFold:
    """
    K-fold CV stratified by event, grouped by patient.
    Ensures no patient leakage and balanced event distribution.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Args:
            n_splits: Number of folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self._group_kf = GroupKFold(n_splits=n_splits)
        self._stratified_kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified group folds.

        Args:
            X: Feature array
            y: Binary event indicators (0/1)
            groups: Patient group identifiers

        Returns:
            List of (train_idx, valid_idx) tuples
        """
        folds = []

        # Use stratified k-fold on groups to get stratification pattern
        for train_groups, valid_groups in self._stratified_kf.split(
            np.arange(len(np.unique(groups))),
            y=self._get_group_labels(y, groups)
        ):
            unique_groups = np.unique(groups)
            train_group_ids = unique_groups[train_groups]
            valid_group_ids = unique_groups[valid_groups]

            train_idx = np.where(np.isin(groups, train_group_ids))[0]
            valid_idx = np.where(np.isin(groups, valid_group_ids))[0]

            folds.append((train_idx, valid_idx))

        logger.info(
            f"StratifiedGroupKFold: {len(folds)} folds with "
            f"{len(np.unique(groups))} unique groups"
        )
        return folds

    @staticmethod
    def _get_group_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Get event label for each group (by majority)."""
        group_labels = []
        for group in np.unique(groups):
            mask = groups == group
            group_label = int(y[mask].mean() > 0.5)
            group_labels.append(group_label)
        return np.array(group_labels)


class LeakageDetector:
    """
    Automated detection of future-data leakage.
    Checks for temporal ordering violations and information leakage.
    """

    def __init__(self, time_col: str = "timestamp", patient_col: str = "patient_id"):
        """
        Args:
            time_col: Column name for timestamps
            patient_col: Column name for patient ID
        """
        self.time_col = time_col
        self.patient_col = patient_col

    def detect_leakage(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        lookahead_days: int = 0,
    ) -> Tuple[bool, Dict]:
        """
        Check for temporal leakage between train and validation sets.

        Args:
            df_train: Training set
            df_valid: Validation set
            lookahead_days: Expected temporal gap (days)

        Returns:
            (leakage_detected, leakage_details_dict)
        """
        leakage_details = {}
        leakage_detected = False

        # Check 1: Overlapping time ranges
        if self.time_col in df_train.columns and self.time_col in df_valid.columns:
            train_max_time = pd.to_datetime(df_train[self.time_col]).max()
            valid_min_time = pd.to_datetime(df_valid[self.time_col]).min()

            if train_max_time >= valid_min_time:
                leakage_detected = True
                leakage_details["temporal_overlap"] = {
                    "train_max": str(train_max_time),
                    "valid_min": str(valid_min_time),
                    "gap_days": (valid_min_time - train_max_time).days,
                }

        # Check 2: Patient overlap with future data
        if self.patient_col in df_train.columns and self.patient_col in df_valid.columns:
            common_patients = set(df_train[self.patient_col]) & set(df_valid[self.patient_col])

            if common_patients:
                # Check if validation data for a patient is earlier than training data
                for pid in list(common_patients)[:10]:  # Sample check
                    train_patient = df_train[df_train[self.patient_col] == pid]
                    valid_patient = df_valid[df_valid[self.patient_col] == pid]

                    if len(train_patient) > 0 and len(valid_patient) > 0:
                        if self.time_col in train_patient.columns:
                            train_time = pd.to_datetime(train_patient[self.time_col]).min()
                            valid_time = pd.to_datetime(valid_patient[self.time_col]).min()

                            if valid_time < train_time:
                                leakage_detected = True
                                leakage_details["patient_time_inversion"] = {
                                    "patient_id": str(pid),
                                    "train_min_time": str(train_time),
                                    "valid_min_time": str(valid_time),
                                }

        # Check 3: Feature value distribution changes
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            train_mean = df_train[col].mean()
            valid_mean = df_valid[col].mean()

            # Flag if mean differs by >50% (potential leakage indicator)
            if train_mean != 0 and abs((valid_mean - train_mean) / train_mean) > 0.5:
                if "feature_shifts" not in leakage_details:
                    leakage_details["feature_shifts"] = {}
                leakage_details["feature_shifts"][col] = {
                    "train_mean": float(train_mean),
                    "valid_mean": float(valid_mean),
                    "pct_change": float(abs((valid_mean - train_mean) / train_mean) * 100),
                }

        return leakage_detected, leakage_details

    def audit_split(
        self,
        df: pd.DataFrame,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        event_col: Optional[str] = None,
    ) -> SplitAuditReport:
        """
        Generate comprehensive audit report for splits.

        Args:
            df: Full dataset
            folds: List of (train_idx, valid_idx) tuples
            event_col: Optional event column for balance checking

        Returns:
            SplitAuditReport instance
        """
        report = SplitAuditReport(
            split_name="dataset_split",
            timestamp=datetime.now().isoformat(),
            n_patients=df[self.patient_col].nunique() if self.patient_col in df.columns else len(df),
            n_samples=len(df),
            n_folds=len(folds),
            event_rate=df[event_col].mean() if event_col and event_col in df.columns else 0.0,
        )

        overall_leakage_detected = False

        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            df_train = df.iloc[train_idx]
            df_valid = df.iloc[valid_idx]

            # Leakage check
            leakage, details = self.detect_leakage(df_train, df_valid)
            if leakage:
                overall_leakage_detected = True
                if "fold_leakages" not in report.leakage_details:
                    report.leakage_details["fold_leakages"] = {}
                report.leakage_details["fold_leakages"][fold_idx] = details

            # Fold statistics
            fold_stat = {
                "fold_idx": fold_idx,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                "train_event_rate": df_train[event_col].mean() if event_col else 0.0,
                "valid_event_rate": df_valid[event_col].mean() if event_col else 0.0,
                "train_patients": df_train[self.patient_col].nunique() if self.patient_col in df.columns else len(train_idx),
                "valid_patients": df_valid[self.patient_col].nunique() if self.patient_col in df.columns else len(valid_idx),
            }
            report.fold_stats.append(fold_stat)

            # Temporal gaps
            if self.time_col in df.columns:
                train_max = pd.to_datetime(df_train[self.time_col]).max()
                valid_min = pd.to_datetime(df_valid[self.time_col]).min()
                report.temporal_gaps[f"fold_{fold_idx}"] = {
                    "train_max": str(train_max),
                    "valid_min": str(valid_min),
                    "gap_days": (valid_min - train_max).days,
                }

        report.leakage_detected = overall_leakage_detected

        logger.info(
            f"Audit completed: {len(folds)} folds, "
            f"leakage detected: {overall_leakage_detected}"
        )

        return report
