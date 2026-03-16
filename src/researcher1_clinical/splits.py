"""
Data Splitting Module: Patient-Level, Time-Aware, and Stratified Partitioning

Provides production-ready data split strategies:
- Patient-level splits: All visits of a patient go to same fold
- Time-aware splits: Temporally ordered splits (train-test, forward-chaining)
- K-fold with patient grouping: Cross-validation without patient leakage
- Stratified splitting: Balances endpoints across folds

Key constraint: All visits of a single patient must stay in the same fold
to prevent data leakage in longitudinal models.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    GroupKFold,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting strategies."""
    strategy: str  # "patient_level", "time_aware", "stratified_group_kfold"
    test_size: float = 0.2
    val_size: float = 0.1
    n_splits: int = 5
    random_state: int = 42
    stratify_column: Optional[str] = None
    time_column: str = "timepoint"


class DataSplitter:
    """
    Production data splitting with patient-level grouping.

    Strategies:
    1. patient_level: Split at patient level (all visits of patient together)
    2. time_aware: Ordered temporal splits (useful for time-series validation)
    3. stratified_group_kfold: K-fold CV with patient grouping and stratification

    Key constraint: Prevents patient leakage by ensuring all visits of a
    patient appear in exactly one fold.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize splitter.

        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig(strategy="stratified_group_kfold")
        logger.info(f"Initialized DataSplitter with strategy={self.config.strategy}")

    def _get_patient_groups(self, df: pd.DataFrame) -> np.ndarray:
        """
        Map each row to patient index for grouping.

        Args:
            df: DataFrame with patient_id column

        Returns:
            Array of patient group indices
        """
        patient_unique, patient_mapping = pd.factorize(df["patient_id"])
        return patient_unique

    def patient_level_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data at patient level.

        All visits of each patient stay together. Partitions patients (not rows)
        into train/val/test at specified ratios.

        Args:
            df: DataFrame with patient_id column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        patients = df["patient_id"].unique()
        n_patients = len(patients)

        np.random.seed(self.config.random_state)
        perm = np.random.permutation(n_patients)

        test_size = int(n_patients * self.config.test_size)
        val_size = int(n_patients * self.config.val_size)
        train_size = n_patients - test_size - val_size

        train_patients = patients[perm[:train_size]]
        val_patients = patients[perm[train_size:train_size + val_size]]
        test_patients = patients[perm[train_size + val_size:]]

        train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
        val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
        test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

        logger.info(
            f"Patient-level split: "
            f"train={len(train_patients)} patients ({len(train_df)} rows), "
            f"val={len(val_patients)} patients ({len(val_df)} rows), "
            f"test={len(test_patients)} patients ({len(test_df)} rows)"
        )

        return train_df, val_df, test_df

    def time_aware_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-aware split: earlier timepoints to train, later to test.

        Useful for validating temporal generalization. Maintains patient-level
        grouping: all visits of a patient go to same fold, ordered by median timepoint.

        Args:
            df: DataFrame with timepoint column and patient_id

        Returns:
            Tuple of (train_df, val_df, test_df) ordered temporally
        """
        # Get median timepoint per patient
        patient_median_time = df.groupby("patient_id")["timepoint"].median().reset_index()
        patient_median_time = patient_median_time.sort_values("timepoint")

        patients_sorted = patient_median_time["patient_id"].values
        n_patients = len(patients_sorted)

        test_size = int(n_patients * self.config.test_size)
        val_size = int(n_patients * self.config.val_size)

        train_patients = patients_sorted[:-test_size - val_size]
        val_patients = patients_sorted[-test_size - val_size:-test_size]
        test_patients = patients_sorted[-test_size:]

        train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
        val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
        test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

        logger.info(
            f"Time-aware split: "
            f"train median_time={patient_median_time[patient_median_time['patient_id'].isin(train_patients)]['timepoint'].median():.0f}, "
            f"val median_time={patient_median_time[patient_median_time['patient_id'].isin(val_patients)]['timepoint'].median():.0f}, "
            f"test median_time={patient_median_time[patient_median_time['patient_id'].isin(test_patients)]['timepoint'].median():.0f}"
        )

        return train_df, val_df, test_df

    def stratified_group_kfold(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        K-fold CV with patient grouping and outcome stratification.

        Prevents patient leakage while maintaining balanced outcome distribution
        across folds. Requires stratify_column in config.

        Args:
            df: DataFrame with patient_id and (optionally) stratify column

        Returns:
            List of (train_indices, test_indices) tuples for each fold
        """
        if self.config.stratify_column is None:
            logger.warning("No stratify_column specified; falling back to GroupKFold")
            splitter = GroupKFold(n_splits=self.config.n_splits)
            groups = self._get_patient_groups(df)
            folds = list(splitter.split(df, groups=groups))
        else:
            # Stratified group k-fold
            splitter = StratifiedGroupKFold(
                n_splits=self.config.n_splits,
                random_state=self.config.random_state,
                shuffle=True,
            )
            groups = self._get_patient_groups(df)
            y = df[self.config.stratify_column].values

            folds = list(splitter.split(df, y=y, groups=groups))

        logger.info(
            f"Created {len(folds)}-fold CV with patient grouping"
            + (f" stratified by {self.config.stratify_column}" if self.config.stratify_column else "")
        )

        return folds

    def split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute splitting strategy specified in config.

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.config.strategy == "patient_level":
            return self.patient_level_split(df)
        elif self.config.strategy == "time_aware":
            return self.time_aware_split(df)
        elif self.config.strategy == "stratified_group_kfold":
            # For stratified_group_kfold, we return train/val/test instead of folds
            # Convert by using first fold as test, second as val, rest as train
            folds = self.stratified_group_kfold(df)
            if len(folds) < 2:
                raise ValueError("Need at least 2 folds to create train/val/test")

            # Use first fold as test
            test_idx = folds[0][1]
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Use second fold as val
            val_idx = folds[1][1]
            val_df = df.iloc[val_idx].reset_index(drop=True)

            # Combine rest for train
            train_idx = folds[0][0]
            train_df = df.iloc[train_idx].reset_index(drop=True)

            logger.info(f"Stratified group k-fold -> train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)} rows")
            return train_df, val_df, test_df
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def get_fold_indices(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get k-fold indices for cross-validation.

        Only applicable for stratified_group_kfold strategy.

        Args:
            df: DataFrame

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.config.strategy != "stratified_group_kfold":
            raise ValueError(f"get_fold_indices only supports stratified_group_kfold, got {self.config.strategy}")

        return self.stratified_group_kfold(df)

    def summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Summarize split results.

        Args:
            df: Original DataFrame

        Returns:
            Dictionary with split statistics
        """
        train_df, val_df, test_df = self.split(df)

        return {
            "strategy": self.config.strategy,
            "n_patients": df["patient_id"].nunique(),
            "n_total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_patients": train_df["patient_id"].nunique(),
            "val_patients": val_df["patient_id"].nunique(),
            "test_patients": test_df["patient_id"].nunique(),
        }
