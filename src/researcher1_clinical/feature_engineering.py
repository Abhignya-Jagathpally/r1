"""
Feature Engineering Module: Temporal and Clinical Summary Features

Builds analysis-ready features from cleaned data:
- Temporal features: slopes, rolling means/variance, time-since-last-treatment
- CRAB/SLiM-CRAB criteria summaries
- Trajectory window aggregations (3, 6, 12-month): mean, std, delta, max, min
- Output to analysis-ready Parquet

Features are computed after cleansing and designed for downstream modeling.
Temporal windows are parameterizable to support sensitivity analyses.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class CRABCriteria:
    """
    CRAB myeloma defining criteria (original, now deprecated in favor of SLiM-CRAB).

    C: Calcium >= 11 mg/dL
    R: Renal dysfunction: Creatinine >= 2 mg/dL
    A: Anemia: Hemoglobin < 10 g/dL
    B: Bone lesions: Lytic lesions on imaging
    """
    pass


@dataclass
class SLiMCRABCriteria:
    """
    SLiM-CRAB myeloma defining criteria (current standard).

    S: Clonality: >=60% plasma cells
    Li: Light chain ratio: Involved/Uninvolved FLC >= 100
    M: MRI: >=1 focal lesion on diffusion-weighted imaging
    C: Calcium >= 11 mg/dL
    R: Renal dysfunction: eGFR < 40 mL/min/1.73m2
    A: Anemia: Hemoglobin < 10 g/dL
    B: Bone lesions: >=1 lytic lesion on CT/plain film
    """
    pass


@dataclass
class TemporalWindowConfig:
    """Configuration for trajectory window aggregations."""
    windows_days: List[int] = None
    methods: List[str] = None

    def __post_init__(self):
        if self.windows_days is None:
            self.windows_days = [90, 180, 365]  # 3, 6, 12 months
        if self.methods is None:
            self.methods = ["mean", "std", "delta", "max", "min"]


class FeatureEngineer:
    """
    Temporal and clinical feature engineering for MM digital twin.

    Computes:
    - Temporal features: slopes, rolling windows, time-since-last-treatment
    - SLiM-CRAB criteria fulfillment
    - Trajectory window aggregations
    - Outputs analysis-ready Parquet

    Features are computed per patient per timepoint, preserving longitudinal structure.
    """

    LAB_COLUMNS = [
        "serum_m_protein_g_dl",
        "free_light_chain_kappa_mg_l",
        "free_light_chain_lambda_mg_l",
        "free_light_chain_ratio",
        "hemoglobin_g_dl",
        "calcium_mg_dl",
        "creatinine_mg_dl",
        "albumin_g_dl",
        "beta2_microglobulin_mg_l",
        "ldh_u_l",
    ]

    def __init__(self, window_config: Optional[TemporalWindowConfig] = None):
        """
        Initialize feature engineer.

        Args:
            window_config: Temporal window configuration
        """
        self.window_config = window_config or TemporalWindowConfig()
        logger.info(f"Initialized FeatureEngineer with windows={self.window_config.windows_days}d")

    def compute_temporal_slopes(
        self,
        df: pd.DataFrame,
        lookback_days: int = 180,
    ) -> pd.DataFrame:
        """
        Compute lab value slopes (trend) over lookback window.

        Uses linear regression on all available visits within lookback period
        to estimate trajectory (positive = increasing, negative = decreasing).

        Args:
            df: Cleaned long-format DataFrame (requires timepoint in days)
            lookback_days: Days to look back from each visit

        Returns:
            DataFrame with slope columns (lab_slope_* columns)
        """
        slopes = pd.DataFrame(index=df.index)

        for patient in df["patient_id"].unique():
            patient_data = df[df["patient_id"] == patient].copy()
            patient_data = patient_data.sort_values("timepoint")

            for idx, (_, row) in enumerate(patient_data.iterrows()):
                current_time = row["timepoint"]
                window_start = current_time - lookback_days

                # Get all visits in lookback window
                window_data = patient_data[
                    (patient_data["timepoint"] >= window_start) &
                    (patient_data["timepoint"] <= current_time)
                ]

                if len(window_data) < 2:
                    # Insufficient data for slope
                    for col in self.LAB_COLUMNS:
                        slopes.loc[row.name, f"{col}_slope"] = np.nan
                    continue

                # Linear regression
                x = window_data["timepoint"].values
                for col in self.LAB_COLUMNS:
                    if col not in df.columns:
                        continue

                    y = window_data[col].values
                    if np.isnan(y).all():
                        slopes.loc[row.name, f"{col}_slope"] = np.nan
                        continue

                    # Only use non-NaN points
                    valid_idx = ~np.isnan(y)
                    if valid_idx.sum() < 2:
                        slopes.loc[row.name, f"{col}_slope"] = np.nan
                        continue

                    try:
                        result = linregress(x[valid_idx], y[valid_idx])
                        slopes.loc[row.name, f"{col}_slope"] = result.slope
                    except Exception as e:
                        logger.debug(f"Slope computation failed for {col}: {e}")
                        slopes.loc[row.name, f"{col}_slope"] = np.nan

        logger.info(f"Computed slopes for {len(slopes.columns)} lab columns")
        return slopes

    def compute_rolling_windows(
        self,
        df: pd.DataFrame,
        window_days: int = 90,
        min_periods: int = 2,
    ) -> pd.DataFrame:
        """
        Compute rolling mean and variance over temporal window.

        Args:
            df: Cleaned long-format DataFrame
            window_days: Size of rolling window in days
            min_periods: Minimum observations required in window

        Returns:
            DataFrame with rolling_mean_* and rolling_std_* columns
        """
        rolling = pd.DataFrame(index=df.index)

        for patient in df["patient_id"].unique():
            patient_data = df[df["patient_id"] == patient].copy()
            patient_data = patient_data.sort_values("timepoint")

            for idx, (_, row) in enumerate(patient_data.iterrows()):
                current_time = row["timepoint"]
                window_start = current_time - window_days

                # Get visits in window
                window_data = patient_data[
                    (patient_data["timepoint"] >= window_start) &
                    (patient_data["timepoint"] <= current_time)
                ]

                for col in self.LAB_COLUMNS:
                    if col not in df.columns:
                        continue

                    values = window_data[col].dropna().values

                    if len(values) < min_periods:
                        rolling.loc[row.name, f"{col}_rolling_mean"] = np.nan
                        rolling.loc[row.name, f"{col}_rolling_std"] = np.nan
                    else:
                        rolling.loc[row.name, f"{col}_rolling_mean"] = values.mean()
                        rolling.loc[row.name, f"{col}_rolling_std"] = values.std()

        logger.info(f"Computed rolling windows for {len(self.LAB_COLUMNS)} lab columns")
        return rolling

    def compute_time_since_last_treatment(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute days since last treatment initiation.

        Args:
            df: DataFrame with treatment line information

        Returns:
            Series: days_since_treatment (NaN if no prior treatment)
        """
        days_since = pd.Series(np.nan, index=df.index)

        for patient in df["patient_id"].unique():
            patient_data = df[df["patient_id"] == patient].copy()
            patient_data = patient_data.sort_values("timepoint")

            treatment_starts = patient_data[patient_data["treatment_line"] > 0]

            if len(treatment_starts) == 0:
                continue

            # For each row, find last treatment initiation
            for idx, (_, row) in enumerate(patient_data.iterrows()):
                prior_treatments = treatment_starts[treatment_starts["timepoint"] <= row["timepoint"]]
                if len(prior_treatments) > 0:
                    last_treatment_time = prior_treatments["timepoint"].iloc[-1]
                    days_since.loc[row.name] = row["timepoint"] - last_treatment_time

        logger.info("Computed time-since-last-treatment")
        return pd.DataFrame({"days_since_treatment": days_since})

    def assess_slim_crab_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assess SLiM-CRAB myeloma-defining event criteria per IMWG 2014.

        Reference: Rajkumar SV et al., Lancet Oncol 2014;15:e538-e548.

        CRAB criteria (end-organ damage):
          C: Calcium > 11 mg/dL (hypercalcemia)
          R: Renal insufficiency — creatinine clearance < 40 mL/min OR
             serum creatinine > 2 mg/dL (creatinine used as proxy here)
          A: Anemia — hemoglobin < 10 g/dL (or >2 g/dL below lower limit of normal)
          B: Bone disease — one or more lytic lesions (not available from labs)

        SLiM criteria (biomarkers of near-inevitable progression to end-organ damage):
          S: Clonal bone marrow plasma cells >= 60% (not available from blood work)
          Li: Involved/uninvolved serum FLC ratio >= 100 (with involved FLC >= 100 mg/L)
          M: > 1 focal lesion (>= 5 mm) on MRI (not available from blood work)

        Args:
            df: DataFrame with lab values

        Returns:
            DataFrame with crab_*, slim_*, and myeloma_defining_event_any columns
        """
        criteria = pd.DataFrame(index=df.index)

        # --- CRAB criteria ---
        if "calcium_mg_dl" in df.columns:
            criteria["crab_calcium"] = (df["calcium_mg_dl"] > 11).astype(int)

        if "creatinine_mg_dl" in df.columns:
            # Proxy for renal criterion: serum creatinine > 2 mg/dL
            # Full IMWG definition also includes creatinine clearance < 40 mL/min
            criteria["crab_renal"] = (df["creatinine_mg_dl"] > 2).astype(int)

        if "hemoglobin_g_dl" in df.columns:
            criteria["crab_anemia"] = (df["hemoglobin_g_dl"] < 10).astype(int)

        # crab_bone: not available from lab data (requires imaging)

        # --- SLiM criteria ---
        if "free_light_chain_ratio" in df.columns:
            # Li criterion: involved/uninvolved FLC ratio >= 100
            # Using absolute FLC ratio as proxy for involved/uninvolved ratio
            criteria["slim_flc_ratio"] = (df["free_light_chain_ratio"] >= 100).astype(int)

        # S criterion: bone marrow clonal plasma cells >= 60% (not available from blood work)
        criteria["slim_plasma_cells"] = np.nan

        # M criterion: > 1 focal lesion on MRI (not available from blood work)
        criteria["slim_mri_lesions"] = np.nan

        # --- Combined myeloma-defining event flag ---
        assessable_cols = [
            col for col in criteria.columns
            if (col.startswith("crab_") or col.startswith("slim_"))
            and criteria[col].notna().any()
        ]
        if assessable_cols:
            criteria["myeloma_defining_event_any"] = (
                criteria[assessable_cols].sum(axis=1) > 0
            ).astype(int)
        else:
            criteria["myeloma_defining_event_any"] = np.nan

        logger.info("Assessed SLiM-CRAB criteria (IMWG 2014)")
        return criteria

    def aggregate_trajectory_windows(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate lab trajectories over fixed windows (3, 6, 12 months).

        For each patient at each timepoint, computes mean, std, delta (current - first),
        max, min of each lab value within the window prior to current timepoint.

        Args:
            df: Long-format DataFrame with timepoint (assumed days from baseline)

        Returns:
            DataFrame with trajectory aggregations: {lab}_{window}d_{method}
        """
        agg_features = pd.DataFrame(index=df.index)

        for patient in df["patient_id"].unique():
            patient_data = df[df["patient_id"] == patient].copy()
            patient_data = patient_data.sort_values("timepoint")

            for idx, (_, row) in enumerate(patient_data.iterrows()):
                current_time = row["timepoint"]

                for window_days in self.window_config.windows_days:
                    window_start = current_time - window_days

                    # Get all visits in window
                    window_data = patient_data[
                        (patient_data["timepoint"] >= window_start) &
                        (patient_data["timepoint"] <= current_time)
                    ]

                    for col in self.LAB_COLUMNS:
                        if col not in df.columns:
                            continue

                        values = window_data[col].dropna().values

                        if len(values) == 0:
                            continue

                        # Mean
                        agg_features.loc[row.name, f"{col}_{window_days}d_mean"] = values.mean()

                        # Std
                        if len(values) > 1:
                            agg_features.loc[row.name, f"{col}_{window_days}d_std"] = values.std()
                        else:
                            agg_features.loc[row.name, f"{col}_{window_days}d_std"] = 0.0

                        # Delta (current - first)
                        agg_features.loc[row.name, f"{col}_{window_days}d_delta"] = (
                            values[-1] - values[0] if len(values) > 1 else 0.0
                        )

                        # Max
                        agg_features.loc[row.name, f"{col}_{window_days}d_max"] = values.max()

                        # Min
                        agg_features.loc[row.name, f"{col}_{window_days}d_min"] = values.min()

        logger.info(
            f"Aggregated {len(self.LAB_COLUMNS)} labs over "
            f"{len(self.window_config.windows_days)} windows"
        )
        return agg_features

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features end-to-end.

        Args:
            df: Cleaned long-format DataFrame

        Returns:
            DataFrame with original + all engineered features
        """
        result = df.copy()

        # Temporal slopes
        slopes = self.compute_temporal_slopes(df)
        result = pd.concat([result, slopes], axis=1)

        # Rolling windows
        rolling = self.compute_rolling_windows(df)
        result = pd.concat([result, rolling], axis=1)

        # Time since treatment
        time_since_tx = self.compute_time_since_last_treatment(df)
        result = pd.concat([result, time_since_tx], axis=1)

        # SLiM-CRAB criteria
        criteria = self.assess_slim_crab_criteria(df)
        result = pd.concat([result, criteria], axis=1)

        # Trajectory windows
        trajectories = self.aggregate_trajectory_windows(df)
        result = pd.concat([result, trajectories], axis=1)

        logger.info(
            f"Engineered features: {df.shape[1]} -> {result.shape[1]} columns"
        )
        return result

    def to_parquet(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save analysis-ready features to Parquet.

        Args:
            df: Engineered feature DataFrame
            output_path: Path to save Parquet file
        """
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Saved {df.shape[0]} rows to {output_path}")
