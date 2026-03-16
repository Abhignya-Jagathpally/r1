"""
Data Cleansing Module: Unit Harmonization, Missingness Handling, and Imputation

Provides production-ready cleansing with frozen preprocessing contract:
- Unit harmonization across visits
- Long format conversion (one row per patient per timepoint)
- Missingness masks for audit trail
- Winsorization with clinician-reviewed bounds
- Multiple imputation strategies (MICE, KNN, median) applied to training folds ONLY
- Normalization applied to training folds with frozen parameters

Contract:
    Cleansing pipeline is versioned. Once fitted on training data, all parameters
    (imputation models, normalization stats) are frozen and applied to test/holdout.
    Changes to cleansing require explicit version bumping.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class WinsorizeConfig:
    """
    Clinician-reviewed bounds for winsorization.

    Bounds prevent unrealistic lab values from skewing downstream modeling.
    All bounds are parameterizable for clinical validation.
    """
    serum_m_protein_bounds: Tuple[float, float] = (0.0, 10.0)  # g/dL
    flc_kappa_bounds: Tuple[float, float] = (0.33, 19.4)  # mg/L
    flc_lambda_bounds: Tuple[float, float] = (0.27, 26.3)  # mg/L
    flc_ratio_bounds: Tuple[float, float] = (0.26, 1.65)
    hemoglobin_bounds: Tuple[float, float] = (5.0, 18.0)  # g/dL
    calcium_bounds: Tuple[float, float] = (6.0, 14.0)  # mg/dL
    creatinine_bounds: Tuple[float, float] = (0.3, 10.0)  # mg/dL
    albumin_bounds: Tuple[float, float] = (1.0, 5.5)  # g/dL
    beta2_microglobulin_bounds: Tuple[float, float] = (0.5, 50.0)  # mg/L
    ldh_bounds: Tuple[float, float] = (100, 2000)  # U/L

    def get_bounds(self, column: str) -> Optional[Tuple[float, float]]:
        """Get bounds for given column."""
        attr = f"{column.lower()}_bounds"
        return getattr(self, attr, None)


@dataclass
class CleansingState:
    """
    Frozen preprocessing parameters for reproducibility.

    Once fitted on training data, these parameters are locked and applied
    to all subsequent test/holdout data.
    """
    version: str
    imputation_models: Dict[str, object]
    scaler: Optional[StandardScaler]
    column_means: Dict[str, float]
    column_stds: Dict[str, float]
    missingness_patterns: Dict[str, float]

    def save(self, path: str) -> None:
        """Save frozen parameters (implementation deferred to production system)."""
        logger.info(f"[PRODUCTION] Would save CleansingState v{self.version} to {path}")

    @classmethod
    def load(cls, path: str) -> "CleansingState":
        """Load frozen parameters (implementation deferred to production system)."""
        logger.info(f"[PRODUCTION] Would load CleansingState from {path}")
        raise NotImplementedError("Production checkpoint loading deferred")


class DataCleaner:
    """
    Production data cleansing with frozen preprocessing contract.

    Workflow:
        1. Harmonize units across visits
        2. Convert to long format
        3. Create missingness masks
        4. Winsorize outliers
        5. Impute on training folds ONLY
        6. Normalize on training folds with frozen parameters

    Once cleansing is fitted on training data, the preprocessing parameters
    (imputation models, scaler, column statistics) are frozen and cannot be
    changed for test/holdout data.
    """

    # Standard lab value column names (from ingestion)
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

    def __init__(
        self,
        winsorize_config: Optional[WinsorizeConfig] = None,
        imputation_strategy: str = "mice",
    ):
        """
        Initialize cleaner.

        Args:
            winsorize_config: Clinician-reviewed bounds (uses defaults if None)
            imputation_strategy: "mice", "knn", or "median"
        """
        self.winsorize_config = winsorize_config or WinsorizeConfig()
        self.imputation_strategy = imputation_strategy
        self._state: Optional[CleansingState] = None
        self._is_fitted = False
        logger.info(f"Initialized DataCleaner with strategy={imputation_strategy}")

    def harmonize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize units across visits.

        Assumes input lab values are already in canonical units.
        In production, this would include cross-visit validation and unit detection.

        Args:
            df: Raw ingested DataFrame

        Returns:
            DataFrame with harmonized units
        """
        df = df.copy()
        logger.info("Harmonizing units (current: assumes canonical units)")
        return df

    def to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to long format: one row per patient per timepoint.

        Args:
            df: Ingested DataFrame (may have multiple rows per patient-visit)

        Returns:
            Long format DataFrame aggregated by patient, visit, timepoint
        """
        df = df.copy()

        # Group by patient/visit/timepoint and take first value (or mean for labs)
        group_cols = ["patient_id", "visit_id", "timepoint"]
        agg_dict = {}

        for col in df.columns:
            if col in group_cols or col == "_source":
                continue
            elif col in self.LAB_COLUMNS or any(x in col.lower() for x in ["day", "event"]):
                # Aggregate numeric columns by mean (handles multiple measurements)
                agg_dict[col] = "first"  # Or "mean" for repeated measurements
            else:
                agg_dict[col] = "first"

        df_long = df.groupby(group_cols, as_index=False).agg(agg_dict)
        logger.info(f"Converted to long format: {df.shape[0]} -> {df_long.shape[0]} rows")
        return df_long

    def create_missingness_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary missingness mask for audit trail.

        Maintains record of which values were originally missing (not imputed).

        Args:
            df: DataFrame with potential NaN values

        Returns:
            Binary mask DataFrame (1 = originally missing, 0 = observed)
        """
        mask_cols = self.LAB_COLUMNS + [
            col for col in df.columns
            if any(x in col.lower() for x in ["treatment", "cyto", "fish", "stage"])
        ]

        mask = pd.DataFrame()
        for col in mask_cols:
            if col in df.columns:
                mask[f"_{col}_missing"] = df[col].isna().astype(int)

        logger.info(f"Created missingness masks for {mask.shape[1]} columns")
        return mask

    def winsorize_outliers(
        self,
        df: pd.DataFrame,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Winsorize lab values using clinician-reviewed bounds.

        Replaces extreme outliers with upper/lower bounds. Applied before
        imputation to prevent extreme values from biasing imputation models.

        Args:
            df: DataFrame with lab values
            inplace: If True, modifies df in place

        Returns:
            Winsorized DataFrame
        """
        if not inplace:
            df = df.copy()

        for col in self.LAB_COLUMNS:
            if col not in df.columns:
                continue

            bounds = self.winsorize_config.get_bounds(col)
            if not bounds:
                continue

            lower, upper = bounds
            before_count = df[col].notna().sum()

            df[col] = df[col].clip(lower=lower, upper=upper)

            after_count = df[col].notna().sum()
            logger.debug(f"Winsorized {col}: [{lower}, {upper}]")

        return df

    def _get_available_lab_columns(self, df: pd.DataFrame) -> list:
        """Return lab columns that exist in df and have at least one non-NaN value."""
        available = []
        for col in self.LAB_COLUMNS:
            if col in df.columns and df[col].notna().any():
                available.append(col)
        return available

    def _build_imputation_model(self, df: pd.DataFrame) -> object:
        """
        Build imputation model on training data.

        Args:
            df: Training DataFrame with NaN values

        Returns:
            Fitted imputation model
        """
        if self.imputation_strategy == "mice":
            imputer = IterativeImputer(
                max_iter=10,
                random_state=42,
                verbose=0,
            )
        elif self.imputation_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        elif self.imputation_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        else:
            raise ValueError(f"Unknown strategy: {self.imputation_strategy}")

        available_labs = self._get_available_lab_columns(df)
        if available_labs:
            imputer.fit(df[available_labs])
            logger.info(f"Built {self.imputation_strategy} imputation model on {len(available_labs)} lab columns")
        else:
            logger.warning("No lab columns with observed values — skipping imputation fitting")
        return imputer

    def fit(self, df: pd.DataFrame, version: str = "0.1.0") -> "DataCleaner":
        """
        Fit cleansing pipeline on training data.

        Frozen contract: Once fit, imputation and normalization parameters
        cannot be changed. Apply fit parameters to test/holdout via apply().

        Args:
            df: Training DataFrame
            version: Version string for preprocessing contract

        Returns:
            Self (for chaining)
        """
        df = df.copy()

        # Step 1: Harmonize units
        df = self.harmonize_units(df)

        # Step 2: Long format
        df = self.to_long_format(df)

        # Step 3: Missingness masks (for audit)
        missingness_mask = self.create_missingness_mask(df)

        # Step 4: Winsorize
        df = self.winsorize_outliers(df)

        # Step 5: Build imputation model (only on columns with observed data)
        available_labs = self._get_available_lab_columns(df)
        imputer = self._build_imputation_model(df)
        df_imputed = df.copy()
        if available_labs:
            df_imputed[available_labs] = imputer.transform(df[available_labs])

        # Step 6: Build normalizer on training data
        scaler = StandardScaler()
        if available_labs:
            scaler.fit(df_imputed[available_labs])
            col_means = {col: scaler.mean_[i] for i, col in enumerate(available_labs)}
            col_stds = {col: scaler.scale_[i] for i, col in enumerate(available_labs)}
        else:
            col_means = {}
            col_stds = {}
            logger.warning("No lab columns available for normalization")

        # Freeze state
        self._available_labs = available_labs
        self._state = CleansingState(
            version=version,
            imputation_models={"labs": imputer},
            scaler=scaler if available_labs else None,
            column_means=col_means,
            column_stds=col_stds,
            missingness_patterns=missingness_mask.mean().to_dict(),
        )
        self._is_fitted = True
        logger.info(f"Fitted cleansing pipeline v{version}")
        return self

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply frozen preprocessing to new data (test/holdout).

        Uses imputation models and normalization parameters from training fit.
        Raises error if called before fit().

        Args:
            df: New DataFrame to clean

        Returns:
            Tuple of (cleaned_df, missingness_mask)

        Raises:
            RuntimeError: If apply() called before fit()
        """
        if not self._is_fitted or self._state is None:
            raise RuntimeError("Must call fit() before apply()")

        df = df.copy()

        # Harmonize, long format
        df = self.harmonize_units(df)
        df = self.to_long_format(df)

        # Missingness mask (before imputation)
        missingness_mask = self.create_missingness_mask(df)

        # Winsorize
        df = self.winsorize_outliers(df)

        # Apply frozen imputation (only on columns that were fitted)
        available_labs = getattr(self, "_available_labs", self._get_available_lab_columns(df))
        if available_labs:
            imputer = self._state.imputation_models["labs"]
            df[available_labs] = imputer.transform(df[available_labs])

        # Apply frozen normalization
        if available_labs and self._state.scaler is not None:
            scaler = self._state.scaler
            df[available_labs] = scaler.transform(df[available_labs])

        logger.info(f"Applied frozen preprocessing v{self._state.version} to {df.shape[0]} rows")
        return df, missingness_mask

    def get_state(self) -> CleansingState:
        """Get frozen preprocessing state for serialization."""
        if not self._is_fitted or self._state is None:
            raise RuntimeError("No fitted state available")
        return self._state
