"""
Data Ingestion Module: CoMMpass Flat File Loading and Parsing

Loads MMRF CoMMpass CSV/TSV files from data/raw/ and extracts clinical variables:
- Lab values: serum M-protein, free light chains, hemoglobin, calcium, creatinine,
              albumin, β2-microglobulin, LDH
- Treatment: line, transplant flags
- Genetics: cytogenetics, FISH, ISS/R-ISS staging
- Endpoints: PFS, OS, time-to-progression, relapse

Output format:
    DataFrame with patient_id, visit_id, timepoint, and all extracted variables.
    Intended for downstream cleansing and feature engineering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CoMMpassIngester:
    """
    Load and parse MMRF CoMMpass clinical data from flat files.

    Attributes:
        data_dir: Path to data/raw/ directory containing CoMMpass CSVs
        lab_mapping: Dict mapping CoMMpass column names to standardized names
        treatment_mapping: Dict mapping treatment-related columns
        genetic_mapping: Dict mapping cytogenetic and FISH columns
    """

    # Standardized lab value column names
    LAB_COLUMNS = {
        "serum_m_protein_g_dl": "SERUM_M_PROTEIN",
        "free_light_chain_kappa_mg_l": "FLC_KAPPA",
        "free_light_chain_lambda_mg_l": "FLC_LAMBDA",
        "free_light_chain_ratio": "FLC_RATIO",
        "hemoglobin_g_dl": "HEMOGLOBIN",
        "calcium_mg_dl": "CALCIUM",
        "creatinine_mg_dl": "CREATININE",
        "albumin_g_dl": "ALBUMIN",
        "beta2_microglobulin_mg_l": "BETA2_MICROGLOBULIN",
        "ldh_u_l": "LDH",
    }

    # Treatment and transplant columns
    TREATMENT_COLUMNS = {
        "treatment_line": "TREATMENT_LINE",
        "prior_transplant": "PRIOR_TRANSPLANT",
        "autologous_transplant": "AUTOLOGOUS_TRANSPLANT",
        "allogeneic_transplant": "ALLOGENEIC_TRANSPLANT",
    }

    # Genetic/Staging columns
    GENETIC_COLUMNS = {
        "cytogenetics_del13": "CYTO_DEL13",
        "cytogenetics_t_4_14": "CYTO_T414",
        "cytogenetics_t_14_16": "CYTO_T1416",
        "cytogenetics_t_14_20": "CYTO_T1420",
        "fish_del13": "FISH_DEL13",
        "fish_del17p": "FISH_DEL17P",
        "fish_t_4_14": "FISH_T414",
        "fish_t_14_16": "FISH_T1416",
        "fish_gain1q": "FISH_GAIN1Q",
        "iss_stage": "ISS_STAGE",
        "r_iss_stage": "R_ISS_STAGE",
    }

    # Endpoint columns
    ENDPOINT_COLUMNS = {
        "pfs_days": "PFS_DAYS",
        "pfs_event": "PFS_EVENT",
        "os_days": "OS_DAYS",
        "os_event": "OS_EVENT",
        "time_to_progression_days": "TTP_DAYS",
        "ttp_event": "TTP_EVENT",
        "relapse_event": "RELAPSE_EVENT",
    }

    def __init__(self, data_dir: Path | str):
        """
        Initialize ingester with data directory.

        Args:
            data_dir: Path to data/raw/ containing CoMMpass files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        logger.info(f"Initialized CoMMpassIngester with data_dir={self.data_dir}")

    def load_raw_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV/TSV files from data directory.

        Returns:
            Dictionary mapping filename stem to DataFrame

        Raises:
            FileNotFoundError: If no CSV/TSV files found
        """
        files = list(self.data_dir.glob("*.csv")) + list(self.data_dir.glob("*.tsv"))

        if not files:
            raise FileNotFoundError(f"No CSV/TSV files found in {self.data_dir}")

        loaded = {}
        for fpath in files:
            try:
                sep = "\t" if fpath.suffix == ".tsv" else ","
                df = pd.read_csv(fpath, sep=sep, low_memory=False)
                loaded[fpath.stem] = df
                logger.info(f"Loaded {fpath.name}: {df.shape[0]} rows, {df.shape[1]} cols")
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")
                raise

        return loaded

    def _extract_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lab values, handling multiple naming conventions.

        Attempts fuzzy matching on common patterns if exact column names absent.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with standardized lab columns (NaN if not found)
        """
        labs = pd.DataFrame()

        for std_col, alt_name in self.LAB_COLUMNS.items():
            # Try exact match first
            candidates = [col for col in df.columns if std_col.upper() in col.upper()]

            if not candidates:
                # Fuzzy match on alt_name
                candidates = [col for col in df.columns if alt_name.upper() in col.upper()]

            if candidates:
                labs[std_col] = pd.to_numeric(df[candidates[0]], errors="coerce")
            else:
                labs[std_col] = np.nan

        return labs

    def _extract_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract treatment and transplant flags.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with standardized treatment columns
        """
        treatment = pd.DataFrame()

        for std_col, alt_name in self.TREATMENT_COLUMNS.items():
            candidates = [col for col in df.columns
                         if std_col.upper() in col.upper() or alt_name.upper() in col.upper()]

            if candidates:
                # Binarize if categorical
                val = df[candidates[0]]
                treatment[std_col] = (val.astype(str).str.upper().isin(["YES", "Y", "1", "TRUE"]))
            else:
                treatment[std_col] = False

        return treatment

    def _extract_genetics_staging(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cytogenetics, FISH, and ISS/R-ISS staging.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with standardized genetic/staging columns
        """
        genetics = pd.DataFrame()

        for std_col, alt_name in self.GENETIC_COLUMNS.items():
            candidates = [col for col in df.columns
                         if std_col.upper() in col.upper() or alt_name.upper() in col.upper()]

            if candidates:
                val = df[candidates[0]]
                if "stage" in std_col.lower():
                    # Keep staging as numeric
                    genetics[std_col] = pd.to_numeric(val, errors="coerce")
                else:
                    # Binarize abnormalities
                    genetics[std_col] = (val.astype(str).str.upper().isin(["YES", "Y", "1", "TRUE", "PRESENT"]))
            else:
                if "stage" in std_col.lower():
                    genetics[std_col] = np.nan
                else:
                    genetics[std_col] = False

        return genetics

    def _extract_endpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract survival and progression endpoints.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with standardized endpoint columns
        """
        endpoints = pd.DataFrame()

        for std_col, alt_name in self.ENDPOINT_COLUMNS.items():
            candidates = [col for col in df.columns
                         if std_col.upper() in col.upper() or alt_name.upper() in col.upper()]

            if candidates:
                val = df[candidates[0]]
                if "days" in std_col.lower():
                    endpoints[std_col] = pd.to_numeric(val, errors="coerce")
                else:
                    endpoints[std_col] = (val.astype(str).str.upper().isin(["YES", "Y", "1", "TRUE"]))
            else:
                if "days" in std_col.lower():
                    endpoints[std_col] = np.nan
                else:
                    endpoints[std_col] = False

        return endpoints

    def ingest(self) -> pd.DataFrame:
        """
        Load, parse, and consolidate CoMMpass data into unified format.

        Process:
            1. Load all CSV/TSV files
            2. Identify patient and visit identifiers
            3. Extract lab, treatment, genetic, and endpoint variables
            4. Consolidate into single DataFrame

        Returns:
            DataFrame with columns:
                - patient_id, visit_id, timepoint
                - All lab values (LAB_COLUMNS keys)
                - All treatment flags (TREATMENT_COLUMNS keys)
                - All genetic/staging vars (GENETIC_COLUMNS keys)
                - All endpoints (ENDPOINT_COLUMNS keys)
                - _source: originating filename

        Raises:
            ValueError: If patient_id or visit structure cannot be inferred
        """
        raw_files = self.load_raw_files()

        # Look for main clinical data file (heuristic: largest file or matches pattern)
        main_dfs = [
            (name, df) for name, df in raw_files.items()
            if any(pat in name.lower() for pat in ["clinical", "baseline", "patient", "main"])
        ]

        if not main_dfs:
            main_dfs = sorted(raw_files.items(), key=lambda x: x[1].shape[0], reverse=True)[:1]

        if not main_dfs:
            raise ValueError("No suitable clinical data file found")

        source_name, main_df = main_dfs[0]
        logger.info(f"Using {source_name} as primary clinical data source")

        # Infer patient and visit IDs
        patient_col = self._infer_patient_column(main_df)
        visit_col = self._infer_visit_column(main_df)
        timepoint_col = self._infer_timepoint_column(main_df)

        if not patient_col:
            raise ValueError("Cannot infer patient_id column")

        result = pd.DataFrame({
            "patient_id": main_df[patient_col],
            "visit_id": main_df[visit_col] if visit_col else range(len(main_df)),
            "timepoint": main_df[timepoint_col] if timepoint_col else 0,
        })

        # Extract all feature groups
        labs = self._extract_labs(main_df)
        treatment = self._extract_treatment(main_df)
        genetics = self._extract_genetics_staging(main_df)
        endpoints = self._extract_endpoints(main_df)

        result = pd.concat([result, labs, treatment, genetics, endpoints], axis=1)
        result["_source"] = source_name

        logger.info(f"Ingested {result.shape[0]} records with {result.shape[1]} features")
        return result.reset_index(drop=True)

    @staticmethod
    def _infer_patient_column(df: pd.DataFrame) -> Optional[str]:
        """Infer patient identifier column."""
        patterns = ["patient", "subject", "pid", "id"]
        for pattern in patterns:
            matches = [col for col in df.columns if pattern.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _infer_visit_column(df: pd.DataFrame) -> Optional[str]:
        """Infer visit identifier column."""
        patterns = ["visit", "encounter", "appointment", "vid"]
        for pattern in patterns:
            matches = [col for col in df.columns if pattern.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _infer_timepoint_column(df: pd.DataFrame) -> Optional[str]:
        """Infer timepoint/date column."""
        patterns = ["date", "time", "day", "daysinclusionjection"]
        for pattern in patterns:
            matches = [col for col in df.columns if pattern.lower() in col.lower()]
            if matches:
                return matches[0]
        return None
