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

    # CoMMpass-specific column name mappings
    COMMPASS_COLUMN_MAP = {
        # Endpoints
        "ttcpfs": "pfs_days",
        "censpfs": "pfs_event",
        "ttcos": "os_days",
        "censos": "os_event",
        # Demographics / staging from PER_PATIENT
        "D_PT_age": "age_at_diagnosis",
        "D_PT_gender": "gender",
        "D_PT_race": "race",
        "D_PT_iss": "iss_stage",
        "D_PT_riss": "r_iss_stage",
        "D_PT_del17p": "fish_del17p",
        "D_PT_t_4_14": "fish_t_4_14",
        "D_PT_t_14_16": "fish_t_14_16",
        "D_PT_gain1q": "fish_gain1q",
        "D_PT_del13": "fish_del13",
        "D_PT_high_risk": "high_risk",
        "D_PT_transplant": "prior_transplant",
        "D_PT_treatment_lines": "treatment_line",
        # Labs from PER_PATIENT_VISIT
        "D_LAB_serum_m_protein": "serum_m_protein_g_dl",
        "D_LAB_serum_flc_kappa": "free_light_chain_kappa_mg_l",
        "D_LAB_serum_flc_lambda": "free_light_chain_lambda_mg_l",
        "D_LAB_cbc_hemoglobin": "hemoglobin_g_dl",
        "D_LAB_chem_calcium": "calcium_mg_dl",
        "D_LAB_chem_creatinine": "creatinine_mg_dl",
        "D_LAB_chem_albumin": "albumin_g_dl",
        "D_LAB_chem_beta2_microglobulin": "beta2_microglobulin_mg_l",
        "D_LAB_chem_ldh": "ldh_u_l",
        # Visit structure
        "PUBLIC_ID": "patient_id",
        "VISIT": "visit_id",
        "VISITDY": "timepoint",
    }

    def ingest(self) -> pd.DataFrame:
        """
        Load, parse, and consolidate CoMMpass data into unified format.

        Strategy: use PER_PATIENT_VISIT as the longitudinal backbone,
        then merge in patient-level demographics, genetics, and endpoints
        from PER_PATIENT and SURVIVAL files.

        Returns:
            DataFrame with columns:
                - patient_id, visit_id, timepoint
                - All lab values (LAB_COLUMNS keys)
                - All treatment flags (TREATMENT_COLUMNS keys)
                - All genetic/staging vars (GENETIC_COLUMNS keys)
                - All endpoints (ENDPOINT_COLUMNS keys)
                - _source: originating filename
        """
        raw_files = self.load_raw_files()

        # ── Identify the visit-level and patient-level files ──
        visit_df = None
        patient_df = None
        survival_df = None

        for name, df in raw_files.items():
            name_lower = name.lower()
            if "per_patient_visit" in name_lower or "visit" in name_lower:
                visit_df = df
                logger.info(f"Using {name} as visit-level longitudinal backbone")
            elif "survival" in name_lower:
                survival_df = df
                logger.info(f"Using {name} as survival endpoints source")
            elif "per_patient" in name_lower or "patient" in name_lower:
                patient_df = df
                logger.info(f"Using {name} as patient-level baseline source")

        # Fall back: if no visit file, use the largest as main
        if visit_df is None:
            biggest = max(raw_files.items(), key=lambda x: x[1].shape[0])
            visit_df = biggest[1]
            logger.info(f"Fallback: using {biggest[0]} as main source")

        # ── Rename columns using CoMMpass mapping ──
        visit_df = visit_df.rename(columns=self.COMMPASS_COLUMN_MAP)

        # ── Merge patient-level data if available ──
        if patient_df is not None:
            patient_df = patient_df.rename(columns=self.COMMPASS_COLUMN_MAP)
            # Only merge columns not already in visit_df (except patient_id)
            merge_cols = ["patient_id"] + [
                c for c in patient_df.columns
                if c not in visit_df.columns and c != "patient_id"
            ]
            merge_cols = [c for c in merge_cols if c in patient_df.columns]
            visit_df = visit_df.merge(
                patient_df[merge_cols], on="patient_id", how="left",
            )
            logger.info(f"Merged patient-level data: added {len(merge_cols)-1} columns")

        # ── Merge survival endpoints if available and not already present ──
        if survival_df is not None:
            survival_df = survival_df.rename(columns=self.COMMPASS_COLUMN_MAP)
            surv_cols = ["patient_id"] + [
                c for c in survival_df.columns
                if c not in visit_df.columns and c != "patient_id"
            ]
            surv_cols = [c for c in surv_cols if c in survival_df.columns]
            if len(surv_cols) > 1:  # more than just patient_id
                visit_df = visit_df.merge(
                    survival_df[surv_cols], on="patient_id", how="left",
                )
                logger.info(f"Merged survival endpoints: added {len(surv_cols)-1} columns")

        # ── Ensure all expected columns exist ──
        for std_col in list(self.LAB_COLUMNS.keys()):
            if std_col not in visit_df.columns:
                visit_df[std_col] = np.nan
        for std_col in list(self.ENDPOINT_COLUMNS.keys()):
            if std_col not in visit_df.columns:
                visit_df[std_col] = np.nan

        # ── Ensure critical endpoint columns are numeric ──
        for col in ["pfs_days", "pfs_event", "os_days", "os_event",
                     "time_to_progression_days", "ttp_event", "relapse_event"]:
            if col in visit_df.columns:
                visit_df[col] = pd.to_numeric(visit_df[col], errors="coerce")
        # Fill event columns with 0 where NaN (censored)
        for col in ["pfs_event", "os_event", "ttp_event", "relapse_event"]:
            if col in visit_df.columns:
                visit_df[col] = visit_df[col].fillna(0).astype(int)

        # ── Ensure patient_id, visit_id, timepoint exist ──
        if "patient_id" not in visit_df.columns:
            patient_col = self._infer_patient_column(visit_df)
            if patient_col:
                visit_df = visit_df.rename(columns={patient_col: "patient_id"})
            else:
                raise ValueError("Cannot infer patient_id column")
        if "visit_id" not in visit_df.columns:
            visit_col = self._infer_visit_column(visit_df)
            if visit_col:
                visit_df = visit_df.rename(columns={visit_col: "visit_id"})
            else:
                visit_df["visit_id"] = range(len(visit_df))
        if "timepoint" not in visit_df.columns:
            tp_col = self._infer_timepoint_column(visit_df)
            if tp_col:
                visit_df = visit_df.rename(columns={tp_col: "timepoint"})
            else:
                visit_df["timepoint"] = 0

        visit_df["_source"] = "merged_compass"
        logger.info(f"Ingested {visit_df.shape[0]} records with {visit_df.shape[1]} features")
        return visit_df.reset_index(drop=True)

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
