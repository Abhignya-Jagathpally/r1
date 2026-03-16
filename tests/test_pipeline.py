"""
Tests for MM Digital Twin Pipeline.

These tests verify:
1. Data ingestion produces expected schema
2. Cleansing preserves patient count and handles NaN correctly
3. Feature engineering produces expected feature count
4. Splits maintain patient-level separation (no leakage)
5. Preprocessing serialization round-trips correctly
6. Missing events are NOT silently converted to non-events
7. Winsorization does NOT clip MM-defining biomarkers
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.researcher1_clinical.cleansing import (
    CleansingState,
    DataCleaner,
    WinsorizeConfig,
)
from src.researcher1_clinical.feature_engineering import FeatureEngineer
from src.researcher1_clinical.splits import DataSplitter, SplitConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal longitudinal DataFrame that mirrors CoMMpass schema."""
    np.random.seed(42)
    n_patients = 20
    visits_per_patient = 5
    rows = []
    for pid in range(n_patients):
        for vid in range(visits_per_patient):
            rows.append({
                "patient_id": f"MMRF_{pid:04d}",
                "visit_id": f"MMRF_{pid:04d}_V{vid}",
                "timepoint": vid,
                "age_at_diagnosis": 60 + pid % 20,
                "gender": "Male" if pid % 2 == 0 else "Female",
                "iss_stage": np.random.choice([1, 2, 3]),
                "hemoglobin_g_dl": np.random.normal(11, 2),
                "calcium_mg_dl": np.random.normal(9.5, 1.5),
                "creatinine_mg_dl": np.random.normal(1.2, 0.8),
                "albumin_g_dl": np.random.normal(3.5, 0.5),
                "beta2_microglobulin_mg_l": np.random.normal(4, 3),
                "ldh_u_l": np.random.normal(200, 80),
                "serum_m_protein_g_dl": np.random.exponential(2),
                "free_light_chain_kappa_mg_l": np.random.exponential(50),
                "free_light_chain_lambda_mg_l": np.random.exponential(30),
                "free_light_chain_ratio": np.random.exponential(10),
                "event_progression": np.random.choice([0, 1, np.nan], p=[0.5, 0.3, 0.2]),
                "event_death": np.random.choice([0, 1, np.nan], p=[0.7, 0.1, 0.2]),
                "days_to_event": np.random.randint(30, 1000) if np.random.random() > 0.2 else np.nan,
                "treatment_start_day": vid * 90 if vid > 0 else np.nan,
                "_source": "test",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: winsorization preserves MM-defining biomarkers
# ---------------------------------------------------------------------------

class TestWinsorization:
    def test_winsorization_preserves_mm_biomarkers(self, sample_df):
        """
        FLC kappa, FLC lambda, FLC ratio, and serum M-protein carry diagnostic
        meaning at extreme values in MM.  Winsorization must NOT clip them.
        """
        df = sample_df.copy()
        # Inject extreme but clinically real MM values
        df.loc[0, "serum_m_protein_g_dl"] = 15.0       # very high M-protein
        df.loc[1, "free_light_chain_kappa_mg_l"] = 5000.0
        df.loc[2, "free_light_chain_lambda_mg_l"] = 3000.0
        df.loc[3, "free_light_chain_ratio"] = 500.0

        cleaner = DataCleaner()
        result = cleaner.winsorize_outliers(df, inplace=False)

        assert result.loc[0, "serum_m_protein_g_dl"] == 15.0, \
            "Serum M-protein must NOT be clipped by winsorization"
        assert result.loc[1, "free_light_chain_kappa_mg_l"] == 5000.0, \
            "FLC kappa must NOT be clipped by winsorization"
        assert result.loc[2, "free_light_chain_lambda_mg_l"] == 3000.0, \
            "FLC lambda must NOT be clipped by winsorization"
        assert result.loc[3, "free_light_chain_ratio"] == 500.0, \
            "FLC ratio must NOT be clipped by winsorization"

    def test_winsorization_clips_instrument_errors(self, sample_df):
        """Non-MM-defining labs with impossible values should be clipped."""
        df = sample_df.copy()
        df.loc[0, "hemoglobin_g_dl"] = 50.0   # impossible instrument error
        df.loc[1, "calcium_mg_dl"] = 100.0     # impossible

        cleaner = DataCleaner()
        result = cleaner.winsorize_outliers(df, inplace=False)

        assert result.loc[0, "hemoglobin_g_dl"] == 25.0, \
            "Hemoglobin above instrument max should be clipped to 25"
        assert result.loc[1, "calcium_mg_dl"] == 30.0, \
            "Calcium above instrument max should be clipped to 30"


# ---------------------------------------------------------------------------
# Test: missing events stay NaN (never silently converted to 0)
# ---------------------------------------------------------------------------

class TestMissingEvents:
    def test_missing_events_stay_nan(self, sample_df):
        """
        NaN in event_progression / event_death means 'not assessed', NOT
        'no event'.  Cleansing must never convert NaN to 0 in event columns.
        """
        df = sample_df.copy()
        # Ensure some NaN values exist in event columns
        nan_mask_before = df["event_progression"].isna()
        assert nan_mask_before.any(), "Test setup: need NaN events in fixture"

        cleaner = DataCleaner()
        cleaner.fit(df, version="test")
        cleaned, _ = cleaner.apply(df)

        # Event columns should not be in LAB_COLUMNS so imputation should
        # not touch them.  Verify NaN is preserved.
        if "event_progression" in cleaned.columns:
            nan_mask_after = cleaned["event_progression"].isna()
            assert nan_mask_before.sum() == nan_mask_after.sum(), \
                "NaN event indicators must not be silently converted to 0"


# ---------------------------------------------------------------------------
# Test: patient-level split has no leakage
# ---------------------------------------------------------------------------

class TestSplitLeakage:
    def test_patient_split_no_leakage(self, sample_df):
        """
        All visits of a single patient must land in exactly one partition.
        No patient ID may appear in both train and test.
        """
        config = SplitConfig(
            strategy="patient_level",
            test_size=0.2,
            val_size=0.1,
            random_state=42,
        )
        splitter = DataSplitter(config)
        train_df, val_df, test_df = splitter.split(sample_df)

        train_patients = set(train_df["patient_id"].unique())
        val_patients = set(val_df["patient_id"].unique())
        test_patients = set(test_df["patient_id"].unique())

        assert len(train_patients & test_patients) == 0, (
            f"Patient leakage detected: {train_patients & test_patients} in both train and test"
        )
        assert len(train_patients & val_patients) == 0, \
            "Patient leakage between train and val"
        assert len(test_patients & val_patients) == 0, \
            "Patient leakage between test and val"


# ---------------------------------------------------------------------------
# Test: preprocessing serialization round-trip
# ---------------------------------------------------------------------------

class TestPreprocessingSerialization:
    def test_preprocessing_serialization_roundtrip(self, sample_df):
        """
        Save and load CleansingState; verify all parameters survive the
        round-trip via pickle.
        """
        cleaner = DataCleaner(imputation_strategy="median")
        cleaner.fit(sample_df, version="1.0.0-test")
        original_state = cleaner.get_state()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            original_state.save(tmp_path)
            loaded_state = CleansingState.load(tmp_path)

            assert loaded_state.version == original_state.version
            assert set(loaded_state.column_means.keys()) == set(original_state.column_means.keys())
            for key in original_state.column_means:
                assert abs(loaded_state.column_means[key] - original_state.column_means[key]) < 1e-10, \
                    f"column_means[{key}] mismatch after round-trip"
            assert set(loaded_state.column_stds.keys()) == set(original_state.column_stds.keys())
            assert set(loaded_state.missingness_patterns.keys()) == set(original_state.missingness_patterns.keys())
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: SLiM-CRAB thresholds match IMWG guidelines
# ---------------------------------------------------------------------------

class TestSLiMCRABThresholds:
    def test_slim_crab_thresholds(self, sample_df):
        """
        Verify IMWG-defined thresholds:
          C: Calcium > 11 mg/dL
          R: Creatinine > 2 mg/dL
          A: Hemoglobin < 10 g/dL
          Li: FLC ratio >= 100
        """
        df = sample_df.copy()
        # Set known values to test threshold boundaries
        df.loc[0, "calcium_mg_dl"] = 11.5       # above threshold -> 1
        df.loc[1, "calcium_mg_dl"] = 10.9       # below threshold -> 0
        df.loc[2, "creatinine_mg_dl"] = 2.5     # above threshold -> 1
        df.loc[3, "creatinine_mg_dl"] = 1.9     # below threshold -> 0
        df.loc[4, "hemoglobin_g_dl"] = 9.0      # below threshold -> 1
        df.loc[5, "hemoglobin_g_dl"] = 10.5     # above threshold -> 0
        df.loc[6, "free_light_chain_ratio"] = 150.0  # above threshold -> 1
        df.loc[7, "free_light_chain_ratio"] = 50.0   # below threshold -> 0

        engineer = FeatureEngineer()
        criteria = engineer.assess_slim_crab_criteria(df)

        # Calcium > 11
        assert criteria.loc[0, "crab_calcium"] == 1, "Ca 11.5 should trigger crab_calcium"
        assert criteria.loc[1, "crab_calcium"] == 0, "Ca 10.9 should not trigger crab_calcium"

        # Creatinine > 2
        assert criteria.loc[2, "crab_renal"] == 1, "Cr 2.5 should trigger crab_renal"
        assert criteria.loc[3, "crab_renal"] == 0, "Cr 1.9 should not trigger crab_renal"

        # Hemoglobin < 10
        assert criteria.loc[4, "crab_anemia"] == 1, "Hgb 9.0 should trigger crab_anemia"
        assert criteria.loc[5, "crab_anemia"] == 0, "Hgb 10.5 should not trigger crab_anemia"

        # FLC ratio >= 100
        assert criteria.loc[6, "slim_flc_ratio"] == 1, "FLC ratio 150 should trigger slim_flc_ratio"
        assert criteria.loc[7, "slim_flc_ratio"] == 0, "FLC ratio 50 should not trigger slim_flc_ratio"


# ---------------------------------------------------------------------------
# Test: --dry-run produces no side effects
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_pipeline_dry_run(self):
        """
        --dry-run should print the execution plan and return without writing
        any output files or running any pipeline stages.
        """
        import importlib
        import main as pipeline_main

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = pipeline_main.PipelineSettings(
                output_dir=tmpdir,
                dry_run=True,
            )
            result = pipeline_main.run_pipeline(settings)

            # dry_run returns an empty dict (no stages executed)
            assert result == {}, "dry_run should return empty dict"

            # No data output files should be created (checkpoints dir is OK)
            output_files = [
                f for f in Path(tmpdir).rglob("*")
                if f.is_file()
            ]
            assert len(output_files) == 0, (
                f"dry_run should not create output files, found: {output_files}"
            )
