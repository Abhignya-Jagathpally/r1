"""
Pipeline Orchestrator: End-to-End Data Processing for MM Digital Twin

Orchestrates the complete workflow:
    1. Data ingestion: Load CoMMpass flat files
    2. Cleansing: Harmonize units, handle missingness, impute, normalize
    3. Feature engineering: Temporal and clinical features
    4. Data splitting: Patient-level, time-aware, stratified partitioning
    5. Output: Analysis-ready Parquet files

CLI with argparse, logging, and config-driven execution.
Preprocessing pipeline is frozen after training and versioned.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .data_ingestion import CoMMpassIngester
from .cleansing import DataCleaner, WinsorizeConfig, CleansingState
from .feature_engineering import FeatureEngineer, TemporalWindowConfig
from .splits import DataSplitter, SplitConfig

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for complete pipeline execution."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Load config from JSON or use defaults.

        Args:
            config_path: Path to config.json (optional)
        """
        self.data_dir = Path("data/raw")
        self.output_dir = Path("data/processed")
        self.raw_output = "raw_ingested.parquet"
        self.cleaned_output = "cleaned.parquet"
        self.engineered_output = "engineered.parquet"
        self.train_output = "train.parquet"
        self.val_output = "val.parquet"
        self.test_output = "test.parquet"
        self.state_output = "preprocessing_state.json"

        # Cleansing config
        self.imputation_strategy = "mice"
        self.winsorize_config = WinsorizeConfig()

        # Feature engineering config
        self.temporal_windows = [90, 180, 365]  # days

        # Split config
        self.split_strategy = "stratified_group_kfold"
        self.test_size = 0.2
        self.val_size = 0.1
        self.n_splits = 5
        self.stratify_column = "pfs_event"

        if config_path and config_path.exists():
            self._load_from_json(config_path)

    def _load_from_json(self, path: Path) -> None:
        """Load config from JSON file."""
        with open(path) as f:
            cfg = json.load(f)
            for key, val in cfg.items():
                if hasattr(self, key):
                    setattr(self, key, val)

    def save(self, path: Path) -> None:
        """Save config to JSON."""
        cfg_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        with open(path, "w") as f:
            json.dump(cfg_dict, f, indent=2)


class Pipeline:
    """End-to-end MM digital twin data pipeline."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Pipeline with config")

    def run(self, mode: str = "train") -> Dict[str, Path]:
        """
        Execute complete pipeline.

        Args:
            mode: "train" (fit preprocessing) or "apply" (use frozen preprocessing)

        Returns:
            Dictionary mapping dataset names to output file paths

        Raises:
            ValueError: If mode is invalid or preprocessing state not found in apply mode
        """
        if mode not in ["train", "apply"]:
            raise ValueError(f"Unknown mode: {mode}")

        logger.info(f"Starting pipeline in {mode} mode")

        # Step 1: Data ingestion
        logger.info("=" * 60)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 60)
        raw_df = self._ingest()

        # Save raw ingested
        raw_path = self.config.output_dir / self.config.raw_output
        raw_df.to_parquet(raw_path, index=False, engine="pyarrow")
        logger.info(f"Saved raw ingested data to {raw_path}")

        # Step 2: Data cleansing
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Data Cleansing")
        logger.info("=" * 60)

        if mode == "train":
            cleaner = DataCleaner(
                winsorize_config=self.config.winsorize_config,
                imputation_strategy=self.config.imputation_strategy,
            )
            cleaned_df, missingness_mask = self._cleanse_train(raw_df, cleaner)

            # Save preprocessing state
            state = cleaner.get_state()
            state.save(str(self.config.output_dir / self.config.state_output))
            logger.info(f"Saved preprocessing state v{state.version}")
        else:
            cleaner = self._load_cleaner()
            cleaned_df, missingness_mask = self._cleanse_apply(raw_df, cleaner)

        # Save cleaned data
        cleaned_path = self.config.output_dir / self.config.cleaned_output
        cleaned_df.to_parquet(cleaned_path, index=False, engine="pyarrow")
        logger.info(f"Saved cleaned data to {cleaned_path}")

        # Step 3: Feature engineering
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 60)
        engineered_df = self._engineer(cleaned_df)

        # Save engineered features
        engineered_path = self.config.output_dir / self.config.engineered_output
        engineered_df.to_parquet(engineered_path, index=False, engine="pyarrow")
        logger.info(f"Saved engineered features to {engineered_path}")

        # Step 4: Data splitting
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Data Splitting")
        logger.info("=" * 60)
        train_df, val_df, test_df = self._split(engineered_df)

        # Save splits
        train_path = self.config.output_dir / self.config.train_output
        val_path = self.config.output_dir / self.config.val_output
        test_path = self.config.output_dir / self.config.test_output

        train_df.to_parquet(train_path, index=False, engine="pyarrow")
        val_df.to_parquet(val_path, index=False, engine="pyarrow")
        test_df.to_parquet(test_path, index=False, engine="pyarrow")

        logger.info(f"Saved train to {train_path}")
        logger.info(f"Saved val to {val_path}")
        logger.info(f"Saved test to {test_path}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total patients: {engineered_df['patient_id'].nunique()}")
        logger.info(f"Total rows: {len(engineered_df)}")
        logger.info(f"Total features: {engineered_df.shape[1]}")
        logger.info(f"Train: {len(train_df)} rows, {train_df['patient_id'].nunique()} patients")
        logger.info(f"Val: {len(val_df)} rows, {val_df['patient_id'].nunique()} patients")
        logger.info(f"Test: {len(test_df)} rows, {test_df['patient_id'].nunique()} patients")

        return {
            "raw": raw_path,
            "cleaned": cleaned_path,
            "engineered": engineered_path,
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }

    def _ingest(self) -> pd.DataFrame:
        """Run data ingestion."""
        ingester = CoMMpassIngester(self.config.data_dir)
        df = ingester.ingest()
        logger.info(f"Ingested {df.shape[0]} records with {df.shape[1]} columns")
        return df

    def _cleanse_train(
        self,
        df: pd.DataFrame,
        cleaner: DataCleaner,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run cleansing in train mode (fit and apply).

        Args:
            df: Raw ingested DataFrame
            cleaner: Initialized cleaner

        Returns:
            Tuple of (cleaned_df, missingness_mask)
        """
        cleaner.fit(df)
        cleaned_df, missingness_mask = cleaner.apply(df)
        logger.info(f"Cleansed {len(cleaned_df)} rows in train mode")
        return cleaned_df, missingness_mask

    def _cleanse_apply(
        self,
        df: pd.DataFrame,
        cleaner: DataCleaner,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run cleansing in apply mode (use frozen preprocessing).

        Args:
            df: Raw ingested DataFrame
            cleaner: Fitted cleaner with frozen state

        Returns:
            Tuple of (cleaned_df, missingness_mask)
        """
        cleaned_df, missingness_mask = cleaner.apply(df)
        logger.info(f"Cleansed {len(cleaned_df)} rows in apply mode")
        return cleaned_df, missingness_mask

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering."""
        window_config = TemporalWindowConfig(windows_days=self.config.temporal_windows)
        engineer = FeatureEngineer(window_config)
        engineered_df = engineer.engineer(df)
        logger.info(f"Engineered {engineered_df.shape[1]} total features")
        return engineered_df

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run data splitting."""
        split_config = SplitConfig(
            strategy=self.config.split_strategy,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            n_splits=self.config.n_splits,
            stratify_column=self.config.stratify_column,
        )
        splitter = DataSplitter(split_config)
        train_df, val_df, test_df = splitter.split(df)
        logger.info(f"Split into {len(train_df)}/{len(val_df)}/{len(test_df)} train/val/test rows")
        return train_df, val_df, test_df

    def _load_cleaner(self) -> DataCleaner:
        """Load frozen cleaner from saved state."""
        state_path = self.config.output_dir / self.config.state_output
        if not state_path.exists():
            raise FileNotFoundError(f"Preprocessing state not found: {state_path}")

        # In production, this would deserialize the frozen state
        logger.info(f"[PRODUCTION] Would load frozen preprocessing state from {state_path}")
        raise NotImplementedError("Production state loading deferred")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MM Digital Twin Pipeline: Data Ingestion → Cleansing → Features → Splits"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "apply"],
        default="train",
        help="Pipeline mode: train (fit preprocessing) or apply (use frozen preprocessing)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json (optional)",
    )
    parser.add_argument(
        "--imputation",
        choices=["mice", "knn", "median"],
        default="mice",
        help="Imputation strategy",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["patient_level", "time_aware", "stratified_group_kfold"],
        default="stratified_group_kfold",
        help="Data split strategy",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("MM Digital Twin Pipeline Starting")
    logger.info(f"Arguments: {args}")

    # Load or create config
    config = PipelineConfig(args.config)
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.imputation_strategy = args.imputation
    config.split_strategy = args.split_strategy

    try:
        pipeline = Pipeline(config)
        outputs = pipeline.run(mode=args.mode)

        logger.info("Pipeline succeeded")
        print("\nOutput files:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
