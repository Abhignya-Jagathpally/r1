"""
Multiple Myeloma Digital Twin Pipeline - Clinical Data Engineering Module

This package provides production-quality data ingestion, cleansing, and feature
engineering for the MMRF CoMMpass dataset, powering the MM digital twin model.

Main Components:
    - data_ingestion: Load and parse CoMMpass flat files
    - cleansing: Harmonize units, handle missingness, impute values
    - feature_engineering: Temporal features, CRAB/SLiM-CRAB, trajectory aggregations
    - splits: Patient-level, time-aware, stratified data partitioning
    - pipeline: Orchestration with CLI, logging, config management

Contract:
    Preprocessing is frozen after initial training. Changes to cleansing/feature
    engineering require explicit version bumping and retraining.
"""

__version__ = "0.1.0"
__author__ = "PhD Researcher 2 - Clinical Data Engineering"

from .data_ingestion import CoMMpassIngester
from .cleansing import DataCleaner
from .feature_engineering import FeatureEngineer
from .splits import DataSplitter

__all__ = [
    "CoMMpassIngester",
    "DataCleaner",
    "FeatureEngineer",
    "DataSplitter",
]
