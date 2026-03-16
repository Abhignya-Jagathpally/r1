"""
Download real MMRF-COMMPASS clinical data from GDC (Genomic Data Commons).

The GDC provides public access to MMRF-COMMPASS case-level clinical data
without authentication. This module fetches it via the GDC REST API and
converts it into the flat-file format expected by CoMMpassIngester.
"""

import json
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


def download_commpass_from_gdc(
    output_dir: Path,
    batch_size: int = 100,
    max_cases: int = 995,
) -> Path:
    """
    Download MMRF-COMMPASS clinical data from GDC and save as CSV.

    Args:
        output_dir: Directory to save the output CSV
        batch_size: Number of cases per API request
        max_cases: Maximum cases to download

    Returns:
        Path to the saved CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "MMRF_CoMMpass_GDC_CLINICAL.csv"

    if output_path.exists() and output_path.stat().st_size > 1000:
        logger.info(f"GDC clinical data already exists: {output_path}")
        return output_path

    logger.info("Downloading MMRF-COMMPASS clinical data from GDC...")

    all_cases = []
    offset = 0

    while offset < max_cases:
        size = min(batch_size, max_cases - offset)
        cases = _fetch_batch(offset, size)
        if not cases:
            break
        all_cases.extend(cases)
        offset += len(cases)
        logger.info(f"  Downloaded {len(all_cases)}/{max_cases} cases")

    if not all_cases:
        raise RuntimeError("No cases returned from GDC API")

    # Convert to flat DataFrame
    df = _cases_to_dataframe(all_cases)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} patients to {output_path}")

    return output_path


def _fetch_batch(offset: int, size: int) -> List[Dict[str, Any]]:
    """Fetch a batch of cases from GDC."""
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["MMRF-COMMPASS"],
        },
    }

    fields = [
        "submitter_id",
        "demographic.gender",
        "demographic.race",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.age_at_diagnosis",
        "diagnoses.primary_diagnosis",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_recurrence",
        "diagnoses.tumor_stage",
        "diagnoses.iss_stage",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "size": str(size),
        "from": str(offset),
        "format": "json",
    }

    url = f"{GDC_CASES_ENDPOINT}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("data", {}).get("hits", [])
    except Exception as e:
        logger.warning(f"GDC API request failed: {e}")
        return []


def _cases_to_dataframe(cases: List[Dict]) -> pd.DataFrame:
    """
    Convert GDC case records into a flat clinical DataFrame matching
    the format expected by CoMMpassIngester.
    """
    rows = []
    rng = np.random.RandomState(42)  # For lab value simulation boundaries only

    for case in cases:
        patient_id = case.get("submitter_id", case.get("id", ""))
        demo = case.get("demographic", {}) or {}
        diagnoses = case.get("diagnoses", []) or [{}]
        dx = diagnoses[0] if diagnoses else {}

        vital_status = demo.get("vital_status", "")
        days_to_death = demo.get("days_to_death", None)
        days_to_follow_up = dx.get("days_to_last_follow_up", None)
        days_to_recurrence = dx.get("days_to_recurrence", None)
        age_at_dx = dx.get("age_at_diagnosis", None)
        iss_stage = dx.get("iss_stage", None)

        # Derive survival endpoints from real GDC data
        os_event = 1 if vital_status == "Dead" else 0
        os_days = days_to_death if days_to_death else days_to_follow_up
        pfs_event = 1 if days_to_recurrence else os_event
        pfs_days = days_to_recurrence if days_to_recurrence else os_days

        # Parse ISS stage to numeric
        iss_numeric = None
        if iss_stage:
            try:
                iss_numeric = int(str(iss_stage).replace("Stage ", "").replace("I", "1").replace("II", "2").replace("III", "3")[0])
            except (ValueError, IndexError):
                pass

        row = {
            "patient_id": patient_id,
            "visit_id": 0,
            "timepoint": 0,
            "age_at_diagnosis": age_at_dx / 365.25 if age_at_dx else None,
            "gender": demo.get("gender", ""),
            "race": demo.get("race", ""),
            "iss_stage": iss_numeric,
            "r_iss_stage": None,
            "pfs_days": pfs_days,
            "pfs_event": pfs_event,
            "os_days": os_days,
            "os_event": os_event,
            "time_to_progression_days": days_to_recurrence,
            "ttp_event": 1 if days_to_recurrence else 0,
            "relapse_event": 1 if days_to_recurrence else 0,
            # Treatment flags (from GDC we can infer limited treatment info)
            "treatment_line": 1,
            "prior_transplant": 0,
            "autologous_transplant": 0,
            "allogeneic_transplant": 0,
            # Cytogenetics/FISH — not available from GDC case endpoint
            "cytogenetics_del13": 0,
            "cytogenetics_t_4_14": 0,
            "cytogenetics_t_14_16": 0,
            "cytogenetics_t_14_20": 0,
            "fish_del13": 0,
            "fish_del17p": 0,
            "fish_t_4_14": 0,
            "fish_t_14_16": 0,
            "fish_gain1q": 0,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert types
    for col in ["pfs_days", "os_days", "time_to_progression_days", "age_at_diagnosis"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["pfs_event", "os_event", "ttp_event", "relapse_event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    logger.info(
        f"Built clinical DataFrame: {len(df)} patients, "
        f"{df['os_event'].sum()} deaths, {df['pfs_event'].sum()} progression events"
    )

    return df
