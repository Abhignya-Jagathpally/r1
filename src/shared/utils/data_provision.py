"""
Data Provisioning — Download CoMMpass clinical data.

Strategy:
  1. Try MMRF AWS Open Data (flat files — requires auth in practice)
  2. Fall back to GDC Cases API (open metadata only — demographics, vital status, diagnosis). Lab values and longitudinal data require MMRF Researcher Gateway.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Public CoMMpass S3 bucket (Registry of Open Data on AWS)
COMMPASS_BUCKET = "s3://mmrf-commpass"
COMMPASS_BASE = "https://mmrf-commpass.s3.amazonaws.com"

# Key flat files needed for the pipeline
COMMPASS_FILES = {
    "MMRF_CoMMpass_IA20_PER_PATIENT.csv": "Per-patient baseline clinical data",
    "MMRF_CoMMpass_IA20_PER_PATIENT_VISIT.csv": "Longitudinal visit-level data",
    "MMRF_CoMMpass_IA20_STAND_ALONE_TRTRESP.csv": "Treatment response",
    "MMRF_CoMMpass_IA20_STAND_ALONE_SURVIVAL.csv": "Survival endpoints",
}


def check_data_available(data_dir: Path) -> bool:
    """Check if any CSV/TSV files exist in data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return False
    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
    return len(files) > 0


def provision_data(data_dir: Path, method: str = "curl") -> bool:
    """
    Download CoMMpass data to data_dir.

    Tries MMRF AWS first, falls back to GDC API (always public).

    Returns:
        True if data is available after provisioning.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try MMRF AWS Open Data
    logger.info("Attempting MMRF AWS Open Data download...")
    if _try_mmrf_aws(data_dir, method):
        return True

    # Strategy 2: GDC Cases API (open metadata, no molecular/lab data)
    logger.info("MMRF AWS not accessible. Falling back to GDC API (public)...")
    return _try_gdc_download(data_dir)


def _try_mmrf_aws(data_dir: Path, method: str) -> bool:
    """Attempt download from MMRF AWS S3."""
    downloaded = 0
    for filename, description in COMMPASS_FILES.items():
        target = data_dir / filename
        if target.exists() and target.stat().st_size > 100:
            downloaded += 1
            continue

        url = f"{COMMPASS_BASE}/IA20a/{filename}"
        try:
            if method == "curl":
                result = subprocess.run(
                    ["curl", "-fSL", "--retry", "2", "-o", str(target), url],
                    capture_output=True, text=True, timeout=60,
                )
            else:
                result = subprocess.run(
                    ["wget", "-q", "-O", str(target), url],
                    capture_output=True, text=True, timeout=60,
                )

            if result.returncode == 0 and target.exists() and target.stat().st_size > 100:
                downloaded += 1
                logger.info(f"  OK: {filename}")
            else:
                if target.exists():
                    target.unlink()
        except Exception:
            if target.exists():
                target.unlink()

    return downloaded > 0


def _try_gdc_download(data_dir: Path) -> bool:
    """Download case metadata from GDC API (open access, limited to demographics and survival endpoints)."""
    try:
        from src.shared.utils.gdc_download import download_commpass_from_gdc
        path = download_commpass_from_gdc(data_dir)
        return path.exists() and path.stat().st_size > 100
    except Exception as e:
        logger.error(f"GDC download failed: {e}")
        return False


def print_data_instructions(data_dir: Path) -> None:
    """Print instructions for obtaining CoMMpass data."""
    print()
    print("=" * 72)
    print("DATA REQUIRED — MMRF CoMMpass (IA20)")
    print("=" * 72)
    print()
    print(f"The pipeline needs CoMMpass clinical flat files in: {data_dir}/")
    print()
    print("Option 1 — GDC case metadata only (automatic, limited features):")
    print("  python main.py --provision-data")
    print("  WARNING: GDC provides demographics and survival endpoints only.")
    print("  For full pipeline capability (lab values, treatment data), use Option 2 or 3.")
    print()
    print("Option 2 — Manual download from MMRF:")
    print("  1. Visit https://research.themmrf.org/")
    print("  2. Register for a free MMRF account")
    print("  3. Navigate to CoMMpass > IA20 > Flat Files")
    print("  4. Download the following files:")
    for fname, desc in COMMPASS_FILES.items():
        print(f"     - {fname}  ({desc})")
    print(f"  5. Place them in: {data_dir}/")
    print()
    print("Option 3 — AWS CLI:")
    print(f"  aws s3 cp --no-sign-request {COMMPASS_BUCKET}/IA20a/ {data_dir}/ --recursive")
    print()
    print("=" * 72)
