#!/usr/bin/env bash
# R1 — Full pipeline: install, provision data, run end-to-end
#
# Usage:
#   git clone https://github.com/Abhignya-Jagathpally/r1.git
#   cd r1
#   bash run.sh
#
set -euo pipefail

echo "========================================"
echo "R1 — MM Digital Twin Pipeline"
echo "========================================"

# 1. Create virtualenv if needed
if [ ! -d "bin" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv .
    source bin/activate
else
    echo "[1/4] Virtual environment exists, activating..."
    source bin/activate
fi

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install -q -r requirements.txt

# 3. Provision data (GDC open metadata — no auth needed)
echo "[3/4] Provisioning CoMMpass data from GDC..."
python main.py --provision-data --dry-run

# 4. Run full pipeline
echo "[4/4] Running full pipeline..."
python main.py --provision-data --verbose

echo ""
echo "========================================"
echo "Pipeline complete. Results in results/"
echo "========================================"
