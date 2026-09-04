#!/usr/bin/env bash
# Render build script - runs before the image is deployed.
# Idempotent: safe to re-run on every deploy.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || echo python)}"

# Render sometimes invokes this script before the dependency install step.
# Ensure the packages needed to build the parquet cache exist before we import
# numpy/pandas for feature generation.
if ! "$PYTHON_BIN" -c "import numpy, pandas, pyarrow" >/dev/null 2>&1; then
    echo "[render-build] Installing Python dependencies for build-time data generation..."
    if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        echo "[render-build] Bootstrapping pip for this Python runtime..."
        "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
    fi
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
        "$PYTHON_BIN" -m pip install -r requirements-render.txt
    elif command -v pip >/dev/null 2>&1; then
        pip install --upgrade pip >/dev/null
        pip install -r requirements-render.txt
    else
        echo "[render-build] ERROR: no pip available for this environment" >&2
        exit 1
    fi
fi

echo "[render-build] python:  $($PYTHON_BIN --version 2>&1)"
echo "[render-build] pip:     $(command -v pip || true)"
echo "[render-build] cwd:     $(pwd)"

# Make sure the runtime directories exist with safe permissions.
mkdir -p data/live/nse
mkdir -p reports

# Sanity-check the artefacts the dashboard reads. If they're missing the
# service will still start (it has a graceful "not built" fallback) but
# the UI will be empty. Fail loudly here so it's obvious in the build log.
if [ -f "data/processed/nifty100_ohlcv.parquet" ]; then
    echo "[render-build] OK  data/processed/nifty100_ohlcv.parquet ($(du -h data/processed/nifty100_ohlcv.parquet | cut -f1))"
else
    echo "[render-build] WARN  data/processed/nifty100_ohlcv.parquet missing - quotes/deep-dive will be empty"
fi

if [ -f "reports/live_signal.json" ]; then
    echo "[render-build] OK  reports/live_signal.json ($(du -h reports/live_signal.json | cut -f1))"
else
    echo "[render-build] WARN  reports/live_signal.json missing - KPI cards / top-5 / bottom-5 will be empty"
fi

if [ -f "web/static/index.html" ]; then
    echo "[render-build] OK  web/static/index.html ($(du -h web/static/index.html | cut -f1))"
else
    echo "[render-build] FAIL  web/static/index.html missing - aborting build"
    exit 1
fi

# Pre-build the features parquet from OHLCV so cold-start is fast
# This avoids the 90s runtime rebuild that causes 502s on free tier
if [ -f "data/processed/nifty100_ohlcv.parquet" ] && [ ! -f "data/processed/nifty100_features.parquet" ]; then
    echo "[render-build] Building nifty100_features.parquet from OHLCV..."
    "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '.')
from src.market_ml.feature_engineering import build_full_dataset
from pathlib import Path
OHLCV_PATH = Path('data/processed/nifty100_ohlcv.parquet')
FEATURES_PATH = Path('data/processed/nifty100_features.parquet')
print('Building features from OHLCV...')
df = build_full_dataset(OHLCV_PATH)
FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(FEATURES_PATH, index=False)
print(f'Built {FEATURES_PATH} with shape {df.shape}')
"
    echo "[render-build] Features parquet built successfully"
elif [ -f "data/processed/nifty100_features.parquet" ]; then
    echo "[render-build] OK  data/processed/nifty100_features.parquet ($(du -h data/processed/nifty100_features.parquet | cut -f1))"
else
    echo "[render-build] WARN  data/processed/nifty100_ohlcv.parquet missing - features will be empty"
fi

# Pre-compute scores at build time to avoid loading 205 MB parquet at runtime
# This saves the scores as a small JSON file that loads instantly
if [ -f "data/processed/nifty100_features.parquet" ] && [ -f "reports/live_signal.json" ]; then
    echo "[render-build] Pre-computing scores for runtime..."
    "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '.')
from src.market_ml.scoring import compute_all_scores
import json
from pathlib import Path
result = compute_all_scores()
CACHE_PATH = Path('reports/precomputed_scores.json')
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_PATH.write_text(json.dumps(result))
print(f'Pre-computed scores cached to {CACHE_PATH} ({len(result.get(\"scores\", []))} symbols)')
"
    echo "[render-build] Scores pre-computed successfully"
else
    echo "[render-build] WARN  Cannot pre-compute scores - missing features or signal"
fi

echo "[render-build] done."
