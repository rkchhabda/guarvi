#!/usr/bin/env bash
# Render build script - runs before the image is deployed.
# Idempotent: safe to re-run on every deploy.
set -euo pipefail

echo "[render-build] python:  $(python --version 2>&1)"
echo "[render-build] pip:     $(pip --version)"
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

echo "[render-build] done."
