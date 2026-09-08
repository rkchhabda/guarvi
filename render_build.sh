#!/usr/bin/env bash
# Render build script - verifies pre-built artifacts exist.
# ML steps (build_features, build_signal, scoring) run LOCALLY before push.
set -euo pipefail

echo "[render-build] python:  $(python --version 2>&1)"
echo "[render-build] pip:     $(command -v pip || true)"
echo "[render-build] cwd:     $(pwd)"

# Make sure runtime directories exist
mkdir -p data/live/nse reports

# Verify pre-built artifacts (built locally, committed to repo)
if [ -f "data/processed/nifty100_ohlcv.parquet" ]; then
    echo "[render-build] OK  data/processed/nifty100_ohlcv.parquet ($(du -h data/processed/nifty100_ohlcv.parquet | cut -f1))"
else
    echo "[render-build] WARN  data/processed/nifty100_ohlcv.parquet missing - quotes/deep-dive will be empty"
fi

if [ -f "data/processed/nifty100_features.parquet" ]; then
    echo "[render-build] OK  data/processed/nifty100_features.parquet ($(du -h data/processed/nifty100_features.parquet | cut -f1))"
else
    echo "[render-build] WARN  data/processed/nifty100_features.parquet missing - scoring endpoints will be empty"
fi

if [ -f "reports/live_signal.json" ]; then
    echo "[render-build] OK  reports/live_signal.json ($(du -h reports/live_signal.json | cut -f1))"
else
    echo "[render-build] WARN  reports/live_signal.json missing - KPI cards / top-5 / bottom-5 will be empty"
fi

if [ -f "reports/precomputed_scores.json" ]; then
    echo "[render-build] OK  reports/precomputed_scores.json ($(du -h reports/precomputed_scores.json | cut -f1))"
else
    echo "[render-build] INFO  reports/precomputed_scores.json not found - will compute at runtime (slower cold-start)"
fi

if [ -f "web/static/index.html" ]; then
    echo "[render-build] OK  web/static/index.html ($(du -h web/static/index.html | cut -f1))"
else
    echo "[render-build] FAIL  web/static/index.html missing - aborting build"
    exit 1
fi

echo "[render-build] done."