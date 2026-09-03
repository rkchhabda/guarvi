#!/usr/bin/env python3
"""Fetch and process FII/DII flow data from NSDL.

This script downloads FII/DII daily equity flows from NSDL,
computes derived features, and saves them for integration
with the main feature pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
import pandas as pd

from market_ml.external_data import fetch_fii_dii_flows, compute_fii_dii_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_fii_dii")


def main():
    logger.info("Fetching FII/DII data from NSDL...")
    
    # Fetch data (will cache locally)
    fii_dii_df = fetch_fii_dii_flows(
        start_date="2015-01-01",
        end_date=None,  # Up to latest
        force_refresh=False,
    )
    
    if fii_dii_df.empty:
        logger.error("No FII/DII data fetched")
        return 1
    
    logger.info(f"Fetched {len(fii_dii_df)} rows of FII/DII data")
    logger.info(f"Date range: {fii_dii_df['date'].min()} to {fii_dii_df['date'].max()}")
    logger.info(f"Columns: {list(fii_dii_df.columns)}")
    
    # Compute derived features
    logger.info("Computing FII/DII derived features...")
    features_df = compute_fii_dii_features(fii_dii_df)
    
    logger.info(f"Computed {len(features_df.columns)-1} features")
    logger.info(f"Feature columns: {[c for c in features_df.columns if c != 'date']}")
    
    # Save features
    output_dir = ROOT / "data" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_path = output_dir / "fii_dii_features.parquet"
    features_df.to_parquet(features_path, index=False)
    logger.info(f"Saved FII/DII features to {features_path}")
    
    # Also save raw data
    raw_path = output_dir / "fii_dii_raw.parquet"
    fii_dii_df.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw FII/DII data to {raw_path}")
    
    # Print sample
    logger.info("\nSample of features (last 10 rows):")
    print(features_df.tail(10).to_string())
    
    # Print feature stats
    logger.info("\nFeature statistics:")
    for col in features_df.columns:
        if col != 'date':
            non_null = features_df[col].notna().sum()
            logger.info(f"  {col}: {non_null}/{len(features_df)} non-null, "
                       f"mean={features_df[col].mean():.4f}, "
                       f"std={features_df[col].std():.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())