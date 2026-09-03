"""Build the processed dataset from raw OHLCV data.

Usage:
    python scripts/build_dataset.py [--ticker NIFTY50] [--synthetic]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_ml.data import load_ohlcv, make_synthetic_ohlcv
from market_ml.features import make_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NIFTY50", help="Ticker/file stem under data/raw/")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate deterministic synthetic data instead of reading data/raw/",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Label horizon in trading days")
    args = parser.parse_args()

    if args.synthetic:
        df = make_synthetic_ohlcv()
        print("Using synthetic OHLCV data (750 trading days)")
    else:
        raw_path = ROOT / "data" / "raw" / f"{args.ticker}.csv"
        df = load_ohlcv(raw_path)
        print(f"Loaded {len(df)} rows from {raw_path}")

    X, y = make_dataset(df, horizon=args.horizon)
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    X_path = out_dir / f"{args.ticker}_features.csv"
    y_path = out_dir / f"{args.ticker}_labels.csv"
    X.to_csv(X_path)
    y.to_frame().to_csv(y_path)
    print(f"Saved features: {X.shape} -> {X_path}")
    print(f"Saved labels:   {y.shape[0]} rows -> {y_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
