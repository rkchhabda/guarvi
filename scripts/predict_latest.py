"""Predict next-day direction using the latest saved model.

Usage:
    python scripts/predict_latest.py [--ticker NIFTY50] [--synthetic]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_ml.data import load_ohlcv, make_synthetic_ohlcv
from market_ml.features import build_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NIFTY50")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    model_path = ROOT / "models" / f"{args.ticker}_model.joblib"
    meta_path = ROOT / "models" / f"{args.ticker}_meta.json"
    if not model_path.exists():
        print(f"No trained model found at {model_path}. Run train_model.py first.")
        return 1

    if args.synthetic:
        df = make_synthetic_ohlcv()
    else:
        df = load_ohlcv(ROOT / "data" / "raw" / f"{args.ticker}.csv")

    model = joblib.load(model_path)
    feature_cols = json.loads(meta_path.read_text())["feature_columns"]
    X_all = build_features(df).dropna()
    latest = X_all.iloc[[-1]][feature_cols]

    prob_up = float(model.predict_proba(latest)[0, 1])
    pred = 1 if prob_up >= 0.5 else 0
    as_of = X_all.index[-1].date()
    print(f"As of {as_of}:")
    print(f"  P(next-day up) = {prob_up:.4f}")
    print(f"  Prediction: {'UP' if pred == 1 else 'DOWN'}")
    print("\nNote: probabilistic estimate only — not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
