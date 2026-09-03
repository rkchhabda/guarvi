"""Train models with walk-forward validation and save the best artifact.

Usage:
    python scripts/train_model.py [--ticker NIFTY50] [--initial-train 500] [--test-size 21]
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

from market_ml.model import AVAILABLE_MODELS, evaluate_walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NIFTY50")
    parser.add_argument("--initial-train", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=21)
    args = parser.parse_args()

    proc = ROOT / "data" / "processed"
    X = pd.read_csv(proc / f"{args.ticker}_features.csv", index_col=0, parse_dates=True)
    y = pd.read_csv(proc / f"{args.ticker}_labels.csv", index_col=0, parse_dates=True)["label"]

    results = {}
    for name in AVAILABLE_MODELS:
        res = evaluate_walk_forward(
            X, y, name, initial_train_size=args.initial_train, test_size=args.test_size
        )
        results[name] = res
        print(f"{name}: walk-forward directional accuracy = {res.mean_accuracy:.4f}")

    best_name = max(results, key=lambda k: results[k].mean_accuracy)
    print(f"\nBest model: {best_name} ({results[best_name].mean_accuracy:.4f})")

    # Refit the winning model on all available data for the saved artifact.
    from market_ml.model import build_model

    model = build_model(best_name)
    model.fit(X, y)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{args.ticker}_model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "best_model": best_name,
        "walk_forward_accuracies": {k: v.mean_accuracy for k, v in results.items()},
        "n_samples": len(X),
        "feature_columns": list(X.columns),
    }
    meta_path = models_dir / f"{args.ticker}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
