"""Evaluate a trained model and write a report.

Usage:
    python scripts/evaluate_model.py [--ticker NIFTY50] [--initial-train 500] [--test-size 21]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

    lines = [
        "# Walk-forward evaluation report",
        "",
        f"- Ticker: {args.ticker}",
        f"- Samples: {len(X)}",
        f"- Initial train size: {args.initial_train}",
        f"- Test size per fold: {args.test_size}",
        "- Metric: directional accuracy (next-day direction)",
        "",
        "| Model | Walk-forward accuracy | Folds |",
        "|---|---|---|",
    ]

    summary = {}
    for name in AVAILABLE_MODELS:
        res = evaluate_walk_forward(
            X, y, name, initial_train_size=args.initial_train, test_size=args.test_size
        )
        summary[name] = res.mean_accuracy
        lines.append(f"| {name} | {res.mean_accuracy:.4f} | {len(res.folds)} |")

    lines += ["", "> Accuracy figures are historical backtest estimates on synthetic or",
              "> past data. They are NOT guaranteed future performance.", ""]

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"{args.ticker}_evaluation.md"
    report_path.write_text("\n".join(lines))
    (reports / f"{args.ticker}_evaluation.json").write_text(json.dumps(summary, indent=2))
    print(f"Report written to {report_path}")
    for name, acc in summary.items():
        print(f"  {name}: {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
