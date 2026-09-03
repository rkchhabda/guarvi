"""Train and evaluate baseline models for Nifty 100.

Runs five baseline models with expanding-window walk-forward validation
and a 12-month held-out final test period.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --models logistic_regression random_forest xgboost
    python scripts/train_baselines.py --holdout-months 6 --validation-days 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_ml.baselines import (
    BaselineResult,
    run_baselines,
    save_predictions,
    aggregate_overall,
)


def _save_evaluation_report(
    results: list[BaselineResult],
    output_dir: Path,
    data_version: str,
    feature_version: str,
) -> None:
    """Save the evaluation report as markdown and JSON."""
    # JSON report
    report = {
        "data_version": data_version,
        "feature_version": feature_version,
        "validation_methodology": "Expanding-window walk-forward with 12-month holdout",
        "models": {},
    }

    for r in results:
        model_report = {
            "overall_metrics": r.overall_metrics,
            "holdout_metrics": r.holdout_metrics,
            "trading_metrics": {
                "hit_rate": round(r.trading_metrics.hit_rate, 4),
                "avg_trade_return": round(r.trading_metrics.avg_trade_return, 6),
                "cumulative_return": round(r.trading_metrics.cumulative_return, 4),
                "max_drawdown": round(r.trading_metrics.max_drawdown, 4),
                "sharpe_ratio": round(r.trading_metrics.sharpe_ratio, 4),
                "turnover": round(r.trading_metrics.turnover, 4),
                "n_trades": r.trading_metrics.n_trades,
                "n_long_signals": r.trading_metrics.n_long_signals,
            },
            "holdout_trading_metrics": {
                "hit_rate": round(r.holdout_trading.hit_rate, 4),
                "avg_trade_return": round(r.holdout_trading.avg_trade_return, 6),
                "cumulative_return": round(r.holdout_trading.cumulative_return, 4),
                "max_drawdown": round(r.holdout_trading.max_drawdown, 4),
                "sharpe_ratio": round(r.holdout_trading.sharpe_ratio, 4),
                "n_trades": r.holdout_trading.n_trades,
            },
            "n_folds": len(r.fold_metrics),
        }
        report["models"][r.model_name] = model_report

    json_path = output_dir / "baseline_evaluation.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    # Markdown report
    md_lines = [
        "# Baseline Model Evaluation Report",
        "",
        f"**Generated**: 2026-08-26",
        f"**Data version**: {data_version}",
        f"**Feature version**: {feature_version}",
        "",
        "## Validation Methodology",
        "",
        "- Expanding-window walk-forward validation",
        "- Final 12 months held out as untouched test set",
        "- Monthly retraining (every 21 trading days)",
        "- Minimum3-year training history required",
        "- Scalers/imputers fitted on training portion only",
        "- No random splitting; strict temporal ordering",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Bal. Accuracy | F1 | ROC-AUC | Brier | Sharpe | Cum. Return | Max DD |",
        "|-------|----------|---------------|-----|---------|-------|--------|-------------|--------|",
    ]

    for r in results:
        m = r.overall_metrics
        t = r.trading_metrics
        md_lines.append(
            f"| {r.model_name} | {m.get('accuracy', 0):.4f} | "
            f"{m.get('balanced_accuracy', 0):.4f} | {m.get('f1', 0):.4f} | "
            f"{m.get('roc_auc', 'N/A')} | {m.get('brier_score', 0):.4f} | "
            f"{t.sharpe_ratio:.4f} | {t.cumulative_return:.4f} | {t.max_drawdown:.4f} |"
        )

    md_lines.extend([
        "",
        "## Holdout Period Results (Last 12 Months)",
        "",
        "| Model | Accuracy | Bal. Accuracy | F1 | Brier | Sharpe | Cum. Return |",
        "|-------|----------|---------------|-----|-------|--------|-------------|",
    ])

    for r in results:
        m = r.holdout_metrics
        t = r.holdout_trading
        md_lines.append(
            f"| {r.model_name} | {m.get('accuracy', 0):.4f} | "
            f"{m.get('balanced_accuracy', 0):.4f} | {m.get('f1', 0):.4f} | "
            f"{m.get('brier_score', 0):.4f} | {t.sharpe_ratio:.4f} | {t.cumulative_return:.4f} |"
        )

    md_lines.extend([
        "",
        "## 95% Confidence Intervals (Accuracy)",
        "",
        "| Model | Accuracy | Lower CI | Upper CI |",
        "|-------|----------|----------|----------|",
    ])

    for r in results:
        ci = r.overall_metrics.get("accuracy_ci_95", [0, 0])
        md_lines.append(
            f"| {r.model_name} | {r.overall_metrics.get('accuracy', 0):.4f} | "
            f"{ci[0]:.4f} | {ci[1]:.4f} |"
        )

    md_lines.extend([
        "",
        "## Trading Metrics (After Costs)",
        "",
        "Configuration: long-only, threshold=0.55, transaction_cost=0.1%, slippage=0.05%",
        "",
        "| Model | Hit Rate | Avg Trade Ret | # Trades | Turnover | Sharpe |",
        "|-------|----------|---------------|----------|----------|--------|",
    ])

    for r in results:
        t = r.trading_metrics
        md_lines.append(
            f"| {r.model_name} | {t.hit_rate:.4f} | {t.avg_trade_return:.6f} | "
            f"{t.n_trades} | {t.turnover:.4f} | {t.sharpe_ratio:.4f} |"
        )

    md_lines.extend([
        "",
        "## Fold-Level Metrics",
        "",
    ])

    for r in results:
        md_lines.append(f"### {r.model_name}")
        md_lines.append("")
        md_lines.append("| Fold | Train End | Test Start | Test End | N | Accuracy | AUC |")
        md_lines.append("|------|-----------|------------|----------|---|----------|-----|")
        for fm in r.fold_metrics:
            md_lines.append(
                f"| {fm.fold_id} | {fm.train_end_date.strftime('%Y-%m-%d')} | "
                f"{fm.test_start_date.strftime('%Y-%m-%d')} | "
                f"{fm.test_end_date.strftime('%Y-%m-%d')} | "
                f"{fm.n_test} | {fm.accuracy:.4f} | "
                f"{fm.roc_auc:.4f if fm.roc_auc else 'N/A'} |"
            )
        md_lines.append("")

    md_lines.extend([
        "## Survivorship-Bias Limitation",
        "",
        "This evaluation uses current Nifty 100 constituents. Stocks that were",
        "removed from the index before the evaluation date are NOT included.",
        "Backtest results may be upwardly biased due to survivorship bias.",
        "",
        "## Limitations and Next Steps",
        "",
        "- Baselines use only price/volume-derived features; no fundamental data",
        "- No market regime detection in this phase",
        "- Hyperparameters are default; tuning may improve results",
        "- Consider adding sector/market-cap features",
        "- Transaction costs are estimates; actual costs vary by broker",
        "- Model ensembling and stacking are potential next steps",
    ])

    md_path = output_dir / "baseline_evaluation.md"
    md_path.write_text("\n".join(md_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", default=str(ROOT / "data" / "processed" / "nifty100_features.parquet"),
        help="Path to feature dataset",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports"),
        help="Directory for output reports",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["always_up", "previous_direction", "logistic_regression", "random_forest", "xgboost"],
        help="Models to train",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")
    parser.add_argument("--trade-threshold", type=float, default=0.55, help="Trading signal threshold")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost (0.001 = 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage (0.0005 = 0.05%)")
    parser.add_argument("--holdout-months", type=int, default=12, help="Holdout period in months")
    parser.add_argument("--min-train-years", type=float, default=3.0, help="Minimum training history in years")
    parser.add_argument("--validation-days", type=int, default=21, help="Validation window in trading days")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 9: BASELINE MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    results = run_baselines(
        data_path=args.input,
        output_dir=args.output_dir,
        models=args.models,
        seed=args.seed,
        threshold=args.threshold,
        trade_threshold=args.trade_threshold,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        holdout_months=args.holdout_months,
        min_train_years=args.min_train_years,
        validation_days=args.validation_days,
    )

    # Save predictions
    output_dir = Path(args.output_dir)
    predictions_path = output_dir.parent / "data" / "processed" / "baseline_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(results, predictions_path)

    # Save evaluation report
    _save_evaluation_report(
        results, output_dir,
        data_version="nifty100_ohlcv_v1_2026-08-26",
        feature_version="nifty100_features_v1_2026-08-26",
    )

    # Save by-symbol and by-period CSVs
    all_by_sym = []
    all_by_per = []
    for r in results:
        if not r.by_symbol.empty:
            all_by_sym.append(r.by_symbol)
        if not r.by_period.empty:
            all_by_per.append(r.by_period)

    if all_by_sym:
        pd.concat(all_by_sym, ignore_index=True).to_csv(
            output_dir / "baseline_by_symbol.csv", index=False
        )
    if all_by_per:
        pd.concat(all_by_per, ignore_index=True).to_csv(
            output_dir / "baseline_by_period.csv", index=False
        )

    # Save trained model artifacts
    models_dir = ROOT / "models" / "baselines"
    models_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        artifact = {
            "model_name": r.model_name,
            "overall_metrics": r.overall_metrics,
            "holdout_metrics": r.holdout_metrics,
            "trading_metrics": {
                "hit_rate": r.trading_metrics.hit_rate,
                "sharpe_ratio": r.trading_metrics.sharpe_ratio,
                "cumulative_return": r.trading_metrics.cumulative_return,
            },
            "n_folds": len(r.fold_metrics),
        }
        (models_dir / f"{r.model_name}_meta.json").write_text(
            json.dumps(artifact, indent=2, default=str)
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Predictions: {predictions_path}")
    print(f"Report: {output_dir / 'baseline_evaluation.md'}")
    print(f"JSON: {output_dir / 'baseline_evaluation.json'}")
    print(f"By-symbol: {output_dir / 'baseline_by_symbol.csv'}")
    print(f"By-period: {output_dir / 'baseline_by_period.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
