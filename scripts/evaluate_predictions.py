"""Generate out-of-sample predictions and run the full quantitative evaluation.

Usage:
    python scripts/evaluate_predictions.py [--n-symbols 100] [--initial-train 500]
                                           [--test-size 21] [--seed 42]

Generates deterministic synthetic OHLCV series for a Nifty-100-style universe,
produces walk-forward out-of-sample predictions (made strictly before the
outcome date), saves them to data/processed/predictions.csv, and writes the
evaluation report to reports/NIFTY100_evaluation.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_ml.data import make_synthetic_ohlcv
from market_ml.evaluation import (
    calibration_summary,
    classification_metrics,
    naive_baselines,
    regression_metrics,
    trading_metrics,
)
from market_ml.features import build_features
from market_ml.model import AVAILABLE_MODELS, evaluate_walk_forward


def generate_predictions(n_symbols: int, initial_train: int, test_size: int, seed: int) -> pd.DataFrame:
    """Walk-forward OOS predictions for each symbol using the best model type."""
    rows = []
    model_name = "xgboost"
    for i in range(n_symbols):
        symbol = f"SYM{i:03d}"
        df = make_synthetic_ohlcv(n_days=750, seed=seed + i)
        X = build_features(df).replace([np.inf, -np.inf], np.nan).dropna()
        y = (df["close"].shift(-1) > df["close"]).astype(float).reindex(X.index)
        ret = (df["close"].shift(-1) / df["close"] - 1.0).reindex(X.index)

        from market_ml.splits import walk_forward_splits

        splits = walk_forward_splits(len(X), initial_train_size=initial_train, test_size=test_size)
        y_arr = y.values
        for split in splits:
            assert split.train_idx[-1] < split.test_idx[0], "leakage guard"
            from market_ml.model import build_model

            model = build_model(model_name)
            model.fit(X.iloc[split.train_idx], y_arr[split.train_idx])
            prob = model.predict_proba(X.iloc[split.test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            dates = X.index[split.test_idx]
            for j in range(len(dates)):
                rows.append(
                    {
                        "date": dates[j],
                        "symbol": symbol,
                        "predicted_direction": int(pred[j]),
                        "predicted_probability": float(prob[j]),
                        "actual_direction": int(y_arr[split.test_idx][j]),
                        "actual_return": float(ret.values[split.test_idx][j]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-symbols", type=int, default=100)
    parser.add_argument("--initial-train", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating OOS predictions for {args.n_symbols} symbols...")
    preds = generate_predictions(args.n_symbols, args.initial_train, args.test_size, args.seed)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.sort_values(["date", "symbol"]).reset_index(drop=True)

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    preds.to_csv(proc / "predictions.csv", index=False)
    print(f"Saved {len(preds)} predictions -> {proc / 'predictions.csv'}")

    # Leakage / quality checks -------------------------------------------------
    checks = {
        "all_rows_have_outcome": bool(preds[["actual_direction", "actual_return"]].notna().all().all()),
        "rows_missing_outcome": int(preds[["actual_direction", "actual_return"]].isna().any(axis=1).sum()),
        "no_duplicate_date_symbol": not preds.duplicated(["date", "symbol"]).any(),
        "probabilities_in_range": bool(preds["predicted_probability"].between(0, 1).all()),
        "directions_binary": bool(preds["predicted_direction"].isin([0, 1]).all()),
    }

    # Universe-level metrics (drop any rows lacking a realised outcome) ---------
    valid = preds.dropna(subset=["actual_direction", "actual_return"])
    cm_all = classification_metrics(
        valid["actual_direction"], valid["predicted_direction"], valid["predicted_probability"]
    )
    reg_all = regression_metrics(valid["actual_return"], valid["predicted_probability"] - 0.5)
    trade_all = trading_metrics(valid["predicted_direction"], valid["actual_return"])
    calib = calibration_summary(valid["actual_direction"], valid["predicted_probability"])
    baselines = naive_baselines(valid["actual_direction"])

    # Period split: first half of OOS window = validation, second half = test ---
    mid = preds["date"].quantile(0.5)
    val_mask = preds["date"] <= mid
    periods = {
        "validation": preds[val_mask],
        "out_of_sample_test": preds[~val_mask],
    }
    period_acc = {k: float((p["actual_direction"] == p["predicted_direction"]).mean()) for k, p in periods.items()}
    period_acc["combined_oos"] = cm_all.accuracy

    # Symbol-wise --------------------------------------------------------------
    sym_stats = []
    for sym, g in valid.groupby("symbol"):
        m = classification_metrics(g["actual_direction"], g["predicted_direction"], g["predicted_probability"])
        t = trading_metrics(g["predicted_direction"], g["actual_return"])
        sym_stats.append({"symbol": sym, "accuracy": m.accuracy, "f1": m.f1, "roc_auc": m.roc_auc,
                          "hit_rate": t.hit_rate, "cumulative_return": t.cumulative_return, "n": m.n})
    sym_df = pd.DataFrame(sym_stats).sort_values("accuracy", ascending=False)

    # Monthly stability --------------------------------------------------------
    preds["month"] = preds["date"].dt.to_period("M").astype(str)
    monthly = valid.assign(month=valid["date"].dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: (g["actual_direction"] == g["predicted_direction"]).mean(), include_groups=False
    )

    acc = cm_all.accuracy
    if acc < 0.55:
        verdict_reason = (
            f"Combined directional accuracy {acc:.4f} is below 55%. Likely reasons: weak "
            "signal-to-noise ratio in daily equity returns, feature set dominated by noise "
            "at horizon=1, and regime shifts across folds."
        )
    elif acc < 0.60:
        spread = sym_df["accuracy"].max() - sym_df["accuracy"].min()
        verdict_reason = (
            f"Accuracy {acc:.4f} lies between 55% and 60%. Cross-symbol accuracy range "
            f"{sym_df['accuracy'].min():.4f}-{sym_df['accuracy'].max():.4f} (spread {spread:.4f}); "
            f"monthly range {monthly.min():.4f}-{monthly.max():.4f}."
        )
    else:
        oos_ok = period_acc["out_of_sample_test"] >= 0.60
        verdict_reason = (
            f"Accuracy {acc:.4f} exceeds 60%; out-of-sample test period accuracy "
            f"{period_acc['out_of_sample_test']:.4f} ({'holds' if oos_ok else 'does NOT hold'})."
        )

    unrealistic = []
    if acc > 0.70:
        unrealistic.append(f"Directional accuracy {acc:.4f} is suspiciously high — check for leakage.")
    if cm_all.roc_auc and cm_all.roc_auc > 0.75:
        unrealistic.append(f"ROC-AUC {cm_all.roc_auc:.4f} is unusually high for daily direction.")
    if abs(reg_all["r2"]) > 0.5:
        unrealistic.append(f"Return-prediction R² {reg_all['r2']:.4f} is implausible (proxy regression on probabilities, not a return model).")

    lines = [
        "# Nifty 100-style universe — quantitative evaluation report",
        "",
        "## 1. Executive summary",
        "",
        f"- Predictions evaluated: **{len(preds):,}** across **{preds['symbol'].nunique()}** symbols "
        f"({preds['date'].min().date()} to {preds['date'].max().date()}), all generated by expanding-window "
        "walk-forward backtests with training indices strictly before test indices.",
        f"- Combined directional accuracy: **{acc:.4f}** vs always-up baseline **{baselines['always_up_accuracy']:.4f}** "
        f"(edge: {acc - baselines['always_up_accuracy']:+.4f}).",
        f"- ROC-AUC: {cm_all.roc_auc if cm_all.roc_auc is None else round(cm_all.roc_auc, 4)}; "
        f"Brier score: {calib['brier_score']}. Calibration verdict: **{calib['verdict']}**.",
        f"- Strategy cumulative return: {trade_all.cumulative_return if trade_all.cumulative_return is None else round(trade_all.cumulative_return, 4)}, "
        f"max drawdown {trade_all.max_drawdown if trade_all.max_drawdown is None else round(trade_all.max_drawdown, 4)}.",
        "",
        "> NOTE: this run uses deterministic SYNTHETIC data because no real Nifty 100 prediction file was provided. "
        "Metrics characterise the pipeline's behaviour, not real-market performance.",
        "",
        "## 2. Metrics table",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples | {cm_all.n} |",
        f"| Directional accuracy | {cm_all.accuracy:.4f} |",
        f"| Precision (up) | {cm_all.precision:.4f} |",
        f"| Recall (up) | {cm_all.recall:.4f} |",
        f"| F1 (up) | {cm_all.f1:.4f} |",
        f"| ROC-AUC | {'' if cm_all.roc_auc is None else f'{cm_all.roc_auc:.4f}'} |",
        f"| Confusion matrix [[TN,FP],[FN,TP]] | {cm_all.confusion_matrix} |",
        f"| MAE (return proxy) | {reg_all['mae']:.6f} |",
        f"| RMSE (return proxy) | {reg_all['rmse']:.6f} |",
        f"| R-squared (return proxy) | {reg_all['r2']:.4f} |",
        f"| Hit rate (long signals) | {trade_all.hit_rate:.4f} |",
        f"| Avg return on buy signals | {None if trade_all.avg_return_on_buy is None else round(trade_all.avg_return_on_buy, 6)} |",
        f"| Cumulative strategy return | {None if trade_all.cumulative_return is None else round(trade_all.cumulative_return, 4)} |",
        f"| Max drawdown | {None if trade_all.max_drawdown is None else round(trade_all.max_drawdown, 4)} |",
        f"| Sharpe (annualised) | {None if trade_all.sharpe is None else round(trade_all.sharpe, 3)} |",
        f"| Always-up baseline accuracy | {baselines['always_up_accuracy']:.4f} |",
        f"| Majority-class baseline | {baselines['majority_class_accuracy']:.4f} |",
        "",
        "### By period",
        "",
        "| Period | Accuracy |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.4f} |" for k, v in period_acc.items()]
    lines += [
        "",
        "### Calibration reliability (deciles)",
        "",
        "| Prob bin | N | Observed up rate |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['prob_bin']} | {r['n']} | {r['observed_up_rate']} |" for r in calib["reliability_by_bin"]
    ]
    lines += [
        "",
        f"Mean predicted prob {calib['mean_predicted_prob']} vs base up-rate {calib['base_up_rate']} "
        f"(gap {calib['confidence_gap']}) → **{calib['verdict']}**.",
        "",
        "## 3. Symbol-wise comparison",
        "",
        "### Best 10",
        "",
        "| Symbol | Accuracy | F1 | ROC-AUC | Hit rate | Cum return | N |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in sym_df.head(10).iterrows():
        lines.append(f"| {r['symbol']} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
                     f"{'' if r['roc_auc'] is None else f'{r['roc_auc']:.4f}'} | {r['hit_rate']:.4f} | "
                     f"{round(r['cumulative_return'], 4)} | {r['n']} |")
    lines += ["", "### Worst 10", "", "| Symbol | Accuracy | F1 | ROC-AUC | Hit rate | Cum return | N |", "|---|---|---|---|---|---|---|"]
    for _, r in sym_df.tail(10).iloc[::-1].iterrows():
        lines.append(f"| {r['symbol']} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
                     f"{'' if r['roc_auc'] is None else f'{r['roc_auc']:.4f}'} | {r['hit_rate']:.4f} | "
                     f"{round(r['cumulative_return'], 4)} | {r['n']} |")

    lines += [
        "",
        "### Monthly stability",
        "",
        "| Month | Accuracy |",
        "|---|---|",
    ]
    lines += [f"| {m} | {a:.4f} |" for m, a in monthly.items()]

    lines += [
        "",
        "## 4. Risk and leakage check",
        "",
        f"- Quality checks: {json.dumps(checks)}",
        f"- Rows excluded for missing outcome: {checks['rows_missing_outcome']} (final test-window rows with no next-day close).",
        "- Every fold trains only on indices strictly earlier than its test window "
        "(asserted at generation time); labels use shift(-1) so predictions precede outcomes.",
        f"- Unrealistic-metric flags: {unrealistic if unrealistic else 'none'}.",
        f"- Instability: cross-symbol accuracy std {sym_df['accuracy'].std():.4f}; "
        f"monthly accuracy std {monthly.std():.4f}.",
        f"- Verdict context: {verdict_reason}",
        "",
        "## 5. Final judgment",
        "",
        f"- Target of 60% directional accuracy: {'MET' if acc >= 0.60 else 'NOT MET'} on this run "
        f"({acc:.4f}). Per AGENTS.md rules, 60% is treated as an evaluation target, never a promise.",
        f"- Model vs naive baseline: {'BETTER' if acc > baselines['always_up_accuracy'] else 'NOT BETTER'} "
        f"than always-up ({acc:.4f} vs {baselines['always_up_accuracy']:.4f}).",
        "- Directional accuracy does NOT imply trading profitability after costs; the trading metrics above ignore fees, slippage, and taxes.",
        "",
        "### Next steps",
        "",
        "1. Re-run with a real Nifty 100 predictions CSV (`data/processed/predictions.csv` schema: date, symbol, predicted_direction, predicted_probability, actual_direction, actual_return).",
        "2. Add transaction costs to the trading metrics before any capital decision.",
        "3. Test stability across regimes (bull/bear sub-periods) and per-sector groupings.",
        "4. If accuracy hovers near 50%, enrich features (breadth, volatility regime flags) or lengthen the horizon.",
        "5. Track calibration drift month over month; recalibrate probabilities if Brier score rises.",
        "",
    ]

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "NIFTY100_evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_json = {
        "checks": checks,
        "combined": {"accuracy": cm_all.accuracy, "precision": cm_all.precision,
                     "recall": cm_all.recall, "f1": cm_all.f1, "roc_auc": cm_all.roc_auc},
        "periods": period_acc,
        "baselines": baselines,
        "calibration": calib,
        "best_symbols": sym_df.head(5).to_dict("records"),
        "worst_symbols": sym_df.tail(5).to_dict("records"),
        "unrealistic_flags": unrealistic,
    }
    (reports / "NIFTY100_evaluation.json").write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")
    print(f"Report written -> {report_path}")
    print(f"\nCombined directional accuracy: {acc:.4f} (baseline {baselines['always_up_accuracy']:.4f})")
    print(f"Period accuracies: {json.dumps({k: round(v, 4) for k, v in period_acc.items()})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
