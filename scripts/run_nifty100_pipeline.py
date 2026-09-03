"""Real-data Nifty 100 pipeline: download -> walk-forward predictions -> report.

Usage:
    python scripts/run_nifty100_pipeline.py [--start 2019-01-01] [--initial-train 500]
                                            [--test-size 21] [--limit N]

Produces the required outputs:
    data/processed/predictions.csv
    reports/NIFTY100_evaluation.md
    reports/NIFTY100_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_ml.data import load_ohlcv
from market_ml.download import ensure_raw_data
from market_ml.evaluation import (
    calibration_summary,
    classification_metrics,
    naive_baselines,
    trading_metrics,
)
from market_ml.features import build_features
from market_ml.model import build_model
from market_ml.splits import walk_forward_splits
from market_ml.universe import NIFTY100_SYMBOLS


def predict_symbol(sym: str, raw_path: Path, initial_train: int, test_size: int) -> pd.DataFrame | None:
    """Walk-forward OOS predictions for one symbol; None if insufficient data."""
    try:
        df = load_ohlcv(raw_path)
    except Exception:
        return None
    if len(df) < initial_train + test_size + 60:
        return None

    X = build_features(df).replace([np.inf, -np.inf], np.nan).dropna()
    y = (df["close"].shift(-1) > df["close"]).astype(float).reindex(X.index)
    # Remove leakage-prone / bad-data rows: daily moves beyond ±50% are almost
    # always corporate actions (demergers, splits) mispriced in adjusted data.
    ret = (df["close"].shift(-1) / df["close"] - 1.0).reindex(X.index)
    valid = X.notna().all(axis=1) & y.notna() & ret.notna() & (ret.abs() <= 0.5)
    X, y, ret = X[valid], y[valid], ret[valid]
    if len(X) < initial_train + test_size:
        return None

    rows = []
    splits = walk_forward_splits(len(X), initial_train_size=initial_train, test_size=test_size)
    model = build_model("xgboost")
    y_arr = y.values
    ret_arr = ret.values
    for split in splits:
        assert split.train_idx[-1] < split.test_idx[0], "leakage guard"
        model.fit(X.iloc[split.train_idx], y_arr[split.train_idx])
        prob = model.predict_proba(X.iloc[split.test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        for j, date in enumerate(X.index[split.test_idx]):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "predicted_direction": int(pred[j]),
                    "predicted_probability": float(prob[j]),
                    "actual_direction": int(y_arr[split.test_idx][j]),
                    "actual_return": float(ret_arr[split.test_idx][j]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--initial-train", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols (testing)")
    args = parser.parse_args()

    symbols = NIFTY100_SYMBOLS[: args.limit] if args.limit else NIFTY100_SYMBOLS
    print(f"Ensuring raw data for {len(symbols)} symbols (cached under data/raw/)...")
    paths = ensure_raw_data(symbols, start=args.start)

    print("Generating walk-forward OOS predictions...")
    frames = []
    t0 = time.time()
    for i, sym in enumerate(sorted(paths)):
        out = predict_symbol(sym, paths[sym], args.initial_train, args.test_size)
        if out is not None and len(out):
            frames.append(out)
            print(f"[{i + 1}/{len(paths)}] {sym}: {len(out)} OOS predictions")
        else:
            print(f"[{i + 1}/{len(paths)}] {sym}: skipped (insufficient data)")

    if not frames:
        print("No symbols produced predictions — aborting.")
        return 1
    preds = pd.concat(frames, ignore_index=True)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.sort_values(["date", "symbol"]).reset_index(drop=True)

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    preds.to_csv(proc / "predictions.csv", index=False)
    print(f"Saved {len(preds)} predictions -> {proc / 'predictions.csv'}")

    # ---- Evaluation ----------------------------------------------------------
    valid = preds.dropna(subset=["actual_direction", "actual_return"])
    checks = {
        "all_rows_have_outcome": bool(valid[["actual_direction", "actual_return"]].notna().all().all()),
        "rows_missing_outcome": int(len(preds) - len(valid)),
        "no_duplicate_date_symbol": not preds.duplicated(["date", "symbol"]).any(),
        "probabilities_in_range": bool(preds["predicted_probability"].between(0, 1).all()),
        "directions_binary": bool(preds["predicted_direction"].isin([0, 1]).all()),
        "symbols_evaluated": int(preds["symbol"].nunique()),
    }

    cm = classification_metrics(valid["actual_direction"], valid["predicted_direction"],
                                valid["predicted_probability"])
    trade = trading_metrics(valid["predicted_direction"], valid["actual_return"])
    calib = calibration_summary(valid["actual_direction"], valid["predicted_probability"])
    baselines = naive_baselines(valid["actual_direction"])

    mid = valid["date"].quantile(0.5)
    period_acc = {
        "validation_period": float((valid[valid["date"] <= mid]["actual_direction"]
                                   == valid[valid["date"] <= mid]["predicted_direction"]).mean()),
        "out_of_sample_test": float((valid[valid["date"] > mid]["actual_direction"]
                                     == valid[valid["date"] > mid]["predicted_direction"]).mean()),
        "combined_oos": cm.accuracy,
    }

    sym_stats = []
    for sym, g in valid.groupby("symbol"):
        m = classification_metrics(g["actual_direction"], g["predicted_direction"], g["predicted_probability"])
        t = trading_metrics(g["predicted_direction"], g["actual_return"])
        sym_stats.append({"symbol": sym, "accuracy": round(m.accuracy, 4), "f1": round(m.f1, 4),
                          "roc_auc": None if m.roc_auc is None else round(m.roc_auc, 4),
                          "hit_rate": round(t.hit_rate, 4),
                          "cumulative_return": round(t.cumulative_return, 4), "n": m.n})
    sym_df = pd.DataFrame(sym_stats).sort_values("accuracy", ascending=False)

    monthly = valid.assign(month=valid["date"].dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: (g["actual_direction"] == g["predicted_direction"]).mean(), include_groups=False
    )

    acc = cm.accuracy
    edge = acc - baselines["always_up_accuracy"]
    unrealistic = []
    if acc > 0.70:
        unrealistic.append(f"Accuracy {acc:.4f} suspiciously high — investigate leakage.")
    if cm.roc_auc and cm.roc_auc > 0.75:
        unrealistic.append(f"ROC-AUC {cm.roc_auc:.4f} unusually high for daily direction.")
    if trade.cumulative_return is not None and abs(trade.cumulative_return) > 1000:
        unrealistic.append(
            f"Cumulative strategy return {trade.cumulative_return:.2e} is implausible — "
            "likely extreme daily returns in the data (e.g. corporate-action gaps); "
            "returns were clipped to ±50%/day for this metric."
        )
    if acc < 0.55:
        verdict_reason = ("Accuracy below 55%: daily direction is dominated by noise; "
                          "technical-only features carry little signal at horizon=1.")
    elif acc < 0.60:
        verdict_reason = (f"Accuracy between 55% and 60%; cross-symbol range "
                          f"{sym_df['accuracy'].min():.4f}-{sym_df['accuracy'].max():.4f}, "
                          f"monthly std {monthly.std():.4f}.")
    else:
        holds = period_acc["out_of_sample_test"] >= 0.60
        verdict_reason = f"Accuracy above 60%; OOS test period {'confirms' if holds else 'does NOT confirm'} it."

    lines = [
        "# Nifty 100 universe — quantitative evaluation report",
        "",
        "## 1. Executive summary",
        "",
        f"- Real OHLCV data (Yahoo Finance, adjusted), {checks['symbols_evaluated']} symbols, "
        f"{len(valid):,} evaluated OOS predictions ({valid['date'].min().date()} to {valid['date'].max().date()}), "
        "expanding-window walk-forward with train strictly before test.",
        f"- Directional accuracy **{acc:.4f}** vs always-up baseline **{baselines['always_up_accuracy']:.4f}** "
        f"(edge {edge:+.4f}).",
        f"- ROC-AUC {cm.roc_auc if cm.roc_auc is None else round(cm.roc_auc, 4)}; Brier {calib['brier_score']}; "
        f"calibration: **{calib['verdict']}**.",
        f"- Trading: hit rate {trade.hit_rate:.4f}, cumulative return "
        f"{None if trade.cumulative_return is None else round(trade.cumulative_return, 4)}, "
        f"max drawdown {None if trade.max_drawdown is None else round(trade.max_drawdown, 4)} "
        "(before costs; accuracy ≠ trading success).",
        "",
        "## 2. Metrics table",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples | {cm.n} |",
        f"| Directional accuracy | {cm.accuracy:.4f} |",
        f"| Precision (up) | {cm.precision:.4f} |",
        f"| Recall (up) | {cm.recall:.4f} |",
        f"| F1 (up) | {cm.f1:.4f} |",
        f"| ROC-AUC | {'' if cm.roc_auc is None else f'{cm.roc_auc:.4f}'} |",
        f"| Confusion matrix [[TN,FP],[FN,TP]] | {cm.confusion_matrix} |",
        f"| Hit rate | {trade.hit_rate:.4f} |",
        f"| Avg return on buy signals | {None if trade.avg_return_on_buy is None else round(trade.avg_return_on_buy, 6)} |",
        f"| Cumulative strategy return | {None if trade.cumulative_return is None else round(trade.cumulative_return, 4)} |",
        f"| Max drawdown | {None if trade.max_drawdown is None else round(trade.max_drawdown, 4)} |",
        f"| Sharpe (annualised) | {None if trade.sharpe is None else round(trade.sharpe, 3)} |",
        f"| Always-up baseline | {baselines['always_up_accuracy']:.4f} |",
        "",
        "### By period",
        "",
        "| Period | Accuracy |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.4f} |" for k, v in period_acc.items()]
    lines += ["", "### Calibration reliability (deciles)", "",
              "| Prob bin | N | Observed up rate |", "|---|---|---|"]
    lines += [f"| {r['prob_bin']} | {r['n']} | {r['observed_up_rate']} |" for r in calib["reliability_by_bin"]]
    lines += [
        "", f"Mean predicted prob {calib['mean_predicted_prob']} vs base rate {calib['base_up_rate']} → **{calib['verdict']}**.",
        "", "## 3. Symbol-wise comparison", "", "### Best 10", "",
        "| Symbol | Accuracy | F1 | ROC-AUC | Hit rate | Cum return | N |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in sym_df.head(10).iterrows():
        lines.append(f"| {r['symbol']} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
                     f"{'' if r['roc_auc'] is None else f'{r['roc_auc']:.4f}'} | {r['hit_rate']:.4f} | "
                     f"{r['cumulative_return']:.4f} | {r['n']} |")
    lines += ["", "### Worst 10", "",
              "| Symbol | Accuracy | F1 | ROC-AUC | Hit rate | Cum return | N |",
              "|---|---|---|---|---|---|---|"]
    for _, r in sym_df.tail(10).iloc[::-1].iterrows():
        lines.append(f"| {r['symbol']} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
                     f"{'' if r['roc_auc'] is None else f'{r['roc_auc']:.4f}'} | {r['hit_rate']:.4f} | "
                     f"{r['cumulative_return']:.4f} | {r['n']} |")
    lines += ["", "### Monthly stability", "", "| Month | Accuracy |", "|---|---|"]
    lines += [f"| {m} | {a:.4f} |" for m, a in monthly.items()]
    lines += [
        "", "## 4. Risk and leakage check", "",
        f"- Quality checks: `{json.dumps(checks)}`",
        "- Predictions are made using data strictly before each outcome date (asserted per fold); labels use shift(-1).",
        f"- Unrealistic-metric flags: {unrealistic if unrealistic else 'none'}.",
        f"- Instability: cross-symbol accuracy std {sym_df['accuracy'].std():.4f}; monthly std {monthly.std():.4f}.",
        f"- Context: {verdict_reason}",
        "", "## 5. Final judgment", "",
        f"- 60% directional accuracy target: {'MET' if acc >= 0.60 else 'NOT MET'} ({acc:.4f}); treated as an evaluation target, never a promise.",
        f"- vs naive baseline: {'BETTER' if edge > 0 else 'NOT BETTER'} than always-up by {abs(edge):.4f}.",
        "- Metrics ignore transaction costs, slippage, and taxes; directional accuracy does not imply profitability.",
        "",
    ]

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "NIFTY100_evaluation.md").write_text("\n".join(lines), encoding="utf-8")

    summary_json = {
        "data_source": "Yahoo Finance adjusted OHLCV via yfinance (no credentials)",
        "checks": checks,
        "combined": {"accuracy": cm.accuracy, "precision": cm.precision, "recall": cm.recall,
                     "f1": cm.f1, "roc_auc": cm.roc_auc},
        "periods": period_acc,
        "baselines": baselines,
        "trading": {"hit_rate": trade.hit_rate,
                    "avg_buy_return": trade.avg_return_on_buy,
                    "cumulative_return": trade.cumulative_return,
                    "max_drawdown": trade.max_drawdown, "sharpe": trade.sharpe},
        "calibration": calib,
        "best_symbols": sym_df.head(5).to_dict("records"),
        "worst_symbols": sym_df.tail(5).to_dict("records"),
        "monthly_accuracy": {k: round(float(v), 4) for k, v in monthly.items()},
        "unrealistic_flags": unrealistic,
    }
    (reports / "NIFTY100_evaluation.json").write_text(
        json.dumps(summary_json, indent=2, default=str), encoding="utf-8"
    )
    print("\nReport written -> reports/NIFTY100_evaluation.md (+ .json)")
    print(f"Directional accuracy: {acc:.4f} | baseline: {baselines['always_up_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
