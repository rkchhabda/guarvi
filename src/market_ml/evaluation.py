"""Quantitative evaluation of directional predictions.

All metrics compare predictions made strictly BEFORE the outcome date
against the realised next-day direction/return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    roc_auc: float | None = None


@dataclass
class TradingMetrics:
    hit_rate: float
    avg_return_on_buy: float | None
    cumulative_return: float | None
    max_drawdown: float | None
    sharpe: float | None


def classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series | None = None
) -> ClassificationMetrics:
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(int).reset_index(drop=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc = None
    if y_prob is not None and len(set(y_true)) > 1:
        auc = float(roc_auc_score(y_true, pd.Series(y_prob).reset_index(drop=True)))
    return ClassificationMetrics(
        n=len(y_true),
        accuracy=float((y_true == y_pred).mean()),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm.tolist(),
        roc_auc=auc,
    )


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    a = pd.Series(actual).astype(float).reset_index(drop=True)
    p = pd.Series(predicted).astype(float).reset_index(drop=True)
    err = p - a
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    denom = a.abs().replace(0.0, np.nan)
    mape = float((err.abs() / denom).mean() * 100) if denom.notna().any() else float("nan")
    ss_res = float((err**2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def trading_metrics(signals: pd.Series, returns: pd.Series) -> TradingMetrics:
    """Evaluate long-when-predicted-up strategy against next-day returns."""
    s = pd.Series(signals).astype(int).reset_index(drop=True)
    r = pd.Series(returns).astype(float).fillna(0.0).reset_index(drop=True)
    # Guard against bad data: clip daily strategy returns to a plausible range
    r = r.clip(-0.5, 0.5)
    strat = s * r
    hit_rate = float((np.sign(strat[strat != 0]) > 0).mean()) if (strat != 0).any() else float("nan")
    buy_ret = r[s == 1]
    avg_buy = float(buy_ret.mean()) if len(buy_ret) else None

    cum = float((1.0 + strat).prod() - 1.0)
    equity = (1.0 + strat).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    sd = strat.std(ddof=0)
    sharpe = float(strat.mean() / sd * np.sqrt(252)) if sd > 0 else None
    return TradingMetrics(hit_rate, avg_buy, cum, mdd, sharpe)


def calibration_summary(
    y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10
) -> dict[str, object]:
    """Reliability table + over/under-confidence verdict."""
    df = pd.DataFrame({"y": pd.Series(y_true).astype(float).values,
                       "p": pd.Series(y_prob).astype(float).values}).dropna()
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grp = df.groupby("bin", observed=False)["y"].agg(["mean", "count"]).rename(
        columns={"mean": "observed_rate"}
    )
    grp["predicted_mean"] = [iv.mid for iv in grp.index]
    reliability = [
        {"prob_bin": round(float(iv.mid), 3), "n": int(c), "observed_up_rate": (None if np.isnan(o) else round(float(o), 4))}
        for iv, c, o in zip(grp.index, grp["count"], grp["observed_rate"])
        if c > 0
    ]
    mean_p = float(df["p"].mean())
    base = float(df["y"].mean())
    gap = mean_p - base
    verdict = (
        "well-calibrated" if abs(gap) < 0.02
        else ("overconfident (predicts up too often)" if gap > 0 else "underconfident (predicts up too rarely)")
    )
    brier = float(((df["p"] - df["y"]) ** 2).mean())
    return {
        "reliability_by_bin": reliability,
        "mean_predicted_prob": round(mean_p, 4),
        "base_up_rate": round(base, 4),
        "confidence_gap": round(gap, 4),
        "brier_score": round(brier, 4),
        "verdict": verdict,
    }


def naive_baselines(y_true: pd.Series) -> dict[str, float]:
    """Always-up baseline and majority-class baseline accuracies."""
    y = pd.Series(y_true).astype(int)
    return {
        "always_up_accuracy": float((y == 1).mean()),
        "majority_class_accuracy": float(max((y == 1).mean(), (y == 0).mean())),
    }
