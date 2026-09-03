#!/usr/bin/env python3
"""Build the NIFTY100 screening signal: train a pooled model and emit live_signal.json.

Two steps, both run by `python -m scripts.build_signal`:
  1. TRAIN: fit an XGBoost model on the full cross-sectional NIFTY100 feature set
     (62 base features, the same hyperparameters as Phases 3-9) and save the artifact.
  2. BUILD: score the most recent row of every symbol, rank by P(next-day up), and
     write reports/live_signal.json for the web portal to serve.

The signal is RESEARCH-GRADE only (see PROJECT_STATUS.md): ~2 bps/day gross, not
profitable net-of-costs. This script never places orders.

Usage:
    python -m scripts.build_signal                 # train (if missing) + build
    python -m scripts.build_signal --retrain       # force retrain
    python -m scripts.build_signal --features data/processed/nifty100_features.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

MODEL_PATH = ROOT / "models" / "nifty100_model.joblib"
META_PATH = ROOT / "models" / "nifty100_meta.json"
SIGNAL_PATH = ROOT / "reports" / "live_signal.json"
LIVE_DIR = ROOT / "data" / "live" / "nse" / "ohlcv"

# Must match the Phase 3-9 protocol so the screener is comparable to the research.
XGB_KW = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
              tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)

BASE_EXCL = {"date", "symbol", "open", "high", "low", "close", "volume", "source",
             "target_direction", "next_ret", "next_ret_1d", "target_1d",
             "next_ret_5d", "target_5d", "sector"}
LEAKY = {"target_1d", "next_ret_1d", "next_ret", "target_direction",
         "target_5d", "next_ret_5d"}

DISCLAIMER = ("Research/education tool only. The directional signal is small (~2 bps/day gross) "
              "and not profitable net of Indian retail costs; naive equal-weight NIFTY100 has "
              "historically been the better strategy. Not SEBI-registered investment advice.")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")["close"]
    df["next_ret_1d"] = g.shift(-1) / df["close"] - 1.0
    bad = df["next_ret_1d"].abs() > 0.5
    df = df[~bad].dropna(subset=["next_ret_1d"])
    df["target_1d"] = (df["next_ret_1d"] > 0).astype(int)
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in BASE_EXCL and c not in LEAKY
            and pd.api.types.is_numeric_dtype(df[c])]
    assert len(cols) == 62, f"expected 62 base features, got {len(cols)}"
    return cols


def train(df: pd.DataFrame, feats: list[str]) -> "xgb.XGBClassifier":
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(df[feats].values)
    m = xgb.XGBClassifier(**XGB_KW).fit(X, df["target_1d"].values)
    # Save imputer + model together via joblib (model first; imputer in meta pipeline).
    import joblib
    joblib.dump({"model": m, "imputer": imp, "features": feats}, MODEL_PATH)
    br = float(df["target_1d"].mean())
    META_PATH.write_text(json.dumps({"features": feats, "base_rate": br,
                                     "trained_rows": int(len(df)),
                                     "as_of_data": str(df["date"].max().date())}, indent=2))
    # Sanity alarm: in-sample accuracy should NOT be absurdly high (leakage guard).
    acc = float((m.predict(X) == df["target_1d"].values).mean())
    auc = _auc(df["target_1d"].values, m.predict_proba(X)[:, 1])
    log(f"trained on {len(df):,} rows; in-sample acc={acc:.4f} auc={auc:.4f} base_rate={br:.4f}")
    if acc > 0.60 or auc > 0.65:
        raise RuntimeError("SANITY ALARM: in-sample acc>0.60 or AUC>0.65 — likely leakage")
    return m


def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_signal(df: pd.DataFrame, feats: list[str],
                 live_freshness: pd.DataFrame | None = None) -> dict:
    import joblib
    bundle = joblib.load(MODEL_PATH)
    m, imp = bundle["model"], bundle["imputer"]
    # Most recent scored row per symbol.
    last = df.sort_values("date").groupby("symbol").tail(1).reset_index(drop=True)
    X = imp.transform(last[feats].values)
    prob = m.predict_proba(X)[:, 1]
    out = last[["symbol", "date", "close"]].copy()
    out["prob_up"] = prob

    # Live freshness overlay: when `data/live/nse/ohlcv/*.csv` is present and
    # newer than the parquet `as_of`, override `close` and `as_of` per symbol.
    # The model and ranking are unchanged - this only refreshes display fields.
    freshness_note = "no live data; using parquet as_of"
    if live_freshness is not None and not live_freshness.empty:
        lf = live_freshness.copy()
        lf["date"] = pd.to_datetime(lf["date"])
        out["date"] = pd.to_datetime(out["date"])
        lf = lf.drop_duplicates(subset="symbol", keep="last")
        out = out.merge(lf, on="symbol", how="left", suffixes=("", "_live"))
        has_live = out["date_live"].notna() & (out["date_live"] >= out["date"])
        n_overlay = int(has_live.sum())
        out.loc[has_live, "close"] = out.loc[has_live, "last_close"]
        out.loc[has_live, "date"] = out.loc[has_live, "date_live"]
        out = out.drop(columns=["date_live", "last_close"], errors="ignore")
        freshness_note = f"live overlay: {n_overlay}/{len(out)} symbols refreshed"
        log(freshness_note)

    out = out.sort_values("prob_up", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    as_of = str(pd.to_datetime(out["date"]).max().date())
    market_prob = float(out["prob_up"].mean())
    signal = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "universe": "NIFTY100",
        "disclaimer": DISCLAIMER,
        "market_breadth_prob": round(market_prob, 4),
        "direction": "UP" if market_prob >= 0.5 else "DOWN",
        "n_symbols": int(len(out)),
        "freshness": freshness_note,
        "symbols": [
            {"symbol": rec["symbol"], "prob_up": round(float(rec["prob_up"]), 4),
             "rank": int(rec["rank"]), "last_close": round(float(rec["close"]), 2),
             "as_of": str(pd.to_datetime(rec["date"]).date())}
            for rec in out.to_dict("records")
        ],
    }
    SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(json.dumps(signal, indent=2))
    log(f"wrote {SIGNAL_PATH} | as_of={as_of} market_prob={market_prob:.4f} "
        f"n={len(out)} top={out.iloc[0]['symbol']} ({out.iloc[0]['prob_up']:.3f}) | "
        f"{freshness_note}")
    return signal


def collect_live_freshness(live_dir: Path = LIVE_DIR) -> pd.DataFrame:
    """Aggregate intraday OHLCV CSVs into one row per (symbol, day) for freshness.

    The live runner writes `data/live/nse/ohlcv/{symbol}_{YYYYMMDD}.csv` (or
    `_YYYY-MM-DD.csv`); we accept either. Output columns: symbol, date, last_close.
    Returns an empty DataFrame if no live data is present.
    """
    if not live_dir.exists():
        return pd.DataFrame(columns=["symbol", "date", "last_close"])
    files = sorted(p for p in live_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date", "last_close"])
    rows = []
    for path in files:
        # Filename: <symbol>_YYYY-MM-DD.csv or <symbol>_YYYYMMDD.csv
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sym, date_part = parts
        d = pd.to_datetime(date_part, errors="coerce",
                           format="%Y-%m-%d") if "-" in date_part else \
            pd.to_datetime(date_part, errors="coerce", format="%Y%m%d")
        if pd.isna(d):
            continue
        try:
            cdf = pd.read_csv(path)
        except Exception:
            continue
        if cdf.empty or "close" not in cdf.columns:
            continue
        # Use the last row in file order (the runner appends over time).
        last = cdf.iloc[-1]
        rows.append({"symbol": sym, "date": d, "last_close": float(last["close"])})
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "last_close"])
    out = pd.DataFrame(rows).sort_values(["symbol", "date"])
    return out.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--features", default=str(ROOT / "data/processed/nifty100_features.parquet"))
    ap.add_argument("--refresh-from-live", action="store_true",
                    help="Overlay live OHLCV freshness from data/live/nse/ohlcv/*.csv "
                         "onto the scored output (does not retrain).")
    args = ap.parse_args()

    df = load_features(Path(args.features))
    feats = feature_cols(df)

    if args.retrain or not MODEL_PATH.exists():
        log("training NIFTY100 model...")
        train(df, feats)
    else:
        log(f"using cached model {MODEL_PATH}")
        feats = json.loads(META_PATH.read_text())["features"]

    live_freshness = None
    if args.refresh_from_live:
        live_freshness = collect_live_freshness()
        if live_freshness.empty:
            log("no live OHLCV files in data/live/nse/ohlcv; overlay is a no-op")
        else:
            log(f"loaded {len(live_freshness)} symbols with live OHLCV freshness")
    build_signal(df, feats, live_freshness=live_freshness)
    log("DONE")


if __name__ == "__main__":
    main()
