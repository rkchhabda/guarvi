#!/usr/bin/env python3
"""Refresh NIFTY100 features with new daily OHLCV from the live data collector.

Why this exists:
  The frozen `nifty100_features.parquet` ends ~2026-08-24 (whatever the last
  full trading day in the snapshot was). The web portal wants fresh technicals
  (RSI, MACD, returns, vol) every day, not just a refreshed close. This
  script:
    1. Globs `data/live/nse/ohlcv/*.csv` (5-min Groww candles, one file per
       symbol per day).
    2. Aggregates the intraday candles into one daily row per (symbol, date)
       using the standard convention: open=first, high=max, low=min,
       close=last, volume=sum.
    3. Loads the historical raw OHLCV (`nifty100_ohlcv.parquet`) and concats
       the live daily rows (deduping on symbol+date, live wins).
    4. For each symbol, runs `build_symbol_features` over the full
       historical+live series (so SMA_200 / 252-day rolling / RSI-14 are
       warm) and slices out only the new rows.
    5. Runs `build_cross_sectional` over the new rows (cross-sectional
       features need every symbol on the same date).
    6. Concats the new rows into `nifty100_features.parquet`, sorts, and
       writes the result. The model and screening pipeline read from this
       parquet via `build_signal.py --use-live-features`.

Leakage safety:
  All features are computed using only past/present data for the symbol.
  The target column (`target_direction`) is left NaN for the new tail rows
  because the next session's close isn't known yet; the model never trains
  on these rows (it retrains nightly at 16:00 IST and only on closed
  sessions). The screener only consumes `prob_up` and the freshness fields;
  it never uses the target.

Usage:
    python -m scripts.refresh_live_features
    python -m scripts.refresh_live_features --ohlcv data/processed/nifty100_ohlcv.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from market_ml.feature_engineering import build_symbol_features, build_cross_sectional

OHLCV_PARQUET = ROOT / "data" / "processed" / "nifty100_ohlcv.parquet"
FEATURES_PARQUET = ROOT / "data" / "processed" / "nifty100_features.parquet"
LIVE_DIR = ROOT / "data" / "live" / "nse" / "ohlcv"
SOURCE_LIVE = "groww_live"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def aggregate_intraday_to_daily(live_dir: Path) -> pd.DataFrame:
    """Aggregate per-symbol intraday CSVs into one daily row per (symbol, date).

    Convention: open=first, high=max, low=min, close=last, volume=sum.
    Returns a DataFrame with columns [date, symbol, open, high, low, close,
    volume, source] matching the historical schema.
    """
    if not live_dir.exists():
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low",
                                     "close", "volume", "source"])
    files = sorted(p for p in live_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low",
                                     "close", "volume", "source"])
    out_rows = []
    for path in files:
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sym, date_part = parts
        d = (pd.to_datetime(date_part, errors="coerce", format="%Y-%m-%d")
             if "-" in date_part
             else pd.to_datetime(date_part, errors="coerce", format="%Y%m%d"))
        if pd.isna(d):
            continue
        try:
            cdf = pd.read_csv(path)
        except Exception:
            continue
        if cdf.empty or not {"open", "high", "low", "close", "volume"}.issubset(cdf.columns):
            continue
        out_rows.append({
            "date": pd.Timestamp(d).normalize(),
            "symbol": sym,
            "open": float(cdf["open"].iloc[0]),
            "high": float(cdf["high"].max()),
            "low": float(cdf["low"].min()),
            "close": float(cdf["close"].iloc[-1]),
            "volume": float(cdf["volume"].sum()),
            "source": SOURCE_LIVE,
        })
    if not out_rows:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low",
                                     "close", "volume", "source"])
    return pd.DataFrame(out_rows)


def refresh(ohlcv_path: Path, features_path: Path, live_dir: Path) -> dict:
    """Compute new feature rows from live data and append to features parquet.

    Returns a summary dict with row counts, dates, and per-symbol stats.
    """
    # Discover the historical schema so we can keep the live rows homogeneous.
    # `build_symbol_features` adds external columns (FII/DII, options,
    # sentiment, ...) when their data files exist on disk; the trained model
    # only knows the 62 base features, so we project new rows to the
    # historical column set before appending.
    if features_path.exists():
        schema_cols = list(pd.read_parquet(features_path).columns)
    else:
        schema_cols = None
    live_daily = aggregate_intraday_to_daily(live_dir)
    if live_daily.empty:
        log("no live OHLCV files; nothing to refresh")
        return {"refreshed_rows": 0, "live_dates": [], "symbols": 0}

    # Normalize to date-only to match the historical schema.
    live_daily["date"] = pd.to_datetime(live_daily["date"]).dt.normalize()
    log(f"aggregated {len(live_daily)} (symbol, day) live rows; "
        f"date range {live_daily['date'].min().date()}..{live_daily['date'].max().date()}")

    # Load historical OHLCV (raw daily, not features).
    hist = pd.read_parquet(ohlcv_path)
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    log(f"loaded historical OHLCV: {len(hist):,} rows, "
        f"through {hist['date'].max().date()}")

    # Identify the "new" dates: live dates strictly after the features
    # parquet's max. The features parquet may already include some live
    # dates from a prior refresh run, so its max date is the correct cutoff
    # (the OHLCV parquet's max stays frozen unless rebuilt). Falling back
    # to the OHLCV max is safe for a first-time run.
    if features_path.exists():
        try:
            existing = pd.read_parquet(features_path, columns=["date"])
            existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
            cutoff_date = existing["date"].max()
        except Exception:
            cutoff_date = hist["date"].max()
    else:
        cutoff_date = hist["date"].max()
    new_dates = sorted(live_daily.loc[live_daily["date"] > cutoff_date, "date"].unique())
    if not new_dates:
        log(f"no live dates are newer than the features parquet max ({cutoff_date.date()}); nothing to refresh")
        return {"refreshed_rows": 0, "live_dates": [], "symbols": 0}
    log(f"new dates to append: {[d.date() for d in new_dates]}")

    # Build the per-symbol feature rows for the new dates only.
    new_symbols = sorted(live_daily["symbol"].unique())
    log(f"computing features for {len(new_symbols)} symbols across "
        f"{len(new_dates)} new date(s)")

    new_frames = []
    skipped_no_history = []
    for sym in new_symbols:
        sym_live = (live_daily.loc[live_daily["symbol"] == sym]
                    .sort_values("date").reset_index(drop=True))
        sym_hist = (hist.loc[hist["symbol"] == sym]
                    .sort_values("date").reset_index(drop=True))
        if sym_hist.empty:
            # Symbol has live data but no history: skip. We cannot compute
            # SMA_200 / RSI-14 / etc. without a warmup period.
            skipped_no_history.append(sym)
            continue
        # Concat historical + live (deduped on date; live wins).
        combined = pd.concat([sym_hist, sym_live], ignore_index=True)
        combined = combined.drop_duplicates(subset="date", keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        # Per-symbol features over the full series.
        feats = build_symbol_features(combined)
        # Slice out only the new dates.
        mask = combined["date"].isin(new_dates)
        sliced = pd.concat([
            combined.loc[mask, ["date", "symbol", "open", "high", "low",
                                 "close", "volume", "source"]].reset_index(drop=True),
            feats.loc[mask].reset_index(drop=True),
        ], axis=1)
        new_frames.append(sliced)

    if not new_frames:
        log("no symbols had both live data and history; nothing to refresh")
        return {"refreshed_rows": 0, "live_dates": [d.date() for d in new_dates],
                "symbols": 0, "skipped": skipped_no_history}

    new_block = pd.concat(new_frames, ignore_index=True)
    log(f"computed per-symbol features for {len(new_block)} (symbol, day) rows")

    # Cross-sectional features need all symbols on the same date. The new
    # dates likely don't have every symbol in the universe (only those with
    # live files). To keep the math well-defined, we compute cross-sectional
    # ranks across the symbols that DO have new data on each date, and
    # document that limitation. The frozen historical cross-sectional values
    # for the *previous* date are not used (we don't have a "previous
    # cross-sectional" cached for the same set of symbols).
    cs = build_cross_sectional(new_block)
    if not cs.empty:
        new_block = pd.concat([new_block, cs.reset_index(drop=True)], axis=1)

    # The target column must be present (NaN for incomplete sessions) so the
    # schema matches the historical parquet.
    if "target_direction" not in new_block.columns:
        new_block["target_direction"] = np.nan

    # Project to the historical schema to keep the parquet homogeneous and
    # avoid breaking the trained model's feature list. Extra columns
    # (external features) are dropped; any historical columns that weren't
    # computed (e.g. cross-sectional for symbols absent on a date) become
    # NaN. The trained model uses median imputation, so this is safe.
    if schema_cols is not None:
        keep = list(schema_cols)  # preserve "date" too; it's a real column
        for c in keep:
            if c not in new_block.columns:
                new_block[c] = np.nan
        new_block = new_block[keep]

    # Drop the rows that are entirely NaN in the per-symbol features (none
    # expected, since we only take new dates; but defensive).
    per_symbol_feature_cols = [c for c in new_block.columns
                                if c not in {"date", "symbol", "open", "high",
                                             "low", "close", "volume", "source",
                                             "target_direction", "rank_return",
                                             "rank_volume_ratio",
                                             "excess_return", "rel_volatility"}]
    if per_symbol_feature_cols:
        all_nan = new_block[per_symbol_feature_cols].isna().all(axis=1)
        n_dropped = int(all_nan.sum())
        if n_dropped:
            log(f"dropping {n_dropped} all-NaN rows (insufficient warmup)")
            new_block = new_block.loc[~all_nan].reset_index(drop=True)

    # Append to the historical features parquet, deduped.
    feat_hist = pd.read_parquet(features_path)
    feat_hist["date"] = pd.to_datetime(feat_hist["date"]).dt.normalize()
    log(f"loaded historical features: {len(feat_hist):,} rows, "
        f"through {feat_hist['date'].max().date()}")

    combined = pd.concat([feat_hist, new_block], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Coerce numeric columns to float where possible (parquet schema drift
    # between live and historical is the usual cause).
    for c in combined.columns:
        if c in ("date", "symbol", "source"):
            continue
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    combined.to_parquet(features_path, index=False)
    log(f"wrote {features_path} ({len(combined):,} rows, "
        f"through {combined['date'].max().date()})")
    return {
        "refreshed_rows": int(len(new_block)),
        "live_dates": [d.date().isoformat() for d in new_dates],
        "symbols": int(new_block["symbol"].nunique()),
        "skipped": skipped_no_history,
        "features_path": str(features_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ohlcv", default=str(OHLCV_PARQUET),
                    help="Historical raw OHLCV parquet (daily).")
    ap.add_argument("--features", default=str(FEATURES_PARQUET),
                    help="Output features parquet (will be appended in place).")
    ap.add_argument("--live-dir", default=str(LIVE_DIR),
                    help="Directory of per-symbol intraday OHLCV CSVs.")
    args = ap.parse_args()
    summary = refresh(Path(args.ohlcv), Path(args.features), Path(args.live_dir))
    log(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
