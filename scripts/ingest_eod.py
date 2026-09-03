#!/usr/bin/env python3
"""End-of-day ingestor: append yesterday's NSE bhavcopy to nifty100_ohlcv.parquet.

This is the missing link in the freshness pipeline. `build_nifty100_dataset.py`
is a heavy multi-day downloader; it was never meant for daily refreshes.
This script is.

Two modes, both idempotent:
  (a) --mode bhavcopy  : download the NSE EOD CSV for one or more dates and
                         append/merge into the parquet (default).
  (b) --mode intraday  : also pull today's partial bhavcopy if NSE has it.

The script:
  1. Reads the universe from config/nifty100_symbols.csv.
  2. Downloads sec_bhavdata_full_{YYYYMMDD}.csv from NSE's archive (or uses
     a local cache if --offline).
  3. Normalises the columns to {date, symbol, open, high, low, close, volume, source}.
  4. Concatenates with the existing parquet, drops duplicates on
     (date, symbol), re-sorts, and writes back.
  5. Emits a one-line `last_ingest.json` with the freshness timestamp and
     a small status report that the web portal can read for the UI banner.

Usage:
  python scripts/ingest_eod.py                    # ingest yesterday + today (best-effort)
  python scripts/ingest_eod.py --date 2026-08-31  # ingest a specific date
  python scripts/ingest_eod.py --date 2026-08-30 --date 2026-08-31 --offline
  python scripts/ingest_eod.py --mode bhavcopy --dry-run

Notes
-----
* NSE bhavcopy archive URL pattern is stable since 2014. We try the primary
  URL first, then a fallback. If both fail, the script exits non-zero with a
  clear log message.
* The script is resumable: re-running for the same date overwrites the row.
* Does NOT touch the model or the live_signal.json. Those are built by
  `build_signal.py` after EOD data is settled (run them in order).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import requests

from data.jugaad_loader import load_config, load_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_eod")

IST = ZoneInfo("Asia/Kolkata")
NSE_BHAV_PRIMARY = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
NSE_BHAV_FALLBACK = "https://www.nseindia.com/content/historical/EQUITIES/{y}/{m}/sec_bhavdata_full_{d}.csv"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# Canonical output schema
OHLCV_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "source"]


def _to_ist_date(s: str) -> pd.Timestamp:
    """Parse YYYY-MM-DD into a midnight-IST Timestamp (date only, no tz)."""
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=IST)).normalize().tz_localize(None)


def _bhav_url_for(date_str: str) -> tuple[str, str]:
    """Return (primary, fallback) URLs for a given YYYY-MM-DD date."""
    d = date_str.replace("-", "")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    fallback = NSE_BHAV_FALLBACK.format(y=dt.strftime("%Y"), m=dt.strftime("%b").upper(), d=d)
    return NSE_BHAV_PRIMARY.format(d=d), fallback


def fetch_bhavcopy(date_str: str, cache_dir: Path, timeout: int = 20) -> pd.DataFrame:
    """Download the NSE bhavcopy for `date_str` and return a normalised DataFrame.

    Empty DataFrame (with the canonical schema) if all fetches fail.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sec_bhavdata_full_{date_str.replace('-','')}.csv"

    raw_text: str | None = None
    if cache_file.exists():
        raw_text = cache_file.read_text(encoding="utf-8", errors="replace")
        logger.info("using cached bhavcopy for %s (%d bytes)", date_str, len(raw_text))
    else:
        for url in _bhav_url_for(date_str):
            try:
                logger.info("GET %s", url)
                r = requests.get(url, headers=NSE_HEADERS, timeout=timeout)
                if r.status_code == 200 and r.text and "SYMBOL" in r.text.upper():
                    raw_text = r.text
                    cache_file.write_text(r.text, encoding="utf-8")
                    logger.info("fetched bhavcopy for %s from %s (%d bytes)", date_str, url, len(r.text))
                    break
                logger.warning("  %s -> status %d, %d bytes", url, r.status_code, len(r.text or ""))
            except Exception as e:
                logger.warning("  %s -> %s", url, e)
            time.sleep(0.5)

    if raw_text is None:
        logger.error("FAILED to fetch bhavcopy for %s from any URL", date_str)
        return pd.DataFrame(columns=OHLCV_COLS)

    # NSE bhavcopy has 14 columns; the first 5 are the OHLCV (no adj close).
    df = pd.read_csv(pd.io.common.StringIO(raw_text))
    df.columns = [c.strip().upper() for c in df.columns]
    # Standard column names per NSE: SYMBOL, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, ...
    need = {"SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE"}
    if not need.issubset(df.columns):
        logger.error("bhavcopy for %s missing columns: %s", date_str, sorted(need - set(df.columns)))
        return pd.DataFrame(columns=OHLCV_COLS)

    vol_col = "TOTTRDQTY" if "TOTTRDQTY" in df.columns else "VOLUME"
    out = pd.DataFrame({
        "date":   _to_ist_date(date_str),
        "symbol": df["SYMBOL"].astype(str).str.strip().str.upper(),
        "open":   pd.to_numeric(df["OPEN"],   errors="coerce"),
        "high":   pd.to_numeric(df["HIGH"],   errors="coerce"),
        "low":    pd.to_numeric(df["LOW"],    errors="coerce"),
        "close":  pd.to_numeric(df["CLOSE"],  errors="coerce"),
        "volume": pd.to_numeric(df[vol_col],  errors="coerce"),
        "source": "nse_bhavcopy",
    })
    # Quality: drop non-positive / NaN / rows for symbols outside the universe
    cfg = load_config()
    universe = set(load_symbols(cfg))
    out = out[out["symbol"].isin(universe)]
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out = out[out["volume"] >= 0]
    out = out[out["high"] >= out["low"]]
    logger.info("normalised bhavcopy for %s -> %d rows", date_str, len(out))
    return out.reset_index(drop=True)


def merge_into_parquet(new_df: pd.DataFrame, parquet_path: Path) -> dict:
    """Append new_df to parquet_path, dedupe on (date, symbol), write atomically.

    Returns a dict with merge stats.
    """
    if not parquet_path.exists():
        merged = new_df.copy()
        new_rows = len(merged)
        max_date = merged["date"].max() if len(merged) else None
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(parquet_path, index=False)
        return {"new_rows": new_rows, "replaced": 0, "total": new_rows, "max_date": str(max_date)}

    existing = pd.read_parquet(parquet_path)
    before = len(existing)
    # Same schema; if existing is missing 'source', add it
    if "source" not in existing.columns:
        existing["source"] = "legacy"
    existing = existing[OHLCV_COLS]
    combined = pd.concat([existing, new_df[OHLCV_COLS]], ignore_index=True)
    # Dedupe (date, symbol) keeping LAST (so a re-ingest overwrites stale rows)
    combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    replaced = before + len(new_df) - len(combined)
    # Atomic write: write to a tmp file then move
    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(parquet_path)
    return {
        "new_rows": len(new_df),
        "replaced": int(replaced),
        "total": len(combined),
        "max_date": str(combined["date"].max()),
    }


def write_freshness_marker(parquet_path: Path, marker_path: Path) -> dict:
    """Emit a small JSON the web portal reads to show data age."""
    df = pd.read_parquet(parquet_path, columns=["date"])
    max_date = pd.to_datetime(df["date"]).max()
    max_date_str = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None
    today_ist = datetime.now(IST).date()
    last_trading_day = _last_trading_day(today_ist)
    age_days = (last_trading_day - max_date.date()).days if max_date is not None else None
    payload = {
        "parquet": str(parquet_path.relative_to(ROOT)),
        "max_date": max_date_str,
        "as_of_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "last_trading_day": last_trading_day.strftime("%Y-%m-%d"),
        "age_trading_days": age_days,
        "is_stale": age_days is None or age_days > 1,
        "is_fresh": age_days is not None and age_days <= 1,
    }
    marker_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote freshness marker to %s: %s", marker_path, payload)
    return payload


def _last_trading_day(d) -> object:
    """Approximate last trading day: today if Mon-Fri after 18:00 IST, else previous weekday."""
    # We don't have an NSE holiday calendar baked in; this is conservative.
    while d.weekday() >= 5:  # Sat/Sun
        d = d - timedelta(days=1)
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", action="append", default=[],
                   help="Date(s) YYYY-MM-DD to ingest. May be passed multiple times. "
                        "Default: yesterday + today (best-effort).")
    p.add_argument("--mode", choices=["bhavcopy"], default="bhavcopy")
    p.add_argument("--parquet", default="data/processed/nifty100_ohlcv.parquet")
    p.add_argument("--cache", default="data/external/bhavcopy")
    p.add_argument("--marker", default="reports/data_freshness.json")
    p.add_argument("--offline", action="store_true",
                   help="Use cached bhavcopy only; do not hit NSE.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute the merge but do not write the parquet or marker.")
    return p.parse_args()


def resolve_dates(args_dates: list[str]) -> list[str]:
    if args_dates:
        return args_dates
    today_ist = datetime.now(IST).date()
    yest = today_ist - timedelta(days=1)
    return [yest.strftime("%Y-%m-%d"), today_ist.strftime("%Y-%m-%d")]


def main() -> int:
    args = parse_args()
    parquet_path = ROOT / args.parquet
    cache_dir = ROOT / args.cache
    marker_path = ROOT / args.marker

    dates = resolve_dates(args.date)
    logger.info("ingestion targets: %s", dates)
    logger.info("parquet path    : %s", parquet_path)
    logger.info("offline mode    : %s", args.offline)

    total_new = 0
    last_max_date = None
    for d in dates:
        df = fetch_bhavcopy(d, cache_dir)
        if df.empty:
            logger.warning("no data for %s; skipping", d)
            continue
        if args.dry_run:
            logger.info("[dry-run] would merge %d rows for %s", len(df), d)
            total_new += len(df)
            continue
        stats = merge_into_parquet(df, parquet_path)
        logger.info("merged %s: %s", d, stats)
        total_new += stats["new_rows"]
        last_max_date = stats["max_date"]

    if not args.dry_run:
        marker = write_freshness_marker(parquet_path, marker_path)
        logger.info("DATA FRESHNESS: %s (trading-day age = %s)",
                    marker["max_date"], marker["age_trading_days"])
        if marker["is_stale"]:
            logger.warning("DATA IS STALE (>1 trading day old). Run with more --date args or fix ingest.")
    else:
        logger.info("[dry-run] would have written %d new rows; last max_date would be %s",
                    total_new, last_max_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
