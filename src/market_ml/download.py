"""Download and cache real Nifty 100 OHLCV data.

Uses yfinance (public Yahoo Finance endpoints, no credentials). Data is
cached as CSVs under data/raw/ so the pipeline is reproducible offline
after the first download.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .data import REQUIRED_COLUMNS
from .universe import NIFTY100_SYMBOLS, yahoo_symbol


def fetch_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted OHLCV for one NSE symbol via yfinance."""
    import yfinance as yf

    ticker = yahoo_symbol(symbol)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,  # prefer adjusted OHLCV per data rules
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[REQUIRED_COLUMNS]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def ensure_raw_data(
    symbols: list[str] | None = None,
    start: str = "2019-01-01",
    raw_dir: str | Path = "data/raw",
    max_failures: int = 10,
) -> dict[str, Path]:
    """Download missing raw CSVs; return mapping symbol -> cached path."""
    symbols = symbols or NIFTY100_SYMBOLS
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    failures: list[str] = []
    for i, sym in enumerate(symbols):
        path = raw_dir / f"{sym}.csv"
        if path.exists() and path.stat().st_size > 0:
            paths[sym] = path
            continue
        try:
            df = fetch_ohlcv(sym, start)
            df.to_csv(path)
            paths[sym] = path
            print(f"[{i + 1}/{len(symbols)}] {sym}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001 - log and continue universe download
            failures.append(sym)
            print(f"[{i + 1}/{len(symbols)}] {sym}: FAILED ({exc})")
        time.sleep(0.3)  # be polite to the public endpoint

    if len(failures) > max_failures:
        raise RuntimeError(f"Too many failed downloads ({len(failures)}): {failures}")
    if failures:
        print(f"Skipped {len(failures)} unavailable symbols: {failures}")
    return paths
