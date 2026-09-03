"""NSE historical data loader backed by ``jugaad-data``.

Primary provider for NSE equity history. Downloads are cached as raw CSVs
under ``data/raw/nse/`` and normalized copies under ``data/interim/nse/``.

Normalized schema (one row per symbol-date):
    date, symbol, open, high, low, close, volume, source

The ``source`` column records which provider produced each row:
``jugaad-data``, ``yfinance``, or ``synthetic``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0


def load_config(config_path: str | Path = "config/data.yaml") -> dict:
    """Load the data collection config (symbols, dates, provider options)."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_symbols(config: dict) -> list[str]:
    """Read the symbol universe from the config's symbols_file CSV."""
    symbols_file = Path(config.get("symbols_file", "config/nifty100_symbols.csv"))
    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
    df = pd.read_csv(symbols_file)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    symbols = df[col].dropna().astype(str).str.strip().tolist()
    return [s for s in symbols if s]


def _normalize(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    """Normalize a provider DataFrame to the canonical schema."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    rename = {"mva": "close", "prev. close": "prev_close"}  # jugaad-data column names
    out = out.rename(columns=rename)
    missing = [c for c in NUMERIC_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Provider data for {symbol} missing columns {missing}")

    date_col = "date" if "date" in out.columns else None
    if date_col is None:
        idx = pd.to_datetime(out.index)
        out = out.reset_index(drop=True)
        out["date"] = idx.tz_localize(None) if idx.tz is not None else idx
    else:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        # timezone-naive datetimes only
        if getattr(out[date_col].dt, "tz", None) is not None:
            out[date_col] = out[date_col].dt.tz_localize(None)
    # jugaad-data DATE strings carry a misleading 18:30 time component; keep the calendar date only
    out["date"] = out["date"].dt.normalize()
    # Ensure consistent datetime64[ns] type for pyarrow parquet compatibility
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out["symbol"] = symbol
    out["source"] = source
    out = out[SCHEMA_COLUMNS + ["source"]]

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["symbol", "date"], kind="stable")

    # When multiple rows exist for the same (symbol, date) — e.g. jugaad-data
    # returning both a normal trading row and a low-volume adjustment row —
    # keep the one with the highest volume (the real trading session).
    out = out.sort_values("volume", ascending=False, kind="stable")
    out = out.drop_duplicates(subset=["symbol", "date"], keep="first")
    out = out.sort_values(["symbol", "date"], kind="stable")
    return out.reset_index(drop=True)


def download_nse_stock(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
    *,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    timeout: float | None = None,
    fallback_provider: str = "none",
    raw_dir: str | Path = "data/raw/nse",
    interim_dir: str | Path = "data/interim/nse",
) -> pd.DataFrame:
    """Download one NSE stock's history via jugaad-data and normalize it.

    Raw (as-returned) data is cached under ``raw_dir``; the normalized frame
    under ``interim_dir``. Raises on total failure unless an explicit
    fallback provider succeeds.
    """
    from jugaad_data.nse import stock_df

    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{symbol}.csv"
    interim_path = interim_dir / f"{symbol}.csv"

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "jugaad-data download %s %s..%s (attempt %d/%d)",
                symbol, start_date, end_date, attempt, retries,
            )
            df = stock_df(
                symbol=symbol,
                from_date=pd.Timestamp(start_date).date(),
                to_date=pd.Timestamp(end_date).date(),
                series="EQ",
            )
            if df is None or len(df) == 0:
                raise ValueError(f"jugaad-data returned no rows for {symbol}")
            raw_df = df.copy()
            raw_df.to_csv(raw_path, index=False)
            norm = _normalize(raw_df, symbol, source="jugaad-data")
            norm.to_csv(interim_path, index=False)
            return norm
        except Exception as exc:  # noqa: BLE001 - retry then fall back / raise
            last_exc = exc
            logger.warning(
                "jugaad-data attempt %d/%d failed for %s: %s",
                attempt, retries, symbol, exc,
            )
            if attempt < retries:
                time.sleep(retry_delay)

    if fallback_provider == "yfinance":
        logger.info("Falling back to yfinance for %s", symbol)
        try:
            from market_ml.download import fetch_ohlcv

            yf_df = fetch_ohlcv(symbol, start=start_date, end=end_date)
            yf_df = yf_df.reset_index()
            norm = _normalize(yf_df, symbol, source="yfinance")
            norm.to_csv(interim_path, index=False)
            return norm
        except Exception as exc:  # noqa: BLE001 - log and re-raise primary error
            logger.error("yfinance fallback also failed for %s: %s", symbol, exc)

    raise RuntimeError(
        f"All providers failed for {symbol} ({start_date}..{end_date}): {last_exc}"
    )


def ensure_nse_data(
    symbols: list[str],
    start_date: str,
    end_date: str | None = None,
    *,
    config: dict | None = None,
    raw_dir: str | Path = "data/raw/nse",
    interim_dir: str | Path = "data/interim/nse",
) -> dict[str, Path]:
    """Download/refresh normalized data for many symbols; return paths."""
    cfg = config or {}
    provider_cfg = cfg.get("provider", {})
    paths_cfg = cfg.get("paths", {})
    raw_dir = Path(paths_cfg.get("raw_dir", raw_dir))
    interim_dir = Path(paths_cfg.get("interim_dir", interim_dir))

    retries = int(provider_cfg.get("retries", DEFAULT_RETRIES))
    retry_delay = float(provider_cfg.get("retry_delay_seconds", DEFAULT_RETRY_DELAY))
    timeout = provider_cfg.get("timeout_seconds")
    fallback = str(provider_cfg.get("fallback_provider", "none")).lower()

    results: dict[str, Path] = {}
    failures: list[str] = []
    for i, sym in enumerate(symbols, 1):
        try:
            results[sym] = download_nse_stock(
                sym,
                start_date,
                end_date,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                fallback_provider=fallback,
                raw_dir=raw_dir,
                interim_dir=interim_dir,
            )
            print(f"[{i}/{len(symbols)}] {sym}: OK ({len(results[sym])} rows)")
        except Exception as exc:  # noqa: BLE001 - log failure, continue universe
            failures.append(sym)
            logger.error("[%d/%d] %s: FAILED (%s)", i, len(symbols), sym, exc)
            print(f"[{i}/{len(symbols)}] {sym}: FAILED ({exc})")

    if failures:
        logger.warning("Failed symbols: %s", failures)
    return results
