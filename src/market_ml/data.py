"""Data loading utilities.

Expects raw OHLCV CSVs in ``data/raw/`` with columns:
``date, open, high, low, close, volume``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV and return a DataFrame indexed by a DatetimeIndex."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def make_synthetic_ohlcv(n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for demos and tests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    log_ret = rng.normal(loc=0.0004, scale=0.012, size=n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n_days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.005, size=n_days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.005, size=n_days)))
    volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype(float)

    dates = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
