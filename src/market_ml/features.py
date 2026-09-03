"""Deterministic technical feature generation.

Implements returns, log returns, SMA/EMA, RSI, MACD, Bollinger Bands,
ATR, volume features and rolling volatility using pandas only
(no external TA dependency), so results are fully reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(100.0).where(avg_loss != 0, 100.0)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from an OHLCV DataFrame.

    Returns a DataFrame with the same DatetimeIndex as the input,
    containing only engineered feature columns.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    # Returns
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["log_ret_1d"] = np.log(close / close.shift(1))

    # Moving averages and price ratios
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    out["sma_20_ratio"] = close / sma_20 - 1.0
    out["sma_50_ratio"] = close / sma_50 - 1.0
    out["ema_20_ratio"] = close / ema_20 - 1.0
    out["sma_20_50_cross"] = sma_20 / sma_50 - 1.0

    # Momentum indicators
    out["rsi_14"] = compute_rsi(close, 14)
    macd_line, signal_line, hist = compute_macd(close)
    out["macd"] = macd_line / close  # normalise by price level
    out["macd_signal"] = signal_line / close
    out["macd_hist"] = hist / close

    # Bollinger Bands
    std_20 = close.rolling(20).std(ddof=0)
    upper = sma_20 + 2.0 * std_20
    lower = sma_20 - 2.0 * std_20
    out["bb_position"] = (close - lower) / (upper - lower).replace(0.0, np.nan)
    out["bb_width"] = (upper - lower) / sma_20

    # Volatility
    out["atr_14_ratio"] = compute_atr(df, 14) / close
    out["volatility_10"] = out["ret_1d"].rolling(10).std(ddof=0)
    out["volatility_20"] = out["ret_1d"].rolling(20).std(ddof=0)

    # Volume features
    vol_sma_20 = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_sma_20.replace(0.0, np.nan)
    prev_volume = volume.shift(1).replace(0.0, np.nan)
    out["volume_change"] = (volume - prev_volume) / prev_volume.abs()

    return out


def make_dataset(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and next-day direction label y.

    Label is 1 when the close ``horizon`` days ahead is higher than today's
    close, else 0. Rows containing NaNs are dropped so the output is model-ready.
    """
    from .labels import make_labels

    X = build_features(df)
    y = make_labels(df["close"], horizon=horizon)
    joined = X.join(y.rename("label")).replace([np.inf, -np.inf], np.nan).dropna()
    return joined.drop(columns=["label"]), joined["label"].astype(int)
