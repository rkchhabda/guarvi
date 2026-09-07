"""Leakage-safe feature engineering for Nifty 100 OHLCV data.

All features are computed per-symbol using only data available on or before
date t. No future values are used in any computation.

Features implemented:
  - Price/return: pct returns (1,2,3,5,10,20d), log returns, high-low range,
    close-open return, gap return
  - Trend: SMA 5/10/20/50/200, EMA 12/26, distance from MAs, crossover flags
  - Momentum: RSI 14, MACD/signal/histogram, Stochastic, ADX
  - Volatility: rolling std 5/10/20/50d, ATR 14, Bollinger Bands
  - Volume: pct change, avg volume 5/10/20d, volume ratio, OBV
  - Cross-sectional: daily rank of return, daily rank of volume ratio,
    stock vol vs universe vol
  - External: FII/DII flows, NSE options (IV, PCR), delivery %, news sentiment,
    fundamentals/earnings, intraday microstructure
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# External data paths
EXTERNAL_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "external"


# ---------------------------------------------------------------------------
# Per-symbol feature computation (no cross-symbol contamination)
# ---------------------------------------------------------------------------

def _compute_returns(close: pd.Series) -> pd.DataFrame:
    """Percentage and log returns at multiple horizons."""
    out = pd.DataFrame(index=close.index)
    for d in (1, 2, 3, 5, 10, 20):
        out[f"ret_{d}d"] = close.pct_change(d)
        out[f"log_ret_{d}d"] = np.log(close / close.shift(d))
    # close-open return
    return out


def _compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """High-low range, close-open, gap return."""
    out = pd.DataFrame(index=df.index)
    out["hl_range"] = (df["high"] - df["low"]) / df["close"]
    out["co_return"] = (df["close"] - df["open"]) / df["open"]
    out["gap_return"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    return out


def _compute_trend(close: pd.Series) -> pd.DataFrame:
    """SMA, EMA, distance from MAs, crossover flags."""
    out = pd.DataFrame(index=close.index)
    smas = {}
    for w in (5, 10, 20, 50, 200):
        sma = close.rolling(w, min_periods=w).mean()
        smas[w] = sma
        out[f"sma_{w}"] = sma
        out[f"dist_sma_{w}"] = (close - sma) / sma

    for w in (12, 26):
        ema = close.ewm(span=w, adjust=False).mean()
        out[f"ema_{w}"] = ema
        out[f"dist_ema_{w}"] = (close - ema) / ema

    # Crossover flags (1 when shorter MA > longer MA)
    out["sma_5_10_cross"] = (smas[5] > smas[10]).astype(float)
    out["sma_10_20_cross"] = (smas[10] > smas[20]).astype(float)
    out["sma_20_50_cross"] = (smas[20] > smas[50]).astype(float)
    out["sma_50_200_cross"] = (smas[50] > smas[200]).astype(float)
    return out


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(100.0).where(avg_loss != 0, 100.0)


def _compute_macd(close: pd.Series) -> pd.DataFrame:
    """MACD line, signal, histogram."""
    out = pd.DataFrame(index=close.index)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    # Normalise by price to make cross-symbol comparable
    out["macd"] = macd_line / close
    out["macd_signal"] = signal_line / close
    out["macd_hist"] = hist / close
    return out


def _compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic oscillator %K and %D."""
    out = pd.DataFrame(index=df.index)
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - low_min) / denom
    out["stoch_k"] = raw_k.clip(0.0, 100.0)
    out["stoch_d"] = out["stoch_k"].rolling(d_period, min_periods=d_period).mean()
    return out


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) and directional indicators."""
    out = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = (high - prev_high).clip(lower=0.0)
    minus_dm = (prev_low - low).clip(lower=0.0)
    # Zero out whichever is smaller
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    out["adx"] = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    return out


def _compute_volatility(close: pd.Series, ret_1d: pd.Series) -> pd.DataFrame:
    """Rolling standard deviation and ATR."""
    out = pd.DataFrame(index=close.index)
    for w in (5, 10, 20, 50):
        out[f"volatility_{w}d"] = ret_1d.rolling(w, min_periods=w).std(ddof=0)
    return out


def _compute_atr(df: pd.DataFrame) -> pd.Series:
    """Average True Range 14."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()


def _compute_bollinger(close: pd.Series) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, position, width."""
    out = pd.DataFrame(index=close.index)
    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std(ddof=0)
    upper = sma_20 + 2.0 * std_20
    lower = sma_20 - 2.0 * std_20
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    band_width = (upper - lower).replace(0.0, np.nan)
    out["bb_position"] = (close - lower) / band_width
    out["bb_width"] = band_width / sma_20
    return out


def _compute_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Volume features: pct change, avg volume, ratio, OBV."""
    out = pd.DataFrame(index=df.index)
    volume = df["volume"]
    close = df["close"]

    # Volume percentage change
    prev_vol = volume.shift(1).replace(0.0, np.nan)
    out["volume_pct_change"] = (volume - prev_vol) / prev_vol.abs()

    # Average volume
    for w in (5, 10, 20):
        out[f"avg_volume_{w}d"] = volume.rolling(w, min_periods=w).mean()

    # Volume ratio vs 20-day average
    vol_sma_20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio_20d"] = volume / vol_sma_20.replace(0.0, np.nan)

    # On-Balance Volume (OBV)
    direction = np.sign(close.diff()).fillna(0.0)
    out["obv"] = (direction * volume).cumsum()
    # Normalise OBV by its own 20-day std for comparability
    obv_std = out["obv"].rolling(20, min_periods=20).std(ddof=0)
    out["obv_normalised"] = out["obv"] / obv_std.replace(0.0, np.nan)

    return out


def _load_external_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load and merge external data features for a symbol.
    
    Loads pre-computed external features from parquet files and aligns
    them to the symbol's date index using forward-fill (leakage-safe).
    """
    external_feats = pd.DataFrame(index=date_index)
    
    # FII/DII flows (market-wide, same for all symbols)
    fii_dii_path = EXTERNAL_DATA_DIR / "fii_dii_features.parquet"
    if fii_dii_path.exists():
        try:
            fii_dii = pd.read_parquet(fii_dii_path)
            fii_dii.index = pd.to_datetime(fii_dii.index)
            # Forward fill to align with symbol dates (leakage-safe)
            fii_dii_aligned = fii_dii.reindex(date_index, method="ffill")
            external_feats = pd.concat([external_feats, fii_dii_aligned], axis=1)
        except Exception as e:
            logger.warning(f"Failed to load FII/DII features: {e}")
    
    # Delivery % and bulk/block deals (per symbol)
    delivery_path = EXTERNAL_DATA_DIR / "delivery_features.parquet"
    if delivery_path.exists():
        try:
            delivery = pd.read_parquet(delivery_path)
            # Expect multi-index (date, symbol) or columns with symbol
            if "symbol" in delivery.columns:
                sym_delivery = delivery[delivery["symbol"] == symbol].set_index("date")
            elif delivery.index.nlevels == 2:
                sym_delivery = delivery.xs(symbol, level="symbol", drop_level=False)
                sym_delivery.index = sym_delivery.index.droplevel("symbol")
            else:
                sym_delivery = delivery
            sym_delivery.index = pd.to_datetime(sym_delivery.index)
            sym_delivery_aligned = sym_delivery.reindex(date_index, method="ffill")
            external_feats = pd.concat([external_feats, sym_delivery_aligned], axis=1)
        except Exception as e:
            logger.warning(f"Failed to load delivery features for {symbol}: {e}")
    
    # Intraday microstructure (per symbol)
    intraday_path = EXTERNAL_DATA_DIR / "intraday_microstructure_features.parquet"
    if intraday_path.exists():
        try:
            intraday = pd.read_parquet(intraday_path)
            if "symbol" in intraday.columns:
                sym_intraday = intraday[intraday["symbol"] == symbol].set_index("date")
            elif intraday.index.nlevels == 2:
                sym_intraday = intraday.xs(symbol, level="symbol", drop_level=False)
                sym_intraday.index = sym_intraday.index.droplevel("symbol")
            else:
                sym_intraday = intraday
            sym_intraday.index = pd.to_datetime(sym_intraday.index)
            sym_intraday_aligned = sym_intraday.reindex(date_index, method="ffill")
            external_feats = pd.concat([external_feats, sym_intraday_aligned], axis=1)
        except Exception as e:
            logger.warning(f"Failed to load intraday features for {symbol}: {e}")
    
    # NSE Options data (per symbol)
    options_path = EXTERNAL_DATA_DIR / "nse_options_features.parquet"
    if options_path.exists():
        try:
            options = pd.read_parquet(options_path)
            if "symbol" in options.columns:
                sym_options = options[options["symbol"] == symbol].set_index("date")
            elif options.index.nlevels == 2:
                sym_options = options.xs(symbol, level="symbol", drop_level=False)
                sym_options.index = sym_options.index.droplevel("symbol")
            else:
                sym_options = options
            sym_options.index = pd.to_datetime(sym_options.index)
            sym_options_aligned = sym_options.reindex(date_index, method="ffill")
            external_feats = pd.concat([external_feats, sym_options_aligned], axis=1)
        except Exception as e:
            logger.warning(f"Failed to load options features for {symbol}: {e}")
    
    # Fundamentals/Earnings (per symbol)
    fundamentals_path = EXTERNAL_DATA_DIR / "fundamentals_earnings_features.parquet"
    if fundamentals_path.exists():
        try:
            fund = pd.read_parquet(fundamentals_path)
            if "symbol" in fund.columns:
                sym_fund = fund[fund["symbol"] == symbol].set_index("date")
            elif fund.index.nlevels == 2:
                sym_fund = fund.xs(symbol, level="symbol", drop_level=False)
                sym_fund.index = sym_fund.index.droplevel("symbol")
            else:
                sym_fund = fund
            sym_fund.index = pd.to_datetime(sym_fund.index)
            sym_fund_aligned = sym_fund.reindex(date_index, method="ffill")
            external_feats = pd.concat([external_feats, sym_fund_aligned], axis=1)
        except Exception as e:
            logger.warning(f"Failed to load fundamentals features for {symbol}: {e}")
    
    # News Sentiment (per symbol + sector + market)
    news_path = EXTERNAL_DATA_DIR / "news_sentiment_features.parquet"
    if news_path.exists():
        try:
            news = pd.read_parquet(news_path)
            if "symbol" in news.columns:
                sym_news = news[news["symbol"] == symbol].set_index("date")
            elif news.index.nlevels == 2:
                sym_news = news.xs(symbol, level="symbol", drop_level=False)
                sym_news.index = sym_news.index.droplevel("symbol")
            else:
                sym_news = news
            sym_news.index = pd.to_datetime(sym_news.index)
            sym_news_aligned = sym_news.reindex(date_index, method="ffill")
            external_feats = pd.concat([external_feats, sym_news_aligned], axis=1)
        except Exception as e:
            logger.warning(f"Failed to load news sentiment features for {symbol}: {e}")
    
    return external_feats


def build_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features for a single symbol.

    Parameters
    ----------
    df : DataFrame with columns [date, symbol, open, high, low, close, volume, source]
        sorted by date ascending for this symbol.

    Returns
    -------
    DataFrame with original columns + all engineered features, indexed by row
    (not date) for easy merging.
    """
    close = df["close"]
    ret_1d = close.pct_change(1)
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else None
    date_index = pd.to_datetime(df["date"])

    parts = [
        _compute_returns(close),
        _compute_price_features(df),
        _compute_trend(close),
        pd.DataFrame({"rsi_14": _compute_rsi(close, 14)}, index=df.index),
        _compute_macd(close),
        _compute_stochastic(df),
        _compute_adx(df),
        _compute_volatility(close, ret_1d),
        pd.DataFrame({"atr_14": _compute_atr(df)}, index=df.index),
        _compute_bollinger(close),
        _compute_volume(df),
    ]

    features = pd.concat(parts, axis=1)
    
    # ===== External Data Integration =====
    if symbol is not None:
        external_feats = _load_external_features(symbol, date_index)
        if not external_feats.empty:
            # Align external features to the feature index
            external_feats.index = features.index
            features = pd.concat([features, external_feats], axis=1)
    
    # ===== PHASE 1: Enhanced Momentum Features (Quick Wins) =====
    # RSI momentum: rate of change of RSI
    if "rsi_14" in features.columns:
        features["rsi_momentum"] = features["rsi_14"].diff()
        features["rsi_overbought"] = (features["rsi_14"] > 70).astype(float)
        features["rsi_oversold"] = (features["rsi_14"] < 30).astype(float)
    
    # MACD momentum: change in MACD histogram
    if "macd_hist" in features.columns:
        features["macd_momentum"] = features["macd_hist"].diff()
    
    # Trend strength: difference between short and long-term trend
    if "sma_20_ratio" in features.columns and "sma_50_ratio" in features.columns:
        features["trend_strength"] = features["sma_20_ratio"] - features["sma_50_ratio"]
    
    # Volume spike indicator
    if "volume_ratio_20d" in features.columns:
        features["volume_spike"] = (features["volume_ratio_20d"] > 2.0).astype(float)
        features["volume_momentum"] = features["volume_ratio_20d"].diff()
    
    # Mean reversion indicator
    if "ret_1d" in features.columns and "ret_5d" in features.columns:
        features["mean_reversion"] = features["ret_1d"] - features["ret_5d"] / 5.0
    
    # ADX strength
    if "adx" in features.columns:
        features["adx_strong"] = (features["adx"] > 25).astype(float)
    
    # ===== PHASE 1C: Advanced Technical Features =====
    
    # Bollinger Band position & squeeze
    if "bb_position" in features.columns and "bb_width" in features.columns:
        features["bb_squeeze"] = (features["bb_width"] < features["bb_width"].rolling(20, min_periods=20).quantile(0.1)).astype(float)
        features["bb_breakout_up"] = (features["bb_position"] > 0.95).astype(float)
        features["bb_breakout_dn"] = (features["bb_position"] < 0.05).astype(float)
    
    # Stochastic crossover & extremes
    if "stoch_k" in features.columns and "stoch_d" in features.columns:
        features["stoch_cross"] = (features["stoch_k"] > features["stoch_d"]).astype(float)
        features["stoch_extreme_high"] = (features["stoch_k"] > 80).astype(float)
        features["stoch_extreme_low"] = (features["stoch_k"] < 20).astype(float)
    
    # MACD signal line crossover
    if "macd" in features.columns and "macd_signal" in features.columns:
        features["macd_cross"] = (features["macd"] > features["macd_signal"]).astype(float)
    
    # Volatility regime
    if "volatility_5d" in features.columns and "volatility_20d" in features.columns:
        features["vol_regime"] = features["volatility_5d"] / features["volatility_20d"].replace(0.0, np.nan)
        features["high_vol_regime"] = (features["vol_regime"] > 1.5).astype(float)
        features["low_vol_regime"] = (features["vol_regime"] < 0.5).astype(float)
    
    # Price acceleration (momentum of momentum)
    if "ret_5d" in features.columns:
        features["price_accel"] = features["ret_5d"].diff()
        features["ret_5d_ma"] = features["ret_5d"].rolling(5, min_periods=5).mean()
    
    # Multi-timeframe trend alignment
    if "dist_sma_5" in features.columns and "dist_sma_20" in features.columns and "dist_sma_50" in features.columns:
        features["trend_alignment"] = (
            (features["dist_sma_5"] > 0).astype(float) +
            (features["dist_sma_20"] > 0).astype(float) +
            (features["dist_sma_50"] > 0).astype(float)
        ) / 3.0
    
    # Distance from 52-week high/low
    if "close" in df.columns:
        rolling_max_252 = df["close"].rolling(252, min_periods=252).max()
        rolling_min_252 = df["close"].rolling(252, min_periods=252).min()
        features["dist_52w_high"] = (df["close"] - rolling_max_252) / rolling_max_252
        features["dist_52w_low"] = (df["close"] - rolling_min_252) / rolling_min_252
        features["pct_52w_range"] = (df["close"] - rolling_min_252) / (rolling_max_252 - rolling_min_252).replace(0.0, np.nan)
    
    return features


def build_target(close: pd.Series) -> pd.Series:
    """Create binary target: 1 if next session close > current close, else 0."""
    future_close = close.shift(-1)
    target = (future_close > close).astype(float)
    target[future_close.isna()] = np.nan
    return target.rename("target_direction")


def build_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional features (daily rank of return, volume ratio,
    stock volatility vs universe volatility).

    Must be called on the full multi-symbol DataFrame after per-symbol
    features are computed.
    """
    out = pd.DataFrame(index=df.index)

    # Daily rank of return within universe (percentile 0-1)
    if "ret_1d" in df.columns:
        out["rank_return"] = df.groupby("date")["ret_1d"].rank(pct=True)

    # Daily rank of volume ratio
    if "volume_ratio_20d" in df.columns:
        out["rank_volume_ratio"] = df.groupby("date")["volume_ratio_20d"].rank(pct=True)

    # Stock return minus universe mean return (if ret_1d exists)
    if "ret_1d" in df.columns:
        universe_mean_ret = df.groupby("date")["ret_1d"].transform("mean")
        out["excess_return"] = df["ret_1d"] - universe_mean_ret

    # Stock volatility relative to universe volatility
    if "volatility_20d" in df.columns:
        universe_vol = df.groupby("date")["volatility_20d"].transform("mean")
        out["rel_volatility"] = df["volatility_20d"] / universe_vol.replace(0.0, np.nan)

    return out


def ensure_dataframe(obj: object, context: str = "feature build") -> pd.DataFrame:
    """Safely convert a build result into a pandas DataFrame.

    This prevents deployment-time ``dict.to_parquet()`` errors when the builder
    returns a metadata dict instead of the raw table. For dict results, we prefer
    the DataFrame that looks like the actual feature table (contains symbol/date
    columns), not metadata tables such as removal_log or symbol_stats.
    """
    print(f"{context}: TYPE: {type(obj)}")
    if isinstance(obj, pd.DataFrame):
        features = obj
    elif isinstance(obj, dict):
        print(f"{context}: KEYS: {list(obj.keys())}")
        candidates = []
        preferred_order = ["features", "data", "df", "output", "table", "frame"]
        for key in preferred_order:
            if key in obj and isinstance(obj[key], pd.DataFrame):
                candidates.append((key, obj[key]))
        for key, value in obj.items():
            if isinstance(value, pd.DataFrame) and (key not in preferred_order):
                candidates.append((key, value))

        features = None
        for key, candidate in candidates:
            if candidate.empty:
                continue
            required_columns = ["symbol", "date"]
            missing = [c for c in required_columns if c not in candidate.columns]
            if not missing:
                features = candidate
                print(f"{context}: Using dataframe from key={key}")
                break

        if features is None:
            raise TypeError(
                f"{context}: No DataFrame found in returned dictionary with required columns. "
                f"Keys={list(obj.keys())}"
            )
    else:
        raise TypeError(f"{context}: Expected DataFrame or dict, got {type(obj)}")

    if features.empty:
        raise ValueError(f"{context}: Feature dataframe is empty.")

    required_columns = ["symbol", "date"]
    missing = [c for c in required_columns if c not in features.columns]
    if missing:
        raise ValueError(f"{context}: Missing columns: {missing}")

    print(f"{context}: Feature shape: {features.shape}")
    return features


def extract_feature_dataframe(result: object, context: str = "feature build") -> pd.DataFrame:
    """Compatibility wrapper for older call sites."""
    return ensure_dataframe(result, context=context)


def build_full_dataset(
    ohlcv_path: str,
    output_path: str = "data/processed/nifty100_features",
    min_rows_for_features: int = 200,
) -> dict:
    """Build feature dataset from frozen OHLCV parquet.

    Returns a dict with metadata for reporting.
    """
    from pathlib import Path

    df = pd.read_parquet(ohlcv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    all_frames = []
    removal_log = []  # (symbol, reason, count)

    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # --- Per-symbol features ---
        feats = build_symbol_features(grp)

        # --- Target ---
        target = build_target(grp["close"])

        # --- Assemble per-symbol frame ---
        sym_df = pd.concat([
            grp[["date", "symbol", "open", "high", "low", "close", "volume", "source"]].reset_index(drop=True),
            feats.reset_index(drop=True),
            target.reset_index(drop=True),
        ], axis=1)

        # Track rows before cleaning
        n_before = len(sym_df)

        # Remove rows where all feature columns are NaN (warm-up period)
        feature_cols = [c for c in feats.columns if c not in sym_df.columns[:8]]
        all_nan_mask = sym_df[feature_cols].isna().all(axis=1)
        n_warmup = all_nan_mask.sum()
        if n_warmup > 0:
            removal_log.append((sym, "warmup_period_all_nan", int(n_warmup)))

        # Keep all rows but leave NaN for warm-up; only drop rows where
        # critical features cannot ever be computed (e.g. first 199 rows for
        # SMA_200)
        sym_df = sym_df[~all_nan_mask].reset_index(drop=True)

        # Remove rows with inf in any feature
        inf_mask = np.isinf(sym_df[feature_cols].values).any(axis=1)
        n_inf = inf_mask.sum()
        if n_inf > 0:
            removal_log.append((sym, "infinite_feature_values", int(n_inf)))
            sym_df = sym_df[~inf_mask].reset_index(drop=True)

        all_frames.append(sym_df)

    combined = pd.concat(all_frames, ignore_index=True)

    # --- Cross-sectional features ---
    cross_feats = build_cross_sectional(combined)
    combined = pd.concat([combined, cross_feats.reset_index(drop=True)], axis=1)

    # --- Final cleaning ---
    # Replace inf with NaN
    feature_cols_all = [c for c in combined.columns if c not in
                        ["date", "symbol", "open", "high", "low", "close",
                         "volume", "source", "target_direction"]]
    combined[feature_cols_all] = combined[feature_cols_all].replace([np.inf, -np.inf], np.nan)

    # --- Compute feature schema ---
    # Core features (always required)
    feature_schema = {
            "price_return_features": [f"ret_{d}d" for d in (1, 2, 3, 5, 10, 20)] +
                                     [f"log_ret_{d}d" for d in (1, 2, 3, 5, 10, 20)] +
                                     ["hl_range", "co_return", "gap_return"],
            "trend_features": [f"sma_{w}" for w in (5, 10, 20, 50, 200)] +
                              [f"dist_sma_{w}" for w in (5, 10, 20, 50, 200)] +
                              [f"ema_{w}" for w in (12, 26)] +
                              [f"dist_ema_{w}" for w in (12, 26)] +
                              ["sma_5_10_cross", "sma_10_20_cross", "sma_20_50_cross", "sma_50_200_cross"],
            "momentum_features": ["rsi_14", "macd", "macd_signal", "macd_hist",
                                  "stoch_k", "stoch_d", "adx", "plus_di", "minus_di"],
            "volatility_features": [f"volatility_{w}d" for w in (5, 10, 20, 50)] +
                                   ["atr_14", "bb_upper", "bb_lower", "bb_position", "bb_width"],
            "volume_features": ["volume_pct_change", "avg_volume_5d", "avg_volume_10d",
                                "avg_volume_20d", "volume_ratio_20d", "obv", "obv_normalised"],
            "cross_sectional_features": ["rank_return", "rank_volume_ratio",
                                         "excess_return", "rel_volatility"],
            "target": ["target_direction"],
        }

    # External features (optional - only validated if data files exist)
    external_feature_categories = {
        "fii_dii_features": ["fii_net_buy", "dii_net_buy", "fii_net_buy_5d_ma", "fii_net_buy_20d_ma",
                             "dii_net_buy_5d_ma", "dii_net_buy_20d_ma", "fii_dii_spread",
                             "fii_net_buy_z", "dii_net_buy_z", "fii_dii_spread_z",
                             "fii_momentum_5d", "fii_momentum_20d", "dii_momentum_5d", "dii_momentum_20d",
                             "fii_dii_corr_20d", "fii_net_buy_ma_ratio", "dii_net_buy_ma_ratio"],
        "delivery_features": ["delivery_pct", "delivery_pct_5d_ma", "delivery_pct_20d_ma",
                              "delivery_pct_z", "high_delivery_flag", "bulk_deal_flag",
                              "block_deal_flag", "bulk_deal_value", "block_deal_value",
                              "delivery_momentum_5d", "delivery_momentum_20d", "delivery_trend_5d",
                              "delivery_trend_20d", "bulk_deal_count_20d", "block_deal_count_20d"],
        "intraday_microstructure_features": ["vwap_dev", "realized_vol", "ofi", "roll_spread",
                                             "kyle_lambda", "amihud_illiq", "vwap_dev_5d_ma",
                                             "realized_vol_5d_ma", "ofi_5d_ma", "roll_spread_5d_ma",
                                             "kyle_lambda_5d_ma", "amihud_illiq_5d_ma",
                                             "vwap_dev_z", "realized_vol_z", "ofi_z",
                                             "roll_spread_z", "kyle_lambda_z", "amihud_illiq_z"],
        "options_features": ["atm_iv", "iv_skew", "pcr_oi", "pcr_vol", "max_pain",
                             "term_structure_slope", "straddle_price", "atm_iv_5d_ma",
                             "atm_iv_20d_ma", "iv_skew_5d_ma", "pcr_oi_5d_ma", "pcr_vol_5d_ma",
                             "max_pain_5d_ma", "term_structure_slope_5d_ma", "straddle_price_5d_ma",
                             "atm_iv_z", "iv_skew_z", "pcr_oi_z", "pcr_vol_z", "max_pain_z",
                             "term_structure_slope_z", "straddle_price_z"],
        "fundamentals_earnings_features": ["earnings_surprise", "earnings_surprise_abs",
                                           "analyst_revision_1m", "analyst_revision_3m",
                                           "pe_ratio", "pb_ratio", "roe", "roce", "debt_to_equity",
                                           "current_ratio", "interest_coverage", "f_score",
                                           "magic_formula_rank", "earnings_surprise_5d_ma",
                                           "analyst_revision_1m_5d_ma", "pe_ratio_5d_ma",
                                           "pb_ratio_5d_ma", "roe_5d_ma", "roce_5d_ma",
                                           "debt_to_equity_5d_ma", "current_ratio_5d_ma",
                                           "interest_coverage_5d_ma", "f_score_5d_ma",
                                           "magic_formula_rank_5d_ma"],
        "news_sentiment_features": ["sentiment_vader", "sentiment_finbert", "article_count",
                                    "sector_sentiment_vader", "sector_sentiment_finbert",
                                    "market_sentiment_vader", "market_sentiment_finbert",
                                    "sentiment_vader_5d_ma", "sentiment_finbert_5d_ma",
                                    "article_count_5d_ma", "sector_sentiment_vader_5d_ma",
                                    "sector_sentiment_finbert_5d_ma", "market_sentiment_vader_5d_ma",
                                    "market_sentiment_finbert_5d_ma", "sentiment_vader_z",
                                    "sentiment_finbert_z", "article_count_z",
                                    "sector_sentiment_vader_z", "sector_sentiment_finbert_z",
                                    "market_sentiment_vader_z", "market_sentiment_finbert_z"],
    }
    
    # Add external categories to schema only if corresponding data files exist
    for category, features in external_feature_categories.items():
        # Check if any feature from this category exists in the combined dataframe
        if any(f in combined.columns for f in features):
            feature_schema[category] = features

    # All feature names (flat)
    all_feature_names = []
    for v in feature_schema.values():
        all_feature_names.extend(v)

    # Verify all features exist
    missing_features = [f for f in all_feature_names if f not in combined.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # --- Missing value summary ---
    missing_summary = {}
    for col in feature_schema.get("target", []):
        if col in combined.columns:
            missing_summary[col] = int(combined[col].isna().sum())
    for category, feats in feature_schema.items():
        if category == "target":
            continue
        for col in feats:
            if col in combined.columns:
                n_miss = int(combined[col].isna().sum())
                if n_miss > 0:
                    missing_summary[col] = n_miss

    # --- Target balance ---
    target_valid = combined["target_direction"].dropna()
    target_balance = {
        "total": len(target_valid),
        "up": int((target_valid == 1).sum()),
        "down": int((target_valid == 0).sum()),
        "pct_up": round(float((target_valid == 1).mean()) * 100, 2),
    }

    # --- Removal summary ---
    removal_df = pd.DataFrame(removal_log, columns=["symbol", "reason", "rows_removed"])

    # --- Per-symbol row counts ---
    symbol_stats = combined.groupby("symbol").agg(
        rows=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        target_nulls=("target_direction", lambda x: int(x.isna().sum())),
    ).reset_index()

    # --- Write outputs ---
    out_prefix = Path(output_path)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Ensure date is string for parquet compatibility
    combined["date"] = pd.to_datetime(combined["date"])

    combined.to_parquet(out_prefix.with_suffix(".parquet"), index=False)
    combined.to_csv(out_prefix.with_suffix(".csv"), index=False)

    # --- Feature schema JSON ---
    import json
    schema_path = Path(output_path).resolve().parents[2] / "reports" / "feature_schema.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    feature_list = []
    for category, feats in feature_schema.items():
        for fname in feats:
            feature_list.append({
                "name": fname,
                "category": category,
                "description": _get_feature_description(fname),
            })

    schema_json = {
        "version": "nifty100_features_v1_2026-08-26",
        "source_dataset": "nifty100_ohlcv_v1_2026-08-26",
        "feature_count": len(feature_cols_all),
        "target": "target_direction",
        "target_description": "1 if next session close > current close, else 0",
        "features": feature_list,
    }
    schema_path.write_text(json.dumps(schema_json, indent=2), encoding="utf-8")

    result = {
        "features": combined,
        "data": combined,
        "df": combined,
        "output": combined,
        "row_count": len(combined),
        "symbol_count": combined["symbol"].nunique(),
        "feature_count": len(feature_cols_all),
        "missing_summary": missing_summary,
        "target_balance": target_balance,
        "removal_log": removal_df,
        "symbol_stats": symbol_stats,
        "feature_schema": feature_schema,
    }
    return result


def _get_feature_description(name: str) -> str:
    """Return a human-readable description for each feature."""
    descs = {
        "ret_1d": "1-day percentage return",
        "ret_2d": "2-day percentage return",
        "ret_3d": "3-day percentage return",
        "ret_5d": "5-day percentage return",
        "ret_10d": "10-day percentage return",
        "ret_20d": "20-day percentage return",
        "log_ret_1d": "1-day log return",
        "log_ret_2d": "2-day log return",
        "log_ret_3d": "3-day log return",
        "log_ret_5d": "5-day log return",
        "log_ret_10d": "10-day log return",
        "log_ret_20d": "20-day log return",
        "hl_range": "(High - Low) / Close",
        "co_return": "(Close - Open) / Open",
        "gap_return": "(Open - Prev Close) / Prev Close",
        "sma_5": "5-day Simple Moving Average",
        "sma_10": "10-day Simple Moving Average",
        "sma_20": "20-day Simple Moving Average",
        "sma_50": "50-day Simple Moving Average",
        "sma_200": "200-day Simple Moving Average",
        "dist_sma_5": "Close distance from 5-day SMA",
        "dist_sma_10": "Close distance from 10-day SMA",
        "dist_sma_20": "Close distance from 20-day SMA",
        "dist_sma_50": "Close distance from 50-day SMA",
        "dist_sma_200": "Close distance from 200-day SMA",
        "ema_12": "12-day Exponential Moving Average",
        "ema_26": "26-day Exponential Moving Average",
        "dist_ema_12": "Close distance from 12-day EMA",
        "dist_ema_26": "Close distance from 26-day EMA",
        "sma_5_10_cross": "SMA5 > SMA10 crossover flag",
        "sma_10_20_cross": "SMA10 > SMA20 crossover flag",
        "sma_20_50_cross": "SMA20 > SMA50 crossover flag",
        "sma_50_200_cross": "SMA50 > SMA200 crossover flag",
        "rsi_14": "14-period Wilder's RSI",
        "macd": "MACD line normalised by price",
        "macd_signal": "MACD signal line normalised by price",
        "macd_hist": "MACD histogram normalised by price",
        "stoch_k": "14-period Stochastic %K",
        "stoch_d": "3-period SMA of Stochastic %K (%D)",
        "adx": "14-period Average Directional Index",
        "plus_di": "14-period +DI",
        "minus_di": "14-period -DI",
        "volatility_5d": "5-day rolling return std",
        "volatility_10d": "10-day rolling return std",
        "volatility_20d": "20-day rolling return std",
        "volatility_50d": "50-day rolling return std",
        "atr_14": "14-period Average True Range",
        "bb_upper": "Bollinger Band upper (SMA20 + 2*std)",
        "bb_lower": "Bollinger Band lower (SMA20 - 2*std)",
        "bb_position": "Position within Bollinger Bands (0=lower, 1=upper)",
        "bb_width": "Bollinger Band width / SMA20",
        "volume_pct_change": "Volume percentage change from previous day",
        "avg_volume_5d": "5-day average volume",
        "avg_volume_10d": "10-day average volume",
        "avg_volume_20d": "20-day average volume",
        "volume_ratio_20d": "Volume / 20-day average volume",
        "obv": "On-Balance Volume (cumulative)",
        "obv_normalised": "OBV normalised by its own 20-day std",
        "rank_return": "Daily percentile rank of stock return within universe",
        "rank_volume_ratio": "Daily percentile rank of volume ratio within universe",
        "excess_return": "Stock return minus universe mean return",
        "rel_volatility": "Stock 20d volatility / universe mean 20d volatility",
        # FII/DII features
        "fii_net_buy": "FII net equity buy (Rs Cr)",
        "dii_net_buy": "DII net equity buy (Rs Cr)",
        "fii_net_buy_5d_ma": "FII net buy 5-day moving average",
        "fii_net_buy_20d_ma": "FII net buy 20-day moving average",
        "dii_net_buy_5d_ma": "DII net buy 5-day moving average",
        "dii_net_buy_20d_ma": "DII net buy 20-day moving average",
        "fii_dii_spread": "FII net buy - DII net buy",
        "fii_net_buy_z": "FII net buy z-score (20d)",
        "dii_net_buy_z": "DII net buy z-score (20d)",
        "fii_dii_spread_z": "FII-DII spread z-score (20d)",
        "fii_momentum_5d": "FII net buy 5-day momentum",
        "fii_momentum_20d": "FII net buy 20-day momentum",
        "dii_momentum_5d": "DII net buy 5-day momentum",
        "dii_momentum_20d": "DII net buy 20-day momentum",
        "fii_dii_corr_20d": "FII-DII 20-day rolling correlation",
        "fii_net_buy_ma_ratio": "FII net buy / 20d MA ratio",
        "dii_net_buy_ma_ratio": "DII net buy / 20d MA ratio",
        # Delivery features
        "delivery_pct": "Delivery percentage",
        "delivery_pct_5d_ma": "Delivery % 5-day MA",
        "delivery_pct_20d_ma": "Delivery % 20-day MA",
        "delivery_pct_z": "Delivery % z-score (20d)",
        "high_delivery_flag": "High delivery % flag (>80th pctl)",
        "bulk_deal_flag": "Bulk deal occurred flag",
        "block_deal_flag": "Block deal occurred flag",
        "bulk_deal_value": "Bulk deal value (Rs Cr)",
        "block_deal_value": "Block deal value (Rs Cr)",
        "delivery_momentum_5d": "Delivery % 5-day momentum",
        "delivery_momentum_20d": "Delivery % 20-day momentum",
        "delivery_trend_5d": "Delivery % 5-day trend (slope)",
        "delivery_trend_20d": "Delivery % 20-day trend (slope)",
        "bulk_deal_count_20d": "Bulk deal count (20d)",
        "block_deal_count_20d": "Block deal count (20d)",
        # Intraday microstructure features
        "vwap_dev": "VWAP deviation (close - VWAP) / VWAP",
        "realized_vol": "Intraday realized volatility (5-min)",
        "ofi": "Order Flow Imbalance",
        "roll_spread": "Roll spread estimator",
        "kyle_lambda": "Kyle's lambda (price impact)",
        "amihud_illiq": "Amihud illiquidity ratio",
        "vwap_dev_5d_ma": "VWAP deviation 5-day MA",
        "realized_vol_5d_ma": "Realized vol 5-day MA",
        "ofi_5d_ma": "OFI 5-day MA",
        "roll_spread_5d_ma": "Roll spread 5-day MA",
        "kyle_lambda_5d_ma": "Kyle's lambda 5-day MA",
        "amihud_illiq_5d_ma": "Amihud illiquidity 5-day MA",
        "vwap_dev_z": "VWAP deviation z-score (20d)",
        "realized_vol_z": "Realized vol z-score (20d)",
        "ofi_z": "OFI z-score (20d)",
        "roll_spread_z": "Roll spread z-score (20d)",
        "kyle_lambda_z": "Kyle's lambda z-score (20d)",
        "amihud_illiq_z": "Amihud illiquidity z-score (20d)",
        # Options features
        "atm_iv": "ATM implied volatility",
        "iv_skew": "IV skew (25d put - 25d call)",
        "pcr_oi": "Put-Call Ratio (Open Interest)",
        "pcr_vol": "Put-Call Ratio (Volume)",
        "max_pain": "Max pain strike price",
        "term_structure_slope": "IV term structure slope",
        "straddle_price": "ATM straddle price",
        "atm_iv_5d_ma": "ATM IV 5-day MA",
        "atm_iv_20d_ma": "ATM IV 20-day MA",
        "iv_skew_5d_ma": "IV skew 5-day MA",
        "pcr_oi_5d_ma": "PCR OI 5-day MA",
        "pcr_vol_5d_ma": "PCR Vol 5-day MA",
        "max_pain_5d_ma": "Max pain 5-day MA",
        "term_structure_slope_5d_ma": "Term structure slope 5-day MA",
        "straddle_price_5d_ma": "Straddle price 5-day MA",
        "atm_iv_z": "ATM IV z-score (20d)",
        "iv_skew_z": "IV skew z-score (20d)",
        "pcr_oi_z": "PCR OI z-score (20d)",
        "pcr_vol_z": "PCR Vol z-score (20d)",
        "max_pain_z": "Max pain z-score (20d)",
        "term_structure_slope_z": "Term structure slope z-score (20d)",
        "straddle_price_z": "Straddle price z-score (20d)",
        # Fundamentals/Earnings features
        "earnings_surprise": "Earnings surprise %",
        "earnings_surprise_abs": "Absolute earnings surprise %",
        "analyst_revision_1m": "Analyst estimate revision (1m)",
        "analyst_revision_3m": "Analyst estimate revision (3m)",
        "pe_ratio": "Price-to-Earnings ratio",
        "pb_ratio": "Price-to-Book ratio",
        "roe": "Return on Equity",
        "roce": "Return on Capital Employed",
        "debt_to_equity": "Debt-to-Equity ratio",
        "current_ratio": "Current ratio",
        "interest_coverage": "Interest coverage ratio",
        "f_score": "Piotroski F-Score",
        "magic_formula_rank": "Magic Formula rank",
        "earnings_surprise_5d_ma": "Earnings surprise 5-day MA",
        "analyst_revision_1m_5d_ma": "Analyst revision 1m 5-day MA",
        "pe_ratio_5d_ma": "P/E ratio 5-day MA",
        "pb_ratio_5d_ma": "P/B ratio 5-day MA",
        "roe_5d_ma": "ROE 5-day MA",
        "roce_5d_ma": "ROCE 5-day MA",
        "debt_to_equity_5d_ma": "D/E ratio 5-day MA",
        "current_ratio_5d_ma": "Current ratio 5-day MA",
        "interest_coverage_5d_ma": "Interest coverage 5-day MA",
        "f_score_5d_ma": "F-Score 5-day MA",
        "magic_formula_rank_5d_ma": "Magic Formula rank 5-day MA",
        # News Sentiment features
        "sentiment_vader": "VADER sentiment score",
        "sentiment_finbert": "FinBERT sentiment score",
        "article_count": "Number of news articles",
        "sector_sentiment_vader": "Sector VADER sentiment",
        "sector_sentiment_finbert": "Sector FinBERT sentiment",
        "market_sentiment_vader": "Market VADER sentiment",
        "market_sentiment_finbert": "Market FinBERT sentiment",
        "sentiment_vader_5d_ma": "VADER sentiment 5-day MA",
        "sentiment_finbert_5d_ma": "FinBERT sentiment 5-day MA",
        "article_count_5d_ma": "Article count 5-day MA",
        "sector_sentiment_vader_5d_ma": "Sector VADER 5-day MA",
        "sector_sentiment_finbert_5d_ma": "Sector FinBERT 5-day MA",
        "market_sentiment_vader_5d_ma": "Market VADER 5-day MA",
        "market_sentiment_finbert_5d_ma": "Market FinBERT 5-day MA",
        "sentiment_vader_z": "VADER sentiment z-score (20d)",
        "sentiment_finbert_z": "FinBERT sentiment z-score (20d)",
        "article_count_z": "Article count z-score (20d)",
        "sector_sentiment_vader_z": "Sector VADER z-score (20d)",
        "sector_sentiment_finbert_z": "Sector FinBERT z-score (20d)",
        "market_sentiment_vader_z": "Market VADER z-score (20d)",
        "market_sentiment_finbert_z": "Market FinBERT z-score (20d)",
        "target_direction": "1 if next session close > current close, else 0",
    }
    return descs.get(name, name)
