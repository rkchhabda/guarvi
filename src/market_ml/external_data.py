"""External data sources for enhanced features.

This module provides functions to fetch and process external data sources
that are not derivable from OHLCV price/volume data alone.

Sources:
- FII/DII flows from NSDL (daily equity flows)
- NSE options data (IV, PCR, max-pain)
- NSE bhavcopy delivery % and bulk/block deals
- Intraday microstructure features
- Fundamentals/earnings data
- News/sentiment data
"""
from __future__ import annotations

import logging
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FII/DII Data from NSDL
# ---------------------------------------------------------------------------

NSDL_FII_DII_URL = "https://www.nsdl.co.in/downloadables/statistics/FII_DII_Statistics.csv"
NSDL_FII_DII_LOCAL = "data/external/fii_dii_flows.csv"
# Free, NSE/NSDL-sourced aggregator (full daily history as JSON) used as a
# reachable fallback when NSDL/NSE block automated access.
AGGREGATOR_FII_DII_URL = "https://fii-diidata.mrchartist.com/api/history-full"


def _tls_context() -> ssl.SSLContext:
    """TLS context that tolerates the lax cert setups some Indian exchanges use."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _open_url(url: str, headers: Optional[dict] = None, timeout: int = 40):
    """Open a URL with a browser-like UA and cookie jar; returns bytes."""
    import http.cookiejar
    import urllib.request

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_tls_context()),
        urllib.request.HTTPCookieProcessor(jar),
    )
    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/json,*/*",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    return opener.open(req, timeout=timeout).read()


def _parse_nsdl_csv(raw: bytes) -> pd.DataFrame:
    """Parse the NSDL FII/DII statistics CSV into the canonical schema."""
    from io import StringIO

    df = pd.read_csv(StringIO(raw.decode("utf-8", "ignore")))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "date" in cl:
            col_map[col] = "date"
        elif "fii" in cl and "net" in cl:
            col_map[col] = "fii_net"
        elif "fii" in cl and ("gross" in cl or "purchas" in cl) and "buy" in cl:
            col_map[col] = "fii_gross_buy"
        elif "fii" in cl and ("gross" in cl or "sales" in cl) and "sell" in cl:
            col_map[col] = "fii_gross_sell"
        elif "dii" in cl and "net" in cl:
            col_map[col] = "dii_net"
        elif "dii" in cl and ("gross" in cl or "purchas" in cl) and "buy" in cl:
            col_map[col] = "dii_gross_buy"
        elif "dii" in cl and ("gross" in cl or "sales" in cl) and "sell" in cl:
            col_map[col] = "dii_gross_sell"
    df = df.rename(columns=col_map)
    for col in ("fii_net", "dii_net"):
        if col not in df.columns:
            if f"{col[:3]}_gross_buy" in df.columns and f"{col[:3]}_gross_sell" in df.columns:
                df[col] = df[f"{col[:3]}_gross_buy"] - df[f"{col[:3]}_gross_sell"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    keep = [c for c in ["date", "fii_net", "fii_gross_buy", "fii_gross_sell",
                        "dii_net", "dii_gross_buy", "dii_gross_sell"] if c in df.columns]
    return df[keep]


def _fetch_nsdl() -> pd.DataFrame:
    try:
        raw = _open_url(NSDL_FII_DII_URL, timeout=40)
        return _parse_nsdl_csv(raw)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("NSDL FII/DII fetch failed: %s", exc)
        return pd.DataFrame()


def _fetch_aggregator() -> pd.DataFrame:
    """Fetch full daily FII/DII history from the free NSE/NSDL-sourced aggregator."""
    try:
        import json

        raw = _open_url(AGGREGATOR_FII_DII_URL, headers={"Accept": "application/json"}, timeout=40)
        rows = json.loads(raw.decode("utf-8"))
        recs = []
        for r in rows:
            recs.append({
                "date": pd.to_datetime(r["d"], format="%d-%b-%Y", errors="coerce"),
                "fii_gross_buy": float(r.get("fb", np.nan)),
                "fii_gross_sell": float(r.get("fs", np.nan)),
                "fii_net": float(r.get("fn", np.nan)),
                "dii_gross_buy": float(r.get("db", np.nan)),
                "dii_gross_sell": float(r.get("ds", np.nan)),
                "dii_net": float(r.get("dn", np.nan)),
            })
        df = pd.DataFrame(recs).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        logger.info("Aggregator returned %d FII/DII rows (%s..%s)", len(df),
                    df["date"].min().date(), df["date"].max().date())
        return df
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Aggregator FII/DII fetch failed: %s", exc)
        return pd.DataFrame()


def fetch_fii_dii_flows(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_path: str = NSDL_FII_DII_LOCAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch FII/DII daily equity flows from NSDL.
    
    NSDL publishes daily FII/DII statistics including:
    - FII Gross Purchases, Gross Sales, Net Investment
    - DII Gross Purchases, Gross Sales, Net Investment
    
    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    cache_path : str
        Local cache file path
    force_refresh : bool
        Force re-download even if cache exists
    
    Returns
    -------
    DataFrame with columns: date, fii_net, fii_gross_buy, fii_gross_sell,
                            dii_net, dii_gross_buy, dii_gross_sell
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if cache_file.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            logger.info(f"Loaded FII/DII data from cache: {len(df)} rows")
            return _filter_date_range(df, start_date, end_date)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will re-download")
    
    # Download: try NSDL (authoritative, full history) then the reachable
    # NSE/NSDL-sourced aggregator as a fallback.
    df = _fetch_nsdl()
    if df.empty:
        df = _fetch_aggregator()

    if df.empty:
        logger.error("Failed to fetch FII/DII data from all sources")
        return pd.DataFrame(columns=['date', 'fii_net', 'dii_net'])

    # Warn if coverage is too short for a proper multi-year walk-forward.
    span_days = (df['date'].max() - df['date'].min()).days
    if span_days < 365:
        logger.warning(
            "FII/DII coverage is only %d days (%s..%s). A valid multi-year "
            "walk-forward test needs full history; treat any Phase-9 result on "
            "this series as a data-limited recent-window probe, not a holdout.",
            span_days, df['date'].min().date(), df['date'].max().date(),
        )

    # Save cache
    df.to_csv(cache_file, index=False)
    logger.info(f"Saved FII/DII data to cache: {len(df)} rows")
    return _filter_date_range(df, start_date, end_date)


def _filter_date_range(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if start_date:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['date'] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


def compute_fii_dii_features(fii_dii_df: pd.DataFrame, lookback_windows: list = None) -> pd.DataFrame:
    """
    Compute FII/DII derived features for merging with OHLCV data.
    
    Features:
    - fii_net_5d_ma, fii_net_20d_ma: Moving averages of FII net flows
    - fii_net_zscore_20d: Z-score of FII net flows (20-day)
    - fii_dii_net_spread: FII net - DII net (divergence indicator)
    - fii_dii_ratio: FII gross buy / DII gross buy (relative aggression)
    - fii_momentum_5d: 5-day momentum of FII net flows
    - dii_momentum_5d: 5-day momentum of DII net flows
    - fii_dii_correlation_20d: Rolling correlation between FII and DII flows
    """
    if lookback_windows is None:
        lookback_windows = [5, 10, 20, 50]
    
    df = fii_dii_df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Ensure we have the required columns
    if 'fii_net' not in df.columns or 'dii_net' not in df.columns:
        logger.warning("FII/DII net columns missing, cannot compute features")
        return pd.DataFrame({'date': df['date']})
    
    out = pd.DataFrame({'date': df['date']})
    
    # Moving averages of net flows
    for w in lookback_windows:
        out[f'fii_net_{w}d_ma'] = df['fii_net'].rolling(w, min_periods=w).mean()
        out[f'dii_net_{w}d_ma'] = df['dii_net'].rolling(w, min_periods=w).mean()
    
    # Z-scores
    for w in [20, 50]:
        fii_ma = df['fii_net'].rolling(w, min_periods=w).mean()
        fii_std = df['fii_net'].rolling(w, min_periods=w).std(ddof=0)
        out[f'fii_net_zscore_{w}d'] = (df['fii_net'] - fii_ma) / fii_std.replace(0, np.nan)
        
        dii_ma = df['dii_net'].rolling(w, min_periods=w).mean()
        dii_std = df['dii_net'].rolling(w, min_periods=w).std(ddof=0)
        out[f'dii_net_zscore_{w}d'] = (df['dii_net'] - dii_ma) / dii_std.replace(0, np.nan)
    
    # FII-DII spread (divergence)
    out['fii_dii_net_spread'] = df['fii_net'] - df['dii_net']
    out['fii_dii_net_spread_5d_ma'] = out['fii_dii_net_spread'].rolling(5, min_periods=5).mean()
    out['fii_dii_net_spread_20d_ma'] = out['fii_dii_net_spread'].rolling(20, min_periods=20).mean()
    
    # FII/DII ratio (relative aggression)
    if 'fii_gross_buy' in df.columns and 'dii_gross_buy' in df.columns:
        out['fii_dii_buy_ratio'] = df['fii_gross_buy'] / df['dii_gross_buy'].replace(0, np.nan)
        out['fii_dii_buy_ratio_5d_ma'] = out['fii_dii_buy_ratio'].rolling(5, min_periods=5).mean()
    
    # Momentum
    out['fii_momentum_5d'] = df['fii_net'].diff(5)
    out['dii_momentum_5d'] = df['dii_net'].diff(5)
    out['fii_momentum_20d'] = df['fii_net'].diff(20)
    out['dii_momentum_20d'] = df['dii_net'].diff(20)
    
    # Rolling correlation
    out['fii_dii_corr_20d'] = df['fii_net'].rolling(20, min_periods=10).corr(df['dii_net'])
    out['fii_dii_corr_50d'] = df['fii_net'].rolling(50, min_periods=20).corr(df['dii_net'])
    
    # Net flow direction (sign)
    out['fii_net_positive'] = (df['fii_net'] > 0).astype(float)
    out['dii_net_positive'] = (df['dii_net'] > 0).astype(float)
    out['fii_dii_same_direction'] = (out['fii_net_positive'] == out['dii_net_positive']).astype(float)
    
    # Consecutive days of same direction
    out['fii_consecutive_buy'] = (df['fii_net'] > 0).astype(int).groupby((df['fii_net'] <= 0).astype(int).cumsum()).cumsum()
    out['fii_consecutive_sell'] = (df['fii_net'] < 0).astype(int).groupby((df['fii_net'] >= 0).astype(int).cumsum()).cumsum()
    out['dii_consecutive_buy'] = (df['dii_net'] > 0).astype(int).groupby((df['dii_net'] <= 0).astype(int).cumsum()).cumsum()
    out['dii_consecutive_sell'] = (df['dii_net'] < 0).astype(int).groupby((df['dii_net'] >= 0).astype(int).cumsum()).cumsum()
    
    return out


# ---------------------------------------------------------------------------
# NSE Options Data
# ---------------------------------------------------------------------------

def fetch_nse_option_chain(
    symbol: str = "NIFTY",
    expiry: Optional[str] = None,
    cache_dir: str = "data/external/options",
) -> pd.DataFrame:
    """
    Fetch NSE option chain data for a given symbol and expiry.
    
    Note: This requires NSE API access or web scraping. For production,
    consider using a paid data vendor or NSE's official API.
    
    Returns DataFrame with: strike, call_oi, call_iv, call_volume,
                            put_oi, put_iv, put_volume, expiry, date
    """
    cache_path = Path(cache_dir) / f"{symbol}_option_chain.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded option chain from cache: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to load option chain cache: {e}")
    
    # Placeholder - in production, implement NSE API fetch
    logger.warning("NSE option chain fetch not implemented - requires NSE API access")
    return pd.DataFrame()


def compute_options_features(option_chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily options features from option chain data.
    
    Features:
    - atm_iv: At-the-money implied volatility
    - iv_skew: 25-delta put IV - 25-delta call IV (risk reversal)
    - pcr_oi: Put-Call ratio (open interest)
    - pcr_volume: Put-Call ratio (volume)
    - max_pain: Strike with maximum pain (min total option value)
    - iv_percentile_20d: IV percentile rank over 20 days
    - iv_term_structure: IV difference between near and far expiry
    """
    if option_chain_df.empty:
        return pd.DataFrame()
    
    # Group by date and compute daily features
    daily_features = []
    
    for date, group in option_chain_df.groupby('date'):
        # Find ATM strike (closest to spot)
        # This requires spot price - would need to merge with OHLCV
        pass
    
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# NSE Bhavcopy Delivery & Bulk/Block Deals
# ---------------------------------------------------------------------------

NSE_BHAVCOPY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
NSE_BULK_DEALS_URL = "https://www.nseindia.com/api/historical/bulk-deals?from={from_date}&to={to_date}"
NSE_BLOCK_DEALS_URL = "https://www.nseindia.com/api/historical/block-deals?from={from_date}&to={to_date}"


def fetch_nse_bhavcopy(
    date: str,
    cache_dir: str = "data/external/bhavcopy",
) -> pd.DataFrame:
    """
    Fetch NSE bhavcopy for a specific date.
    
    Bhavcopy contains delivery percentage data which is not available
    from regular OHLCV sources.
    """
    cache_path = Path(cache_dir) / f"bhavcopy_{date}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    
    # Format date for URL (DDMMMYYYY)
    dt = pd.Timestamp(date)
    date_str = dt.strftime('%d%b%Y').upper()
    url = NSE_BHAVCOPY_URL.format(date=date_str)
    
    try:
        logger.info(f"Fetching bhavcopy for {date}...")
        df = pd.read_csv(url)
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch bhavcopy for {date}: {e}")
        return pd.DataFrame()


def fetch_bulk_block_deals(
    from_date: str,
    to_date: str,
    cache_dir: str = "data/external/bulk_block",
) -> pd.DataFrame:
    """
    Fetch bulk and block deals from NSE.
    
    Bulk deals: >0.5% of equity shares traded in a single transaction
    Block deals: >5 lakh shares or >5 crore value in a single transaction
    """
    cache_path = Path(cache_dir) / f"bulk_block_{from_date}_to_{to_date}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    
    # Requires NSE API with proper headers/authentication
    logger.warning("Bulk/block deals fetch requires NSE API authentication")
    return pd.DataFrame()


def compute_delivery_features(bhavcopy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delivery-based features from bhavcopy data.
    
    Features:
    - delivery_pct: Delivery quantity / Traded quantity
    - delivery_value: Delivery quantity * Close price
    - delivery_to_volume_ratio: Delivery qty / Volume
    - high_delivery_flag: Delivery % > 70th percentile (accumulation)
    - low_delivery_flag: Delivery % < 30th percentile (distribution)
    """
    if bhavcopy_df.empty:
        return pd.DataFrame()
    
    # Expected columns in bhavcopy: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE,
    # LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES,
    # ISIN, DELIVERY_QTY, DELIVERY_PERCENT
    
    df = bhavcopy_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Standardize column names
    col_map = {}
    for col in df.columns:
        if 'symbol' in col.lower():
            col_map[col] = 'symbol'
        elif 'delivery' in col.lower() and 'qty' in col.lower():
            col_map[col] = 'delivery_qty'
        elif 'delivery' in col.lower() and 'per' in col.lower():
            col_map[col] = 'delivery_pct'
        elif 'tottrdqty' in col.lower() or 'volume' in col.lower():
            col_map[col] = 'volume'
        elif 'close' in col.lower():
            col_map[col] = 'close'
        elif 'date' in col.lower() or 'timestamp' in col.lower():
            col_map[col] = 'date'
    
    df = df.rename(columns=col_map)
    
    if 'delivery_qty' not in df.columns or 'volume' not in df.columns:
        logger.warning("Required delivery columns not found in bhavcopy")
        return pd.DataFrame()
    
    out = pd.DataFrame()
    out['symbol'] = df['symbol']
    out['date'] = pd.to_datetime(df['date'])
    out['delivery_pct'] = df['delivery_pct'] if 'delivery_pct' in df.columns else df['delivery_qty'] / df['volume'].replace(0, np.nan) * 100
    out['delivery_qty'] = df['delivery_qty']
    out['delivery_value'] = df['delivery_qty'] * df['close']
    out['delivery_to_volume'] = df['delivery_qty'] / df['volume'].replace(0, np.nan)
    
    # Rolling features (per symbol)
    for sym, group in out.groupby('symbol'):
        group = group.sort_values('date')
        idx = group.index
        out.loc[idx, 'delivery_pct_5d_ma'] = group['delivery_pct'].rolling(5, min_periods=5).mean().values
        out.loc[idx, 'delivery_pct_20d_ma'] = group['delivery_pct'].rolling(20, min_periods=20).mean().values
        out.loc[idx, 'delivery_pct_zscore_20d'] = (
            (group['delivery_pct'] - group['delivery_pct'].rolling(20, min_periods=20).mean()) /
            group['delivery_pct'].rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
        ).values
    
    # High/low delivery flags
    out['high_delivery'] = (out['delivery_pct'] > out['delivery_pct'].rolling(252, min_periods=100).quantile(0.7)).astype(float)
    out['low_delivery'] = (out['delivery_pct'] < out['delivery_pct'].rolling(252, min_periods=100).quantile(0.3)).astype(float)
    
    return out


# ---------------------------------------------------------------------------
# Intraday Microstructure Features
# ---------------------------------------------------------------------------

def compute_intraday_microstructure_features(
    intraday_df: pd.DataFrame,
    freq: str = '5min',
) -> pd.DataFrame:
    """
    Compute microstructure features from intraday data.
    
    Requires intraday OHLCV data at 5-min or 15-min frequency.
    
    Features:
    - vwap: Volume-weighted average price
    - vwap_deviation: (Close - VWAP) / VWAP
    - intraday_range: (High - Low) / Close
    - close_to_vwap: Close / VWAP - 1
    - volume_weighted_momentum: Correlation of price change with volume
    - order_flow_imbalance_proxy: (Buy volume - Sell volume) / Total volume
      (approximated using uptick/downtick rule)
    - realized_volatility: Sum of squared 5-min returns
    - first_hour_return: Return from open to 10:15 AM
    - last_hour_return: Return from 2:15 PM to close
    - am_pm_volume_ratio: Morning volume / Afternoon volume
    """
    # This requires intraday data which is not in the current dataset
    # Placeholder for future implementation
    logger.warning("Intraday microstructure features require intraday data feed")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fundamentals / Earnings Data
# ---------------------------------------------------------------------------

def fetch_earnings_calendar(
    symbols: list,
    start_date: str,
    end_date: str,
    cache_dir: str = "data/external/earnings",
) -> pd.DataFrame:
    """
    Fetch earnings announcement dates and estimates.
    
    Sources: Tickertape, Trendlyne, Screener.in, or NSE corporate announcements.
    Requires paid subscription or web scraping.
    """
    logger.warning("Earnings calendar fetch requires paid data source or scraping")
    return pd.DataFrame()


def compute_earnings_features(earnings_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute earnings-related features.
    
    Features:
    - days_to_earnings: Days until next earnings announcement
    - days_since_earnings: Days since last earnings announcement
    - earnings_surprise: Actual EPS - Estimated EPS (normalized)
    - earnings_surprise_1d_ret: 1-day return around earnings
    - earnings_surprise_5d_ret: 5-day return around earnings
    - guidance_revision: Upward/downward guidance change flag
    - analyst_revision: Mean estimate revision (30-day)
    """
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# News / Sentiment Data
# ---------------------------------------------------------------------------

def fetch_news_sentiment(
    symbols: list,
    start_date: str,
    end_date: str,
    cache_dir: str = "data/external/news",
) -> pd.DataFrame:
    """
    Fetch news sentiment data.
    
    Sources: NewsAPI, GNews, RSS feeds, or paid terminals (Bloomberg, Reuters).
    Free sources have significant delays.
    """
    logger.warning("News sentiment fetch requires API keys or paid subscription")
    return pd.DataFrame()


def compute_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment features from news data.
    
    Features:
    - sentiment_score: Average sentiment (-1 to 1)
    - sentiment_momentum: Change in sentiment over 5 days
    - news_volume: Number of articles per day
    - positive_news_ratio: Positive articles / Total articles
    - negative_news_ratio: Negative articles / Total articles
    - sentiment_zscore_20d: Z-score of sentiment over 20 days
    """
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Integration Helper
# ---------------------------------------------------------------------------

def merge_external_features(
    base_df: pd.DataFrame,
    external_features: dict[str, pd.DataFrame],
    on: list = ['date', 'symbol'],
) -> pd.DataFrame:
    """
    Merge multiple external feature DataFrames with base OHLCV/features DataFrame.
    
    Parameters
    ----------
    base_df : DataFrame
        Base DataFrame with at least ['date', 'symbol'] columns
    external_features : dict
        Dictionary of {feature_name: DataFrame} to merge
    on : list
        Columns to merge on
    
    Returns
    -------
    DataFrame with all features merged
    """
    result = base_df.copy()
    
    for name, feat_df in external_features.items():
        if feat_df.empty:
            logger.warning(f"Skipping empty feature set: {name}")
            continue
        
        # Ensure date column is datetime
        if 'date' in feat_df.columns:
            feat_df['date'] = pd.to_datetime(feat_df['date'])
        
        # Merge
        result = result.merge(feat_df, on=on, how='left', suffixes=('', f'_{name}'))
        logger.info(f"Merged {name}: {feat_df.shape[1]-len(on)} new columns")
    
    return result


def get_available_external_data(start_date: str, end_date: str) -> dict:
    """
    Check which external data sources are available for the given date range.
    
    Returns dict with availability status for each source.
    """
    return {
        'fii_dii': True,  # NSDL public data
        'nse_options': False,  # Requires NSE API
        'delivery_bulk': False,  # Requires NSE bhavcopy/API
        'intraday': False,  # Requires intraday data feed
        'fundamentals': False,  # Requires paid subscription
        'news_sentiment': False,  # Requires API keys
    }