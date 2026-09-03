#!/usr/bin/env python3
"""Fetch and process NSE bhavcopy delivery data and bulk/block deals.

This script downloads NSE bhavcopy files (which contain delivery percentage)
and bulk/block deal data, computes derived features, and saves them for
integration with the main feature pipeline.

NOTE: NSE bhavcopy requires specific date formatting and may need
proper headers/authentication. Bulk/block deals require NSE API access.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_nse_bhavcopy")


# NSE URLs
NSE_BHAVCOPY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
NSE_BHAVCOPY_URL_ALT = "https://www.nseindia.com/content/historical/EQUITIES/{year}/{month}/sec_bhavdata_full_{date}.csv"

# Headers to mimic browser
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


def fetch_bhavcopy_for_date(
    date: str,
    cache_dir: str = "data/external/bhavcopy",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch NSE bhavcopy for a specific date.
    
    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format
    cache_dir : str
        Directory to cache downloaded files
    max_retries : int
        Maximum number of retry attempts
    
    Returns
    -------
    DataFrame with bhavcopy data including delivery percentage
    """
    cache_path = Path(cache_dir) / f"bhavcopy_{date}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check cache first
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            logger.debug(f"Loaded bhavcopy from cache for {date}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {date}: {e}")
    
    dt = pd.Timestamp(date)
    
    # Skip weekends
    if dt.weekday() >= 5:
        logger.debug(f"Skipping weekend: {date}")
        return pd.DataFrame()
    
    # Try multiple URL formats
    urls = []
    
    # Format 1: DDMMMYYYY (e.g., 01JAN2024)
    date_str1 = dt.strftime('%d%b%Y').upper()
    urls.append(NSE_BHAVCOPY_URL.format(date=date_str1))
    
    # Format 2: YYYY/MMM/DDMMMYYYY
    year = dt.strftime('%Y')
    month = dt.strftime('%b').upper()
    date_str2 = dt.strftime('%d%b%Y').upper()
    urls.append(NSE_BHAVCOPY_URL_ALT.format(year=year, month=month, date=date_str2))
    
    for url in urls:
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching bhavcopy for {date} from {url} (attempt {attempt+1})")
                response = requests.get(url, headers=NSE_HEADERS, timeout=30)
                
                if response.status_code == 200:
                    # Parse CSV
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    
                    # Save to cache
                    df.to_csv(cache_path, index=False)
                    logger.info(f"Fetched and cached bhavcopy for {date}: {len(df)} rows")
                    return df
                elif response.status_code == 404:
                    logger.debug(f"Bhavcopy not found for {date} (404)")
                    break
                else:
                    logger.warning(f"HTTP {response.status_code} for {date}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {date}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.warning(f"Failed to fetch bhavcopy for {date} after all attempts")
    return pd.DataFrame()


def fetch_bhavcopy_range(
    start_date: str,
    end_date: str,
    cache_dir: str = "data/external/bhavcopy",
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch bhavcopy for a range of dates.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    cache_dir : str
        Cache directory
    delay : float
        Delay between requests (seconds) to be polite to NSE servers
    
    Returns
    -------
    Combined DataFrame with all bhavcopy data
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    all_data = []
    current = start
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        df = fetch_bhavcopy_for_date(date_str, cache_dir)
        
        if not df.empty:
            all_data.append(df)
        
        current += timedelta(days=1)
        time.sleep(delay)  # Be polite to NSE servers
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined bhavcopy data: {len(combined)} rows")
        return combined
    else:
        logger.warning("No bhavcopy data fetched")
        return pd.DataFrame()


def process_bhavcopy_data(bhavcopy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw bhavcopy data to extract delivery features.
    
    Expected columns in NSE bhavcopy:
    SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE,
    TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN,
    DELIVERY_QTY, DELIVERY_PERCENT
    """
    if bhavcopy_df.empty:
        return pd.DataFrame()
    
    df = bhavcopy_df.copy()
    
    # Standardize column names (NSE uses uppercase)
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Column mapping
    col_map = {
        'SYMBOL': 'symbol',
        'SERIES': 'series',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'LAST': 'last',
        'PREVCLOSE': 'prev_close',
        'TOTTRDQTY': 'volume',
        'TOTTRDVAL': 'turnover',
        'TIMESTAMP': 'date',
        'TOTALTRADES': 'trades',
        'ISIN': 'isin',
        'DELIVERY_QTY': 'delivery_qty',
        'DELIVERY_PERCENT': 'delivery_pct',
    }
    
    # Rename known columns
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Filter for EQ series only (regular equity)
    if 'series' in df.columns:
        df = df[df['series'] == 'EQ'].copy()
    
    # Ensure required columns
    required = ['symbol', 'date', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
    if df['date'].isna().any():
        # Try alternative format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Compute delivery percentage if not present
    if 'delivery_pct' not in df.columns and 'delivery_qty' in df.columns:
        df['delivery_pct'] = df['delivery_qty'] / df['volume'].replace(0, np.nan) * 100
    elif 'delivery_pct' in df.columns:
        # Ensure numeric
        df['delivery_pct'] = pd.to_numeric(df['delivery_pct'], errors='coerce')
    
    # Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'delivery_qty', 'delivery_pct']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def compute_delivery_features(bhavcopy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delivery-based features from processed bhavcopy data.
    
    Features per symbol per date:
    - delivery_pct: Delivery quantity / Traded quantity * 100
    - delivery_qty: Absolute delivery quantity
    - delivery_value: Delivery quantity * Close price
    - delivery_to_volume: Delivery qty / Volume
    - delivery_pct_5d_ma: 5-day MA of delivery %
    - delivery_pct_20d_ma: 20-day MA of delivery %
    - delivery_pct_zscore_20d: Z-score of delivery % over 20 days
    - high_delivery_flag: Delivery % > 70th percentile (252-day lookback)
    - low_delivery_flag: Delivery % < 30th percentile (252-day lookback)
    - delivery_momentum_5d: 5-day change in delivery %
    - delivery_trend: 20-day slope of delivery %
    """
    if bhavcopy_df.empty:
        return pd.DataFrame()
    
    df = bhavcopy_df.copy()
    
    # Ensure required columns
    required = ['symbol', 'date', 'delivery_pct', 'delivery_qty', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column for delivery features: {col}")
            return pd.DataFrame()
    
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Output DataFrame
    out = pd.DataFrame()
    out['symbol'] = df['symbol']
    out['date'] = df['date']
    out['delivery_pct'] = df['delivery_pct']
    out['delivery_qty'] = df['delivery_qty']
    out['delivery_value'] = df['delivery_qty'] * df['close']
    out['delivery_to_volume'] = df['delivery_qty'] / df['volume'].replace(0, np.nan)
    
    # Rolling features per symbol
    for symbol, group in df.groupby('symbol'):
        idx = group.index
        grp = group.sort_values('date')
        
        # Moving averages
        out.loc[idx, 'delivery_pct_5d_ma'] = grp['delivery_pct'].rolling(5, min_periods=5).mean().values
        out.loc[idx, 'delivery_pct_10d_ma'] = grp['delivery_pct'].rolling(10, min_periods=10).mean().values
        out.loc[idx, 'delivery_pct_20d_ma'] = grp['delivery_pct'].rolling(20, min_periods=20).mean().values
        out.loc[idx, 'delivery_pct_50d_ma'] = grp['delivery_pct'].rolling(50, min_periods=50).mean().values
        
        # Z-scores
        for w in [20, 50, 252]:
            ma = grp['delivery_pct'].rolling(w, min_periods=w//2).mean()
            std = grp['delivery_pct'].rolling(w, min_periods=w//2).std(ddof=0)
            out.loc[idx, f'delivery_pct_zscore_{w}d'] = ((grp['delivery_pct'] - ma) / std.replace(0, np.nan)).values
        
        # Percentile ranks (for high/low flags)
        out.loc[idx, 'delivery_pct_rank_252d'] = grp['delivery_pct'].rolling(252, min_periods=100).rank(pct=True).values
        
        # Momentum
        out.loc[idx, 'delivery_pct_momentum_5d'] = grp['delivery_pct'].diff(5).values
        out.loc[idx, 'delivery_pct_momentum_20d'] = grp['delivery_pct'].diff(20).values
        
        # Trend (slope of linear regression over 20 days)
        def rolling_slope(series, window):
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    if np.isnan(y).any():
                        slopes.append(np.nan)
                    else:
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
            return pd.Series(slopes, index=series.index)
        
        out.loc[idx, 'delivery_pct_trend_20d'] = rolling_slope(grp['delivery_pct'], 20).values
        
        # High/Low delivery flags
        out.loc[idx, 'high_delivery_flag'] = (grp['delivery_pct'] > grp['delivery_pct'].rolling(252, min_periods=100).quantile(0.7)).astype(float).values
        out.loc[idx, 'low_delivery_flag'] = (grp['delivery_pct'] < grp['delivery_pct'].rolling(252, min_periods=100).quantile(0.3)).astype(float).values
        
        # Consecutive high/low delivery days
        out.loc[idx, 'consecutive_high_delivery'] = (out.loc[idx, 'high_delivery_flag'] == 1).astype(int).groupby((out.loc[idx, 'high_delivery_flag'] == 0).astype(int).cumsum()).cumsum().values
        out.loc[idx, 'consecutive_low_delivery'] = (out.loc[idx, 'low_delivery_flag'] == 1).astype(int).groupby((out.loc[idx, 'low_delivery_flag'] == 0).astype(int).cumsum()).cumsum().values
    
    return out


def fetch_bulk_block_deals(
    from_date: str,
    to_date: str,
    cache_dir: str = "data/external/bulk_block",
) -> pd.DataFrame:
    """
    Fetch bulk and block deals from NSE.
    
    NOTE: This requires NSE API authentication. The public API endpoints
    are not officially documented and may change.
    
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
    
    # NSE bulk/block deals API (requires proper headers and possibly authentication)
    # This is a placeholder - actual implementation needs NSE API access
    logger.warning("Bulk/block deals fetch requires NSE API authentication")
    logger.warning("See: https://www.nseindia.com/api/historical/bulk-deals")
    logger.warning("See: https://www.nseindia.com/api/historical/block-deals")
    
    return pd.DataFrame()


def compute_bulk_block_features(bulk_df: pd.DataFrame, block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from bulk and block deals data.
    
    Features:
    - bulk_deal_count: Number of bulk deals per symbol per day
    - bulk_deal_net_qty: Net quantity (buy - sell) in bulk deals
    - bulk_deal_net_value: Net value in bulk deals
    - block_deal_count: Number of block deals per symbol per day
    - block_deal_net_qty: Net quantity in block deals
    - block_deal_net_value: Net value in block deals
    - institutional_buy_flag: Net institutional buying (bulk + block)
    - institutional_sell_flag: Net institutional selling
    """
    # Placeholder - requires actual bulk/block deal data
    return pd.DataFrame()


def main():
    logger.info("Fetching NSE bhavcopy delivery data...")
    
    # For testing, fetch a small date range
    # In production, fetch the full history
    start_date = "2024-01-01"
    end_date = "2024-01-31"  # Small range for testing
    
    logger.info(f"Fetching bhavcopy from {start_date} to {end_date}")
    bhavcopy_df = fetch_bhavcopy_range(start_date, end_date, delay=0.5)
    
    if bhavcopy_df.empty:
        logger.error("No bhavcopy data fetched")
        logger.info("This is expected if NSE blocks automated access")
        logger.info("For production, use a data vendor or NSE's official API")
        return 1
    
    # Process bhavcopy data
    logger.info("Processing bhavcopy data...")
    processed_df = process_bhavcopy_data(bhavcopy_df)
    
    if processed_df.empty:
        logger.error("Failed to process bhavcopy data")
        return 1
    
    # Compute delivery features
    logger.info("Computing delivery features...")
    delivery_features = compute_delivery_features(processed_df)
    
    if delivery_features.empty:
        logger.error("Failed to compute delivery features")
        return 1
    
    # Save
    output_dir = ROOT / "data" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed bhavcopy
    bhavcopy_path = output_dir / "bhavcopy_processed.parquet"
    processed_df.to_parquet(bhavcopy_path, index=False)
    logger.info(f"Saved processed bhavcopy to {bhavcopy_path}")
    
    # Save delivery features
    features_path = output_dir / "delivery_features.parquet"
    delivery_features.to_parquet(features_path, index=False)
    logger.info(f"Saved delivery features to {features_path}")
    
    logger.info(f"Delivery features shape: {delivery_features.shape}")
    logger.info(f"Columns: {list(delivery_features.columns)}")
    
    # Print sample
    logger.info("\nSample of delivery features (last 10 rows):")
    print(delivery_features.tail(10).to_string())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())