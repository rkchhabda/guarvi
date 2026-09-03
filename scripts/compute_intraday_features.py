#!/usr/bin/env python3
"""Compute intraday microstructure features from high-frequency data.

This script computes microstructure features from intraday OHLCV data
(5-min or 15-min bars) and aggregates them to daily features.

Features computed:
- VWAP and VWAP deviation
- Intraday range and volatility
- Order flow imbalance proxies
- Realized volatility
- First/last hour returns
- Volume distribution (AM/PM ratio)
- Price impact measures
- Microstructure noise estimators

NOTE: This requires intraday data which is not available from standard
Yahoo Finance / jugaad-data. You would need a data vendor providing
intraday data (TrueData, GlobalDatafeeds, Kite Connect historical API, etc.)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("intraday_microstructure")


def compute_intraday_features(
    intraday_df: pd.DataFrame,
    freq: str = '5min',
    market_open: str = '09:15',
    market_close: str = '15:30',
) -> pd.DataFrame:
    """
    Compute daily microstructure features from intraday data.
    
    Parameters
    ----------
    intraday_df : DataFrame
        Intraday OHLCV data with columns:
        - symbol, date, time (or datetime index), open, high, low, close, volume
        Index should be DatetimeIndex or have 'datetime' column
    freq : str
        Frequency of intraday bars ('5min', '15min', '1min')
    market_open : str
        Market open time (HH:MM)
    market_close : str
        Market close time (HH:MM)
    
    Returns
    -------
    DataFrame with daily microstructure features per symbol
    """
    if intraday_df.empty:
        logger.warning("Empty intraday data")
        return pd.DataFrame()
    
    df = intraday_df.copy()
    
    # Ensure datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame must have DatetimeIndex or 'datetime' column")
        return pd.DataFrame()
    
    # Ensure required columns
    required = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    df = df.sort_index()
    
    # Filter to market hours only
    df = df.between_time(market_open, market_close)
    
    daily_features = []
    
    for (symbol, date), group in df.groupby([df['symbol'], df.index.date]):
        date = pd.Timestamp(date)
        group = group.sort_index()
        
        if len(group) < 10:  # Need minimum bars
            continue
        
        # Basic data
        opens = group['open'].values
        highs = group['high'].values
        lows = group['low'].values
        closes = group['close'].values
        volumes = group['volume'].values
        
        # VWAP
        typical_price = (highs + lows + closes) / 3
        vwap = np.sum(typical_price * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else np.nan
        
        # VWAP deviation
        close_price = closes[-1]
        vwap_deviation = (close_price - vwap) / vwap if vwap > 0 else np.nan
        
        # Intraday range
        day_high = highs.max()
        day_low = lows.min()
        intraday_range = (day_high - day_low) / close_price if close_price > 0 else np.nan
        
        # Realized volatility (sum of squared returns)
        returns = np.diff(np.log(closes))
        realized_vol = np.sqrt(np.sum(returns**2)) * np.sqrt(252 * 78)  # Annualized (78 5-min bars per day)
        
        # Realized volatility (alternative: Parkinson)
        # Parkinson volatility uses high-low range
        parkinson_vol = np.sqrt(np.sum((np.log(highs / lows))**2) / (4 * np.log(2) * len(group))) * np.sqrt(252 * 78)
        
        # First hour return (9:15 - 10:15)
        first_hour_end = date.replace(hour=10, minute=15)
        first_hour_data = group[group.index <= first_hour_end]
        if len(first_hour_data) > 1:
            first_hour_ret = (first_hour_data['close'].iloc[-1] - first_hour_data['open'].iloc[0]) / first_hour_data['open'].iloc[0]
        else:
            first_hour_ret = np.nan
        
        # Last hour return (14:30 - 15:30)
        last_hour_start = date.replace(hour=14, minute=30)
        last_hour_data = group[group.index >= last_hour_start]
        if len(last_hour_data) > 1:
            last_hour_ret = (last_hour_data['close'].iloc[-1] - last_hour_data['open'].iloc[0]) / last_hour_data['open'].iloc[0]
        else:
            last_hour_ret = np.nan
        
        # AM/PM volume ratio
        noon = date.replace(hour=12, minute=0)
        am_volume = group[group.index < noon]['volume'].sum()
        pm_volume = group[group.index >= noon]['volume'].sum()
        am_pm_volume_ratio = am_volume / pm_volume if pm_volume > 0 else np.nan
        
        # Volume-weighted momentum (correlation of price change with volume)
        if len(group) > 5:
            price_changes = np.diff(closes)
            vol_weights = volumes[1:] / volumes[1:].sum() if volumes[1:].sum() > 0 else np.ones(len(volumes)-1) / (len(volumes)-1)
            vw_momentum = np.sum(price_changes * vol_weights) / close_price if close_price > 0 else np.nan
        else:
            vw_momentum = np.nan
        
        # Order flow imbalance proxy (using uptick/downtick rule)
        # Uptick: price > previous price, Downtick: price < previous price
        price_changes_tick = np.diff(closes)
        uptick_volume = volumes[1:][price_changes_tick > 0].sum()
        downtick_volume = volumes[1:][price_changes_tick < 0].sum()
        total_tick_volume = uptick_volume + downtick_volume
        ofi = (uptick_volume - downtick_volume) / total_tick_volume if total_tick_volume > 0 else np.nan
        
        # Alternative OFI: using quote rule (bid/ask) - not available without L2 data
        
        # Microstructure noise estimator (Roll's measure)
        # Roll's spread estimator: 2 * sqrt(-cov(r_t, r_{t-1}))
        if len(returns) > 10:
            cov_roll = np.cov(returns[:-1], returns[1:])[0, 1]
            roll_spread = 2 * np.sqrt(-cov_roll) if cov_roll < 0 else np.nan
        else:
            roll_spread = np.nan
        
        # Kyle's lambda (price impact) - simplified
        # Regress returns on signed volume
        if len(group) > 10:
            signed_vol = volumes[1:] * np.sign(price_changes_tick)
            if np.std(signed_vol) > 0:
                kyle_lambda = np.cov(returns, signed_vol)[0, 1] / np.var(signed_vol)
            else:
                kyle_lambda = np.nan
        else:
            kyle_lambda = np.nan
        
        # Amihud illiquidity ratio (daily)
        # |return| / volume (averaged over day)
        daily_return = (closes[-1] - opens[0]) / opens[0] if opens[0] > 0 else 0
        daily_volume = volumes.sum()
        amihud = abs(daily_return) / daily_volume if daily_volume > 0 else np.nan
        
        # Intraday volatility pattern (U-shape)
        # Volatility by time of day
        group = group.copy()
        group['returns'] = group['close'].pct_change()
        group['hour'] = group.index.hour
        hourly_vol = group.groupby('hour')['returns'].std()
        
        # Morning volatility (9:15-11:00) vs Afternoon (13:00-15:30)
        morning_vol = hourly_vol.loc[9:11].mean() if any(h in hourly_vol.index for h in [9, 10, 11]) else np.nan
        afternoon_vol = hourly_vol.loc[13:15].mean() if any(h in hourly_vol.index for h in [13, 14, 15]) else np.nan
        vol_ratio_morning_afternoon = morning_vol / afternoon_vol if afternoon_vol > 0 else np.nan
        
        # Close-to-close return
        cc_return = (closes[-1] - opens[0]) / opens[0] if opens[0] > 0 else np.nan
        
        # Open-to-close return
        oc_return = (closes[-1] - opens[0]) / opens[0] if opens[0] > 0 else np.nan
        
        # Overnight return (previous close to today's open) - need previous day
        # This would be computed when merging across days
        
        daily_features.append({
            'symbol': symbol,
            'date': date,
            'vwap': vwap,
            'vwap_deviation': vwap_deviation,
            'intraday_range': intraday_range,
            'realized_volatility': realized_vol,
            'parkinson_volatility': parkinson_vol,
            'first_hour_return': first_hour_ret,
            'last_hour_return': last_hour_ret,
            'am_pm_volume_ratio': am_pm_volume_ratio,
            'volume_weighted_momentum': vw_momentum,
            'order_flow_imbalance': ofi,
            'roll_spread': roll_spread,
            'kyle_lambda': kyle_lambda,
            'amihud_illiquidity': amihud,
            'morning_volatility': morning_vol,
            'afternoon_volatility': afternoon_vol,
            'vol_ratio_morning_afternoon': vol_ratio_morning_afternoon,
            'close_to_close_return': cc_return,
            'open_to_close_return': oc_return,
            'total_volume': daily_volume,
            'num_trades': len(group),
        })
    
    features_df = pd.DataFrame(daily_features)
    
    if features_df.empty:
        return features_df
    
    # Add rolling features per symbol
    features_df = features_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    for symbol, group in features_df.groupby('symbol'):
        idx = group.index
        grp = group.sort_values('date')
        
        # Rolling statistics for key features
        for feat in ['vwap_deviation', 'realized_volatility', 'order_flow_imbalance',
                     'roll_spread', 'kyle_lambda', 'amihud_illiquidity',
                     'first_hour_return', 'last_hour_return', 'am_pm_volume_ratio']:
            if feat in grp.columns:
                # 5-day MA
                features_df.loc[idx, f'{feat}_5d_ma'] = grp[feat].rolling(5, min_periods=5).mean().values
                # 20-day MA
                features_df.loc[idx, f'{feat}_20d_ma'] = grp[feat].rolling(20, min_periods=20).mean().values
                # Z-score (20-day)
                ma = grp[feat].rolling(20, min_periods=10).mean()
                std = grp[feat].rolling(20, min_periods=10).std(ddof=0)
                features_df.loc[idx, f'{feat}_zscore_20d'] = ((grp[feat] - ma) / std.replace(0, np.nan)).values
        
        # Volatility regime
        features_df.loc[idx, 'realized_vol_regime'] = (
            grp['realized_volatility'] / grp['realized_volatility'].rolling(60, min_periods=20).mean()
        ).values
        
        # OFI regime
        features_df.loc[idx, 'ofi_regime'] = (
            grp['order_flow_imbalance'] / grp['order_flow_imbalance'].rolling(20, min_periods=10).std(ddof=0).replace(0, np.nan)
        ).values
    
    return features_df


def load_intraday_data_from_vendor(
    symbol: str,
    start_date: str,
    end_date: str,
    vendor: str = 'kite',  # 'kite', 'truedata', 'globaldf', 'custom'
    freq: str = '5min',
) -> pd.DataFrame:
    """
    Load intraday data from a data vendor.
    
    This is a template - implement based on your vendor's API.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., 'NIFTY', 'RELIANCE')
    start_date : str
        Start date in YYYY-MM-DD
    end_date : str
        End date in YYYY-MM-DD
    vendor : str
        Data vendor name
    freq : str
        Bar frequency
    
    Returns
    -------
    DataFrame with intraday OHLCV data
    """
    logger.warning(f"Intraday data loading from {vendor} not implemented")
    logger.warning("Implement this function based on your data vendor's API")
    
    # Example structure for Kite Connect:
    # from kiteconnect import KiteConnect
    # kite = KiteConnect(api_key=YOUR_API_KEY)
    # kite.set_access_token(YOUR_ACCESS_TOKEN)
    # data = kite.historical_data(instrument_token, from_date, to_date, freq)
    
    return pd.DataFrame()


def main():
    logger.info("Intraday microstructure feature computation")
    logger.warning("This requires intraday data from a vendor")
    logger.warning("Implement load_intraday_data_from_vendor() for your data source")
    
    # Example usage (with synthetic data for testing)
    # Create synthetic intraday data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='B')
    all_bars = []
    
    for date in dates:
        # Generate 5-min bars for market hours (9:15-15:30 = 375 min = 75 bars)
        times = pd.date_range(
            date.replace(hour=9, minute=15),
            date.replace(hour=15, minute=30),
            freq='5min'
        )
        
        # Synthetic price path
        np.random.seed(int(date.timestamp()) % 1000)
        n = len(times)
        returns = np.random.normal(0, 0.001, n)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Add some intraday pattern (higher vol at open/close)
        vol_pattern = np.ones(n)
        vol_pattern[:10] *= 2  # Morning volatility
        vol_pattern[-10:] *= 1.5  # Afternoon volatility
        returns = returns * vol_pattern
        price = 100 * np.exp(np.cumsum(returns))
        
        # OHLC from close prices (simplified)
        for i, t in enumerate(times):
            o = price[i] * (1 + np.random.normal(0, 0.0005))
            h = max(o, price[i]) * (1 + abs(np.random.normal(0, 0.001)))
            l = min(o, price[i]) * (1 - abs(np.random.normal(0, 0.001)))
            c = price[i]
            v = np.random.randint(1000, 10000)
            all_bars.append({
                'symbol': 'TEST',
                'datetime': t,
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
            })
    
    intraday_df = pd.DataFrame(all_bars)
    intraday_df = intraday_df.set_index('datetime')
    
    logger.info(f"Generated synthetic intraday data: {len(intraday_df)} bars")
    
    # Compute features
    features = compute_intraday_features(intraday_df, freq='5min')
    
    if features.empty:
        logger.error("Failed to compute features")
        return 1
    
    logger.info(f"Computed features: {features.shape}")
    logger.info(f"Columns: {list(features.columns)}")
    
    # Save
    output_dir = ROOT / "data" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_path = output_dir / "intraday_microstructure_features.parquet"
    features.to_parquet(features_path, index=False)
    logger.info(f"Saved intraday features to {features_path}")
    
    print(features.tail(10).to_string())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())