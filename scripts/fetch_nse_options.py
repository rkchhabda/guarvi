#!/usr/bin/env python3
"""Fetch and process NSE options market data.

This script fetches NSE option chain data for NIFTY and Bank NIFTY,
computes daily options features (IV, PCR, max-pain, etc.), and saves
them for integration with the main feature pipeline.

NOTE: This requires NSE API access. For now, it provides a framework
that can be extended when API credentials are available.
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
logger = logging.getLogger("fetch_nse_options")


def fetch_nse_option_chain_historical(
    symbol: str = "NIFTY",
    start_date: str = "2020-01-01",
    end_date: str = None,
    cache_dir: str = "data/external/options",
) -> pd.DataFrame:
    """
    Fetch historical NSE option chain data.
    
    This is a placeholder implementation. In production, you would:
    1. Use NSE's official API (requires subscription)
    2. Use a paid data vendor (TrueData, GlobalDatafeeds, etc.)
    3. Scrape NSE's option chain page (fragile, may violate ToS)
    
    For now, this creates a synthetic template showing the expected structure.
    """
    cache_path = Path(cache_dir) / f"{symbol}_option_chain_historical.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded option chain from cache: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    logger.warning("NSE option chain fetch requires NSE API subscription")
    logger.warning("This is a placeholder - implement with your data vendor")
    
    # Return empty DataFrame with expected structure
    return pd.DataFrame(columns=[
        'date', 'symbol', 'expiry', 'strike', 'option_type',
        'open_interest', 'change_in_oi', 'volume', 'iv',
        'last_price', 'bid_price', 'ask_price', 'spot_price'
    ])


def compute_daily_options_features(option_chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily options features from option chain data.
    
    Expected input columns:
    - date, symbol, expiry, strike, option_type (CE/PE)
    - open_interest, volume, iv, last_price, spot_price
    
    Output features per symbol per date:
    - atm_iv: At-the-money IV (interpolated)
    - iv_skew_25d: 25-delta risk reversal (put IV - call IV)
    - iv_smile_curvature: Butterfly (call_25d + put_25d - 2*atm) / 2
    - pcr_oi: Put-Call ratio (open interest)
    - pcr_volume: Put-Call ratio (volume)
    - max_pain: Strike with minimum total option value
    - iv_percentile_20d: IV rank over 20 days
    - iv_term_structure: Near-term IV - Far-term IV
    - atm_straddle_price: ATM call + ATM put price
    - straddle_return_1d: 1-day return of ATM straddle
    """
    if option_chain_df.empty:
        logger.warning("Empty option chain data")
        return pd.DataFrame()
    
    # Ensure required columns
    required = ['date', 'symbol', 'expiry', 'strike', 'option_type', 
                'open_interest', 'volume', 'iv', 'spot_price']
    for col in required:
        if col not in option_chain_df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    df = option_chain_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['expiry'] = pd.to_datetime(df['expiry'])
    df = df.sort_values(['symbol', 'date', 'expiry', 'strike', 'option_type'])
    
    daily_features = []
    
    for (symbol, date), group in df.groupby(['symbol', 'date']):
        spot = group['spot_price'].iloc[0]
        
        # Get near-term expiry (first expiry after date)
        expiries = sorted(group['expiry'].unique())
        if len(expiries) < 2:
            continue
        near_expiry = expiries[0]
        far_expiry = expiries[1] if len(expiries) > 1 else expiries[0]
        
        near_chain = group[group['expiry'] == near_expiry]
        far_chain = group[group['expiry'] == far_expiry]
        
        # Separate calls and puts
        calls = near_chain[near_chain['option_type'] == 'CE']
        puts = near_chain[near_chain['option_type'] == 'PE']
        
        if calls.empty or puts.empty:
            continue
        
        # Find ATM strike (closest to spot)
        atm_strike = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]['strike'].values[0]
        
        # ATM IV (interpolate between strikes)
        atm_call_iv = calls[calls['strike'] == atm_strike]['iv'].values
        atm_put_iv = puts[puts['strike'] == atm_strike]['iv'].values
        
        if len(atm_call_iv) > 0 and len(atm_put_iv) > 0:
            atm_iv = (atm_call_iv[0] + atm_put_iv[0]) / 2
        else:
            # Interpolate
            atm_iv = np.nan
        
        # 25-delta strikes (approximate)
        # For simplicity, use strikes at ~25 delta (roughly ATM ± 1-2 strikes)
        # In production, use proper delta calculation from IV and time to expiry
        
        # PCR
        total_call_oi = calls['open_interest'].sum()
        total_put_oi = puts['open_interest'].sum()
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan
        pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan
        
        # Max Pain calculation
        # For each strike, compute total option value if expired at that strike
        all_strikes = sorted(group['strike'].unique())
        max_pain_strike = all_strikes[0]
        min_pain = float('inf')
        
        for strike in all_strikes:
            call_val = calls[calls['strike'] <= strike]['open_interest'].sum() * (strike - calls[calls['strike'] <= strike]['strike']).clip(lower=0)
            put_val = puts[puts['strike'] >= strike]['open_interest'].sum() * (puts[puts['strike'] >= strike]['strike'] - strike).clip(lower=0)
            total_pain = call_val.sum() + put_val.sum()
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike
        
        # ATM Straddle
        atm_call_price = calls[calls['strike'] == atm_strike]['last_price'].values
        atm_put_price = puts[puts['strike'] == atm_strike]['last_price'].values
        atm_straddle = (atm_call_price[0] if len(atm_call_price) > 0 else 0) + \
                       (atm_put_price[0] if len(atm_put_price) > 0 else 0)
        
        # IV term structure
        near_atm_iv = near_chain[near_chain['strike'] == atm_strike]['iv'].mean()
        far_atm_iv = far_chain[far_chain['strike'] == atm_strike]['iv'].mean()
        iv_term_structure = near_atm_iv - far_atm_iv if not np.isnan(near_atm_iv) and not np.isnan(far_atm_iv) else np.nan
        
        daily_features.append({
            'date': date,
            'symbol': symbol,
            'atm_iv': atm_iv,
            'pcr_oi': pcr_oi,
            'pcr_volume': pcr_volume,
            'max_pain': max_pain_strike,
            'max_pain_distance': (max_pain_strike - spot) / spot,
            'atm_straddle_price': atm_straddle,
            'iv_term_structure': iv_term_structure,
            'near_expiry_days': (near_expiry - date).days,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'total_call_volume': total_call_vol,
            'total_put_volume': total_put_vol,
        })
    
    features_df = pd.DataFrame(daily_features)
    
    if features_df.empty:
        return features_df
    
    # Add rolling features
    features_df = features_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    for symbol, group in features_df.groupby('symbol'):
        idx = group.index
        # IV percentile
        features_df.loc[idx, 'iv_percentile_20d'] = group['atm_iv'].rolling(20, min_periods=10).rank(pct=True).values
        features_df.loc[idx, 'iv_percentile_60d'] = group['atm_iv'].rolling(60, min_periods=20).rank(pct=True).values
        
        # IV z-score
        iv_ma_20 = group['atm_iv'].rolling(20, min_periods=10).mean()
        iv_std_20 = group['atm_iv'].rolling(20, min_periods=10).std(ddof=0)
        features_df.loc[idx, 'iv_zscore_20d'] = ((group['atm_iv'] - iv_ma_20) / iv_std_20.replace(0, np.nan)).values
        
        # PCR momentum
        features_df.loc[idx, 'pcr_oi_5d_ma'] = group['pcr_oi'].rolling(5, min_periods=5).mean().values
        features_df.loc[idx, 'pcr_oi_momentum'] = group['pcr_oi'].diff(5).values
        
        # Max pain distance momentum
        features_df.loc[idx, 'max_pain_dist_5d_ma'] = group['max_pain_distance'].rolling(5, min_periods=5).mean().values
        
        # Straddle return (need previous day's straddle)
        features_df.loc[idx, 'straddle_return_1d'] = group['atm_straddle_price'].pct_change().values
    
    return features_df


def main():
    logger.info("Fetching NSE options data...")
    
    # Fetch for NIFTY and BANKNIFTY
    for symbol in ['NIFTY', 'BANKNIFTY']:
        logger.info(f"Processing {symbol}...")
        
        option_chain = fetch_nse_option_chain_historical(
            symbol=symbol,
            start_date="2020-01-01",
        )
        
        if option_chain.empty:
            logger.warning(f"No option chain data for {symbol}")
            continue
        
        features = compute_daily_options_features(option_chain)
        
        if features.empty:
            logger.warning(f"No features computed for {symbol}")
            continue
        
        # Save
        output_dir = ROOT / "data" / "external"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = output_dir / f"{symbol.lower()}_options_features.parquet"
        features.to_parquet(features_path, index=False)
        logger.info(f"Saved {symbol} options features to {features_path}")
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Columns: {list(features.columns)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())