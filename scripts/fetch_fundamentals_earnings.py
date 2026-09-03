#!/usr/bin/env python3
"""Fetch and process fundamentals and earnings data.

This script provides a framework for integrating fundamental data
(earnings, financials, analyst estimates) and earnings-related features.

Sources (require paid subscription or web scraping):
- Tickertape API
- Trendlyne API
- Screener.in (web scraping)
- NSE corporate announcements
- BSE corporate announcements
- Financial data vendors (Refinitiv, Bloomberg, FactSet)

Features computed:
- Days to/from earnings
- Earnings surprise (actual vs estimate)
- Earnings momentum (surprise trend)
- Guidance revisions
- Analyst estimate revisions
- Fundamental ratios (PE, PB, ROE, etc.)
- Financial health scores
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fundamentals_earnings")


# ---------------------------------------------------------------------------
# Earnings Calendar
# ---------------------------------------------------------------------------

def fetch_earnings_calendar(
    symbols: List[str],
    start_date: str,
    end_date: str,
    source: str = 'tickertape',  # 'tickertape', 'trendlyne', 'screener', 'nse', 'custom'
    cache_dir: str = "data/external/earnings",
) -> pd.DataFrame:
    """
    Fetch earnings announcement dates and estimates.
    
    This is a template - implement based on your data source.
    
    Expected output columns:
    - symbol, earnings_date, fiscal_period, fiscal_year
    - estimated_eps, actual_eps, surprise_pct
    - revenue_estimate, revenue_actual, revenue_surprise_pct
    - guidance_next_quarter, guidance_fy
    - conference_call_date
    """
    cache_path = Path(cache_dir) / f"earnings_calendar_{start_date}_to_{end_date}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded earnings calendar from cache: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    logger.warning(f"Earnings calendar fetch from {source} not implemented")
    logger.warning("Implement based on your data vendor:")
    logger.warning("  - Tickertape: API (paid)")
    logger.warning("  - Trendlyne: API (paid) or web scraping")
    logger.warning("  - Screener.in: Web scraping")
    logger.warning("  - NSE/BSE: Corporate announcements API")
    logger.warning("  - Refinitiv/Bloomberg/FactSet: Paid terminals")
    
    # Return empty DataFrame with expected structure
    return pd.DataFrame(columns=[
        'symbol', 'earnings_date', 'fiscal_period', 'fiscal_year',
        'estimated_eps', 'actual_eps', 'surprise_pct',
        'revenue_estimate', 'revenue_actual', 'revenue_surprise_pct',
        'guidance_next_quarter', 'guidance_fy',
        'conference_call_date', 'source'
    ])


def fetch_analyst_estimates(
    symbols: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "data/external/analyst_estimates",
) -> pd.DataFrame:
    """
    Fetch analyst consensus estimates and revisions.
    
    Expected output columns:
    - symbol, date, estimate_type (eps_1q, eps_fy1, eps_fy2, revenue_fy1, etc.)
    - mean_estimate, median_estimate, high_estimate, low_estimate
    - num_analysts, revision_up, revision_down
    - target_price_mean, target_price_median
    """
    cache_path = Path(cache_dir) / f"analyst_estimates_{start_date}_to_{end_date}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    
    logger.warning("Analyst estimates fetch not implemented - requires paid data source")
    return pd.DataFrame()


def fetch_fundamental_ratios(
    symbols: List[str],
    date: str,
    cache_dir: str = "data/external/fundamentals",
) -> pd.DataFrame:
    """
    Fetch fundamental ratios for a specific date (point-in-time).
    
    Expected output columns:
    - symbol, date, pe_ratio, pb_ratio, ps_ratio, ev_ebitda
    - roe, roa, roic, debt_to_equity, current_ratio
    - profit_margin, operating_margin, net_margin
    - revenue_growth_yoy, earnings_growth_yoy
    - dividend_yield, payout_ratio
    - market_cap, enterprise_value
    """
    cache_path = Path(cache_dir) / f"fundamental_ratios_{date}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    
    logger.warning("Fundamental ratios fetch not implemented - requires paid data source")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_earnings_features(
    earnings_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute earnings-related features for each symbol/date.
    
    Parameters
    ----------
    earnings_df : DataFrame
        Earnings calendar with columns: symbol, earnings_date, estimated_eps,
        actual_eps, surprise_pct, revenue_surprise_pct, guidance_next_quarter
    price_df : DataFrame
        Daily OHLCV data with columns: symbol, date, close
    
    Returns
    -------
    DataFrame with earnings features per symbol per date
    """
    if earnings_df.empty or price_df.empty:
        return pd.DataFrame()
    
    earnings = earnings_df.copy()
    prices = price_df.copy()
    
    earnings['earnings_date'] = pd.to_datetime(earnings['earnings_date'])
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Sort
    earnings = earnings.sort_values(['symbol', 'earnings_date'])
    prices = prices.sort_values(['symbol', 'date'])
    
    daily_features = []
    
    for symbol in earnings['symbol'].unique():
        sym_earnings = earnings[earnings['symbol'] == symbol].copy()
        sym_prices = prices[prices['symbol'] == symbol].copy()
        
        if sym_earnings.empty or sym_prices.empty:
            continue
        
        # For each trading day, find nearest earnings events
        for _, price_row in sym_prices.iterrows():
            date = price_row['date']
            close = price_row['close']
            
            # Find past earnings (most recent before this date)
            past_earnings = sym_earnings[sym_earnings['earnings_date'] <= date]
            if not past_earnings.empty:
                last_earnings = past_earnings.iloc[-1]
                days_since_earnings = (date - last_earnings['earnings_date']).days
                
                # Earnings surprise
                surprise = last_earnings.get('surprise_pct', np.nan)
                revenue_surprise = last_earnings.get('revenue_surprise_pct', np.nan)
                
                # Post-earnings drift (return since earnings)
                earnings_close = sym_prices[sym_prices['date'] == last_earnings['earnings_date']]['close']
                if not earnings_close.empty:
                    post_earnings_ret = (close - earnings_close.values[0]) / earnings_close.values[0]
                else:
                    post_earnings_ret = np.nan
            else:
                days_since_earnings = np.nan
                surprise = np.nan
                revenue_surprise = np.nan
                post_earnings_ret = np.nan
            
            # Find future earnings (next after this date)
            future_earnings = sym_earnings[sym_earnings['earnings_date'] > date]
            if not future_earnings.empty:
                next_earnings = future_earnings.iloc[0]
                days_to_earnings = (next_earnings['earnings_date'] - date).days
                next_estimated_eps = next_earnings.get('estimated_eps', np.nan)
            else:
                days_to_earnings = np.nan
                next_estimated_eps = np.nan
            
            # Earnings momentum (trend of surprises)
            recent_earnings = past_earnings.tail(4)  # Last 4 quarters
            if len(recent_earnings) >= 2:
                surprise_trend = recent_earnings['surprise_pct'].diff().mean()
                surprise_consistency = (recent_earnings['surprise_pct'] > 0).mean()
            else:
                surprise_trend = np.nan
                surprise_consistency = np.nan
            
            # Guidance
            guidance = last_earnings.get('guidance_next_quarter', np.nan) if not past_earnings.empty else np.nan
            
            daily_features.append({
                'symbol': symbol,
                'date': date,
                'days_to_earnings': days_to_earnings,
                'days_since_earnings': days_since_earnings,
                'last_earnings_surprise': surprise,
                'last_revenue_surprise': revenue_surprise,
                'post_earnings_return': post_earnings_ret,
                'next_estimated_eps': next_estimated_eps,
                'earnings_surprise_trend': surprise_trend,
                'earnings_surprise_consistency': surprise_consistency,
                'guidance_next_quarter': guidance,
                'earnings_announcement_week': (days_to_earnings >= 0) & (days_to_earnings <= 5),
                'post_earnings_week': (days_since_earnings >= 0) & (days_since_earnings <= 5),
            })
    
    features_df = pd.DataFrame(daily_features)
    
    if features_df.empty:
        return features_df
    
    # Add rolling features
    features_df = features_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    for symbol, group in features_df.groupby('symbol'):
        idx = group.index
        grp = group.sort_values('date')
        
        # Rolling surprise average
        features_df.loc[idx, 'surprise_4q_avg'] = grp['last_earnings_surprise'].rolling(4, min_periods=1).mean().values
        features_df.loc[idx, 'surprise_4q_std'] = grp['last_earnings_surprise'].rolling(4, min_periods=2).std(ddof=0).values
        
        # Earnings season flag (quarterly)
        features_df.loc[idx, 'earnings_season'] = (
            grp['date'].dt.month.isin([1, 4, 7, 10])  # Jan, Apr, Jul, Oct
        ).astype(float).values
    
    return features_df


def compute_analyst_features(
    analyst_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute analyst estimate revision features.
    
    Features:
    - eps_revision_1m: 1-month change in consensus EPS estimate
    - eps_revision_3m: 3-month change in consensus EPS estimate
    - target_price_revision_1m: 1-month change in target price
    - num_analysts_change: Change in number of analysts covering
    - estimate_dispersion: (high - low) / mean (uncertainty)
    - upgrade_downgrade_ratio: Upgrades / Downgrades
    """
    if analyst_df.empty or price_df.empty:
        return pd.DataFrame()
    
    # Analyst data should have: symbol, date, estimate_type, mean_estimate, 
    # high_estimate, low_estimate, num_analysts, target_price_mean
    
    analyst = analyst_df.copy()
    analyst['date'] = pd.to_datetime(analyst['date'])
    analyst = analyst.sort_values(['symbol', 'date', 'estimate_type'])
    
    # Focus on FY1 EPS estimates
    fy1_eps = analyst[analyst['estimate_type'].str.contains('eps_fy1|eps_1y', case=False, na=False)]
    
    if fy1_eps.empty:
        return pd.DataFrame()
    
    daily_features = []
    
    for symbol in fy1_eps['symbol'].unique():
        sym_estimates = fy1_eps[fy1_eps['symbol'] == symbol].copy()
        sym_prices = price_df[price_df['symbol'] == symbol].copy()
        sym_prices['date'] = pd.to_datetime(sym_prices['date'])
        sym_prices = sym_prices.sort_values('date')
        
        for _, price_row in sym_prices.iterrows():
            date = price_row['date']
            
            # Get latest estimate before this date
            past_estimates = sym_estimates[sym_estimates['date'] <= date]
            if past_estimates.empty:
                continue
            
            latest = past_estimates.iloc[-1]
            
            # Revisions
            est_1m_ago = sym_estimates[
                (sym_estimates['date'] <= date) & 
                (sym_estimates['date'] >= date - timedelta(days=30))
            ]
            est_3m_ago = sym_estimates[
                (sym_estimates['date'] <= date) & 
                (sym_estimates['date'] >= date - timedelta(days=90))
            ]
            
            eps_rev_1m = np.nan
            eps_rev_3m = np.nan
            if len(est_1m_ago) >= 2:
                eps_rev_1m = (est_1m_ago.iloc[-1]['mean_estimate'] - est_1m_ago.iloc[0]['mean_estimate']) / abs(est_1m_ago.iloc[0]['mean_estimate'])
            if len(est_3m_ago) >= 2:
                eps_rev_3m = (est_3m_ago.iloc[-1]['mean_estimate'] - est_3m_ago.iloc[0]['mean_estimate']) / abs(est_3m_ago.iloc[0]['mean_estimate'])
            
            # Dispersion
            dispersion = (latest['high_estimate'] - latest['low_estimate']) / abs(latest['mean_estimate']) if latest['mean_estimate'] != 0 else np.nan
            
            daily_features.append({
                'symbol': symbol,
                'date': date,
                'eps_estimate': latest['mean_estimate'],
                'eps_high': latest['high_estimate'],
                'eps_low': latest['low_estimate'],
                'eps_dispersion': dispersion,
                'eps_revision_1m': eps_rev_1m,
                'eps_revision_3m': eps_rev_3m,
                'num_analysts': latest['num_analysts'],
                'target_price': latest.get('target_price_mean', np.nan),
            })
    
    features_df = pd.DataFrame(daily_features)
    
    if features_df.empty:
        return features_df
    
    # Rolling features
    features_df = features_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    for symbol, group in features_df.groupby('symbol'):
        idx = group.index
        grp = group.sort_values('date')
        
        features_df.loc[idx, 'eps_revision_1m_ma'] = grp['eps_revision_1m'].rolling(5, min_periods=3).mean().values
        features_df.loc[idx, 'target_price_vs_current'] = (grp['target_price'] - price_df[price_df['symbol']==symbol].set_index('date').reindex(grp['date'])['close']) / price_df[price_df['symbol']==symbol].set_index('date').reindex(grp['date'])['close']
    
    return features_df


def compute_fundamental_features(
    fundamental_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fundamental valuation and quality features.
    
    Features:
    - pe_ratio, pb_ratio, ps_ratio, ev_ebitda
    - pe_percentile_1y: PE percentile vs 1-year history
    - roe, roa, roic
    - debt_to_equity, current_ratio
    - profit_margin, operating_margin
    - revenue_growth_yoy, earnings_growth_yoy
    - f_score: Piotroski F-Score (9-point financial health)
    - magic_formula_rank: Greenblatt's Magic Formula rank (ROIC + EY)
    """
    if fundamental_df.empty or price_df.empty:
        return pd.DataFrame()
    
    fund = fundamental_df.copy()
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'date'])
    
    prices = price_df.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['symbol', 'date'])
    
    daily_features = []
    
    for symbol in fund['symbol'].unique():
        sym_fund = fund[fund['symbol'] == symbol].copy()
        sym_prices = prices[prices['symbol'] == symbol].copy()
        
        if sym_fund.empty or sym_prices.empty:
            continue
        
        # Forward fill fundamental data to daily frequency
        # Fundamentals are typically quarterly/annual, so forward fill
        sym_fund_daily = sym_fund.set_index('date').reindex(sym_prices['date']).ffill().reset_index()
        sym_fund_daily = sym_fund_daily.rename(columns={'index': 'date'})
        sym_fund_daily['symbol'] = symbol
        
        # Merge with prices
        merged = sym_fund_daily.merge(sym_prices[['date', 'close']], on='date', how='left')
        
        for _, row in merged.iterrows():
            if pd.isna(row.get('close')):
                continue
            
            # Valuation ratios (already in fundamental data)
            pe = row.get('pe_ratio', np.nan)
            pb = row.get('pb_ratio', np.nan)
            ps = row.get('ps_ratio', np.nan)
            ev_ebitda = row.get('ev_ebitda', np.nan)
            
            # Quality metrics
            roe = row.get('roe', np.nan)
            roa = row.get('roa', np.nan)
            roic = row.get('roic', np.nan)
            debt_equity = row.get('debt_to_equity', np.nan)
            current_ratio = row.get('current_ratio', np.nan)
            
            # Profitability
            profit_margin = row.get('profit_margin', np.nan)
            operating_margin = row.get('operating_margin', np.nan)
            
            # Growth
            rev_growth = row.get('revenue_growth_yoy', np.nan)
            earn_growth = row.get('earnings_growth_yoy', np.nan)
            
            # Piotroski F-Score components (simplified)
            f_score = 0
            if roe > 0: f_score += 1
            if roa > 0: f_score += 1
            if profit_margin > 0: f_score += 1
            if operating_margin > 0: f_score += 1
            if debt_equity < 1: f_score += 1  # Low leverage
            if current_ratio > 1: f_score += 1  # Liquidity
            if rev_growth > 0: f_score += 1
            if earn_growth > 0: f_score += 1
            # Would need more data for full 9-point score
            
            # Magic Formula: Earnings Yield (EBIT/EV) + ROIC rank
            # Simplified: use 1/PE as earnings yield proxy
            ey = 1 / pe if pe > 0 else np.nan
            
            daily_features.append({
                'symbol': symbol,
                'date': row['date'],
                'pe_ratio': pe,
                'pb_ratio': pb,
                'ps_ratio': ps,
                'ev_ebitda': ev_ebitda,
                'earnings_yield': ey,
                'roe': roe,
                'roa': roa,
                'roic': roic,
                'debt_to_equity': debt_equity,
                'current_ratio': current_ratio,
                'profit_margin': profit_margin,
                'operating_margin': operating_margin,
                'revenue_growth_yoy': rev_growth,
                'earnings_growth_yoy': earn_growth,
                'f_score_proxy': f_score,
                'dividend_yield': row.get('dividend_yield', np.nan),
            })
    
    features_df = pd.DataFrame(daily_features)
    
    if features_df.empty:
        return features_df
    
    # Add percentile ranks (cross-sectional)
    features_df = features_df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    for date, group in features_df.groupby('date'):
        idx = group.index
        for col in ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'earnings_yield',
                    'roe', 'roic', 'profit_margin', 'revenue_growth_yoy', 'earnings_growth_yoy']:
            if col in group.columns:
                features_df.loc[idx, f'{col}_rank'] = group[col].rank(pct=True).values
    
    return features_df


# ---------------------------------------------------------------------------
# NSE Corporate Announcements (Free Source)
# ---------------------------------------------------------------------------

def fetch_nse_corporate_announcements(
    symbol: str,
    from_date: str,
    to_date: str,
    cache_dir: str = "data/external/nse_announcements",
) -> pd.DataFrame:
    """
    Fetch corporate announcements from NSE.
    
    NSE provides corporate announcements via their API:
    https://www.nseindia.com/api/corporate-announcements?symbol=SYMBOL&from_date=YYYY-MM-DD&to_date=YYYY-MM-DD
    
    Announcement types include:
    - Board Meeting
    - Financial Results
    - Dividend
    - Bonus
    - Split
    - Merger/Amalgamation
    - etc.
    """
    cache_path = Path(cache_dir) / f"announcements_{symbol}_{from_date}_to_{to_date}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    
    # NSE API requires proper headers and session
    url = f"https://www.nseindia.com/api/corporate-announcements?symbol={symbol}&from_date={from_date}&to_date={to_date}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nseindia.com/',
    }
    
    try:
        import requests
        session = requests.Session()
        session.get('https://www.nseindia.com', headers=headers, timeout=10)  # Get cookies
        response = session.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df.to_csv(cache_path, index=False)
            logger.info(f"Fetched {len(df)} announcements for {symbol}")
            return df
    except Exception as e:
        logger.warning(f"Failed to fetch NSE announcements for {symbol}: {e}")
    
    return pd.DataFrame()


def process_nse_announcements(announcements_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process NSE corporate announcements into features.
    
    Features:
    - board_meeting_flag: Board meeting scheduled
    - results_announcement_flag: Financial results announced
    - dividend_flag: Dividend announced
    - bonus_flag: Bonus issue announced
    - split_flag: Stock split announced
    - corporate_action_flag: Any corporate action
    - days_to_board_meeting: Days until next board meeting
    - days_since_results: Days since last results
    """
    if announcements_df.empty:
        return pd.DataFrame()
    
    df = announcements_df.copy()
    
    # NSE announcement columns vary, but typically include:
    # symbol, subject, description, broadcast_date, attachment_url
    
    # Categorize announcements
    df['subject_lower'] = df.get('subject', '').astype(str).str.lower()
    df['desc_lower'] = df.get('description', '').astype(str).str.lower()
    
    df['is_board_meeting'] = df['subject_lower'].str.contains('board meeting|board_meeting', case=False, na=False)
    df['is_results'] = df['subject_lower'].str.contains('financial result|quarterly result|annual result|result', case=False, na=False)
    df['is_dividend'] = df['subject_lower'].str.contains('dividend', case=False, na=False)
    df['is_bonus'] = df['subject_lower'].str.contains('bonus', case=False, na=False)
    df['is_split'] = df['subject_lower'].str.contains('split|sub-division', case=False, na=False)
    df['is_merger'] = df['subject_lower'].str.contains('merger|amalgamation|scheme', case=False, na=False)
    
    df['broadcast_date'] = pd.to_datetime(df.get('broadcast_date', df.get('date', '')), errors='coerce')
    df = df.dropna(subset=['broadcast_date'])
    df = df.sort_values(['symbol', 'broadcast_date'])
    
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Fundamentals/Earnings data pipeline")
    logger.warning("This requires paid data subscriptions or web scraping")
    logger.warning("Implement fetch_* functions for your data sources")
    
    # Example: Create synthetic earnings data for testing
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
    
    # Synthetic earnings (quarterly)
    earnings_data = []
    for symbol in symbols:
        for year in [2023, 2024]:
            for quarter in [1, 4, 7, 10]:  # Approximate quarter ends
                earnings_date = pd.Timestamp(f'{year}-{quarter:02d}-15') + pd.offsets.BDay(np.random.randint(-5, 5))
                if earnings_date > pd.Timestamp('2024-12-31'):
                    continue
                estimated = np.random.normal(20, 5)
                actual = estimated + np.random.normal(0, 3)
                surprise = (actual - estimated) / abs(estimated) * 100
                earnings_data.append({
                    'symbol': symbol,
                    'earnings_date': earnings_date,
                    'fiscal_period': f'Q{(quarter-1)//3+1}',
                    'fiscal_year': year,
                    'estimated_eps': estimated,
                    'actual_eps': actual,
                    'surprise_pct': surprise,
                    'revenue_surprise_pct': np.random.normal(0, 5),
                    'guidance_next_quarter': np.random.choice(['positive', 'neutral', 'negative']),
                })
    
    earnings_df = pd.DataFrame(earnings_data)
    
    # Synthetic price data
    price_data = []
    for symbol in symbols:
        price = 1000
        for date in dates:
            ret = np.random.normal(0.0005, 0.015)
            price *= (1 + ret)
            price_data.append({'symbol': symbol, 'date': date, 'close': price})
    
    price_df = pd.DataFrame(price_data)
    
    # Compute features
    logger.info("Computing earnings features...")
    earnings_features = compute_earnings_features(earnings_df, price_df)
    
    if not earnings_features.empty:
        output_dir = ROOT / "data" / "external"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = output_dir / "earnings_features.parquet"
        earnings_features.to_parquet(features_path, index=False)
        logger.info(f"Saved earnings features to {features_path}")
        logger.info(f"Shape: {earnings_features.shape}")
        print(earnings_features.tail(10).to_string())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())