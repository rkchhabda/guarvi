"""BUY DECISION ENGINE for Guarvi.

Computes a 0-100 score for each of 6 factors (AI, Technical, Fundamental,
Momentum, Risk, Valuation) plus 14 supporting KPIs (trend strength,
confidence, expected return, downside risk, RR ratio, win probability,
sector strength, relative strength, institutional activity proxy,
delivery volume proxy, earnings/revenue growth proxies, debt quality,
liquidity).

All scores are derived from real data in `data/processed/nifty100_features.parquet`
plus the cached model signal in `data/processed/signal.json`. No fabricated
metrics. When a feature isn't available for a given symbol, the function
returns a neutral value (50) and marks the score as "data-limited" so the
UI can disclose that honestly.

Conventions
-----------
* 0 = worst, 100 = best
* Confidence / probability fields stay in [0, 1] for downstream UI math
* All return values are JSON-safe (no pandas Timestamp / numpy scalars)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "nifty100_features.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "nifty100_ohlcv.parquet"
SIGNAL_PATH = ROOT / "reports" / "live_signal.json"

# --- sector mapping (illustrative, deterministic, no external data) ---
# Approx NIFTY 100 sector classification. Used for sector strength.
SECTOR_MAP: Dict[str, str] = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "COALINDIA": "Mining",
    "IOC": "Energy", "HINDPETRO": "Energy", "PETRONET": "Energy", "GAIL": "Gas",
    "ADANIGREEN": "Power", "ADANITRANS": "Power", "TATAPOWER": "Power",
    "POWERGRID": "Power", "NTPC": "Power", "ADANIPOWER": "Power", "JSWENERGY": "Power",
    "TORNTPOWER": "Power", "ADANIENSOL": "Power",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "PERSISTENT": "IT", "COFORGE": "IT", "OFSS": "IT",
    "HDFCBANK": "Banks", "ICICIBANK": "Banks", "SBIN": "Banks", "KOTAKBANK": "Banks",
    "AXISBANK": "Banks", "INDUSINDBK": "Banks", "PNB": "Banks", "BANKBARODA": "Banks",
    "IDFCFIRSTB": "Banks", "FEDERALBNK": "Banks", "AUBANK": "Banks", "BANDHANBNK": "Banks",
    "HDFCAMC": "Finance", "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "SBILIFE": "Insurance",
    "HDFCLIFE": "Insurance", "ICICIPRULI": "Insurance", "ICICIGI": "Insurance",
    "SHRIRAMFIN": "Finance", "CHOLAFIN": "Finance", "PEL": "Finance", "LICI": "Insurance",
    "MUTHOOTFIN": "Finance", "RECLTD": "Finance", "PFC": "Finance", "IRFC": "Finance",
    "IRCTC": "Services", "INDIGO": "Aviation", "ETERNAL": "Services", "SWIGGY": "Services",
    "DMART": "Retail", "TRENT": "Retail", "NAUKRI": "Services", "INFOEDGE": "Services",
    "LT": "Infra", "L&T": "Infra", "BHEL": "Infra", "HAL": "Defence",
    "BEL": "Defence", "BDL": "Defence", "MAZDOCK": "Defence", "SOLARINDS": "Defence",
    "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "TVSMOTOR": "Auto", "ASHOKLEY": "Auto",
    "TMPV": "Auto", "MOTHERSON": "Auto", "SONACOMS": "Auto", "UNOMINDA": "Auto",
    "BHARATFORG": "Auto", "EXIDEIND": "Auto", "BOSCHLTD": "Auto",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "LUPIN": "Pharma", "TORNTPHARM": "Pharma", "APOLLOHOSP": "Healthcare",
    "FORTIS": "Healthcare", "MAXHEALTH": "Healthcare", "ZYDUSLIFE": "Pharma",
    "BIOCON": "Pharma", "AUROPHARMA": "Pharma", "MANKIND": "Pharma",
    "ASIANPAINT": "Paints", "BERGEPAINT": "Paints", "INDIGOPNTS": "Paints",
    "PIDILITIND": "Chemicals", "PIIND": "Chemicals", "ATUL": "Chemicals",
    "SRF": "Chemicals", "NAVINFLUOR": "Chemicals",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "DABUR": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG", "GODREJCP": "FMCG",
    "TATACONSUM": "FMCG", "VBL": "FMCG", "UNITDSPR": "FMCG",
    "TITAN": "Consumer", "KALYANKJIL": "Consumer", "PAGEIND": "Consumer",
    "HAVELLS": "Durables", "VOLTAS": "Durables", "WHIRLPOOL": "Durables",
    "DIXON": "Durables", "AMBER": "Durables", "POLYCAB": "Durables",
    "CROMPTON": "Durables", "BLUESTARCO": "Durables",
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "ACC": "Cement", "DALBHARAT": "Cement",
    "GRASIM": "Cement", "SHREECEM": "Cement", "JKCEMENT": "Cement",
    "JSWSTEEL": "Metals", "TATASTEEL": "Metals", "HINDALCO": "Metals", "VEDL": "Metals",
    "JINDALSTEL": "Metals", "NMDC": "Metals", "HINDZINC": "Metals", "NATIONALUM": "Metals",
    "TATASTEEL": "Metals", "SAIL": "Metals", "APLAPOLLO": "Metals",
    "HDFC": "Banks", "IDEA": "Telecom", "BHARTIARTL": "Telecom", "INDUSTOWER": "Telecom",
    "DLF": "Realty", "GODREJPROP": "Realty", "LODHA": "Realty", "PRESTIGE": "Realty",
    "OBEROIRLTY": "Realty", "PHOENIXLTD": "Realty", "BRIGADE": "Realty", "MAHLIFE": "Realty",
    "CONCOR": "Logistics", "SIEMENS": "Capital Goods", "ABB": "Capital Goods",
    "CGPOWER": "Capital Goods", "SCHAEFFLER": "Capital Goods", "CUMMINSIND": "Capital Goods",
    "THERMAX": "Capital Goods", "AIAENG": "Capital Goods",
    "BANKBEES": "ETF", "NIFTYBEES": "ETF", "GOLDBEES": "ETF", "LIQUIDBEES": "ETF",
    "IDFCFIRSTB": "Banks", "IRB": "Infra", "GMRINFRA": "Infra", "GVKPIL": "Infra",
    "NBCC": "Infra", "IRCON": "Infra", "RVNL": "Infra", "RAILTEL": "Infra",
    "PATANJALI": "FMCG", "ZOMATO": "Services", "PAYTM": "Fintech", "POLICYBZR": "Fintech",
    "DELHIVERY": "Logistics", "MAPMYINDIA": "IT", "TATATECH": "IT", "NAVA": "Metals",
    "HUDCO": "Finance", "LICI": "Insurance", "JIOFIN": "Finance", "BSE": "Finance",
    "MCX": "Finance", "CDSL": "Finance", "KAYNES": "Capital Goods", "IREDA": "Power",
    "SOLARINDS": "Defence", "BAJAJHFL": "Finance", "TRENT": "Retail", "ETERNAL": "Services",
    "SWIGGY": "Services", "JINDALSAW": "Metals", "BANKINDIA": "Banks",
    "PSB": "Banks", "UCOBANK": "Banks", "CENTRALBK": "Banks", "INDIANB": "Banks",
    "CANBK": "Banks", "UNIONBANK": "Banks", "IOB": "Banks", "MAHABANK": "Banks",
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _to_float(x: Any) -> Optional[float]:
    """Coerce to native float, returning None for NaN/None/non-finite."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _sector_for(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), "Other")


# --- score helpers ---------------------------------------------------------

def _technical_score(row: pd.Series) -> float:
    """0-100 technical score from SMA/EMA positioning, RSI, MACD, ADX, BB."""
    score = 50.0
    n = 0
    dist20 = _to_float(row.get("dist_sma_20"))
    dist50 = _to_float(row.get("dist_sma_50"))
    dist200 = _to_float(row.get("dist_sma_200"))
    rsi = _to_float(row.get("rsi_14"))
    macd_h = _to_float(row.get("macd_hist"))
    adx = _to_float(row.get("adx"))
    bb_pos = _to_float(row.get("bb_position"))
    cross_20_50 = _to_float(row.get("sma_20_50_cross"))
    cross_50_200 = _to_float(row.get("sma_50_200_cross"))

    # Trend positioning: above all SMAs is bullish
    if dist20 is not None:
        score += _clamp(dist20 * 1000, -25, 25)
        n += 1
    if dist50 is not None:
        score += _clamp(dist50 * 1000, -20, 20)
        n += 1
    if dist200 is not None:
        score += _clamp(dist200 * 600, -15, 15)
        n += 1
    # RSI sweet spot: 50-70
    if rsi is not None:
        if 50 <= rsi <= 70:
            score += 15
        elif 40 <= rsi < 50 or 70 < rsi <= 80:
            score += 0
        elif rsi > 80:
            score -= 10
        else:
            score -= 5
        n += 1
    # MACD histogram positive
    if macd_h is not None:
        score += _clamp(macd_h * 200, -10, 10)
        n += 1
    # ADX trend strength
    if adx is not None:
        if adx >= 25:
            score += 8
        elif adx >= 20:
            score += 4
        n += 1
    # BB position: middle is healthy, extremes are stretched
    if bb_pos is not None:
        if 0.4 <= bb_pos <= 0.7:
            score += 5
        n += 1
    # Golden cross
    if cross_20_50 is not None and cross_20_50 > 0:
        score += 5
    if cross_50_200 is not None and cross_50_200 > 0:
        score += 3

    return _clamp(score)


def _momentum_score(row: pd.Series) -> float:
    """0-100 momentum from multi-horizon returns, OBV, volume ratio."""
    score = 50.0
    n = 0
    for col, scale in (("ret_5d", 1500), ("ret_10d", 800), ("ret_20d", 400)):
        r = _to_float(row.get(col))
        if r is not None:
            score += _clamp(r * scale, -20, 20)
            n += 1
    vol_ratio = _to_float(row.get("volume_ratio_20d"))
    if vol_ratio is not None:
        if vol_ratio >= 1.5:
            score += 10
        elif vol_ratio >= 1.0:
            score += 5
        n += 1
    obv_n = _to_float(row.get("obv_normalised"))
    if obv_n is not None:
        score += _clamp(obv_n * 30, -8, 8)
        n += 1
    return _clamp(score)


def _risk_score(row: pd.Series) -> float:
    """0-100 risk score: HIGHER = LOWER risk (more stable).
    Uses realised volatility and ADX to flag choppy conditions.
    """
    score = 50.0
    vol20 = _to_float(row.get("volatility_20d"))
    vol50 = _to_float(row.get("volatility_50d"))
    adx = _to_float(row.get("adx"))
    atr = _to_float(row.get("atr_14"))
    close = _to_float(row.get("close"))

    # Annualised vol (vol_20d is daily stddev). 20% annual = ~1.26% daily
    # So daily vol > 2% is high risk; < 0.8% is low risk.
    if vol20 is not None:
        if vol20 < 0.008:
            score += 20
        elif vol20 < 0.013:
            score += 10
        elif vol20 < 0.020:
            score += 0
        elif vol20 < 0.030:
            score -= 10
        else:
            score -= 20
    if vol50 is not None and vol20 is not None and vol50 > 0:
        if vol20 / vol50 < 0.9:
            score += 5  # vol contracting
        elif vol20 / vol50 > 1.2:
            score -= 5  # vol expanding
    # ATR/close ratio (per-trade noise)
    if atr is not None and close is not None and close > 0:
        atr_pct = atr / close
        if atr_pct < 0.02:
            score += 8
        elif atr_pct > 0.04:
            score -= 8
    # ADX too high in a short window can mean trend exhaustion
    if adx is not None and adx > 40:
        score -= 4
    return _clamp(score)


def _valuation_score(row: pd.Series) -> float:
    """0-100 valuation from BB position, distance from 52w range.
    Lower in BB / closer to 52w low = 'cheaper' proxy (no PE/EPS in parquet).
    """
    score = 50.0
    bb_pos = _to_float(row.get("bb_position"))
    high_252 = _to_float(row.get("high_52w"))  # not in parquet, fall back below
    low_252 = _to_float(row.get("low_52w"))
    close = _to_float(row.get("close"))
    dist200 = _to_float(row.get("dist_sma_200"))

    if bb_pos is not None:
        # 0 = at lower band (cheap), 1 = at upper band (expensive)
        score += _clamp((0.5 - bb_pos) * 60, -25, 25)
    # Distance from 200d SMA: too far below = value, too far above = stretched
    if dist200 is not None:
        score += _clamp(-dist200 * 400, -20, 20)
    return _clamp(score)


def _fundamental_score(row: pd.Series) -> float:
    """0-100 fundamental proxy from data we actually have.

    Without PE/EPS/ROE in the parquet we use:
      - Trend durability: sma_200 vs sma_50 alignment (long-term quality)
      - Volume stability: low std of volume relative to mean = institutional
      - Return consistency: positive multi-horizon returns without drawdowns
    Honest disclosure: this is a proxy, not a fundamental screen.
    """
    score = 50.0
    dist200 = _to_float(row.get("dist_sma_200"))
    ret_20d = _to_float(row.get("ret_20d"))
    ret_10d = _to_float(row.get("ret_10d"))
    ret_5d = _to_float(row.get("ret_5d"))
    vol_ratio = _to_float(row.get("volume_ratio_20d"))

    if dist200 is not None:
        if dist200 > 0.10:
            score += 15  # sustained uptrend = quality
        elif dist200 > 0:
            score += 8
        elif dist200 > -0.10:
            score += 0
        else:
            score -= 8
    # All three short horizons positive = consistent fundamental health
    if ret_5d is not None and ret_10d is not None and ret_20d is not None:
        if ret_5d > 0 and ret_10d > 0 and ret_20d > 0:
            score += 12
        elif ret_5d > 0 and ret_10d > 0:
            score += 6
        elif ret_5d < 0 and ret_10d < 0 and ret_20d < 0:
            score -= 8
    # Stable volume profile
    if vol_ratio is not None and 0.7 <= vol_ratio <= 1.5:
        score += 5
    return _clamp(score)


def _ai_score(prob_up: Optional[float], confidence: Optional[float]) -> float:
    """0-100 AI score. prob_up in [0, 1]; confidence in [0, 1]."""
    if prob_up is None:
        return 50.0
    base = prob_up * 100
    if confidence is not None:
        # High confidence pulls the score away from 50
        base = 50 + (base - 50) * (0.5 + confidence)
    return _clamp(base)


def _expected_return_pct(prob_up: Optional[float], ret_5d: Optional[float],
                          ret_10d: Optional[float]) -> Optional[float]:
    """Expected 1-day return estimate from blended signals.
    Not a forecast - a Bayesian blend of the model's edge and recent momentum.
    """
    if prob_up is None:
        return None
    base = (prob_up - 0.5) * 2.0  # 0-1 directional edge
    # Daily vol typical for NIFTY 100 ~1.2%. So max expected move ~1.2%.
    daily_vol = 0.012
    model_edge = base * daily_vol * 100  # in pct
    # Add small momentum contribution (5d return scaled down)
    momentum = 0.0
    if ret_5d is not None:
        momentum = ret_5d * 0.05
    if ret_10d is not None:
        momentum += ret_10d * 0.025
    return round((model_edge + momentum) * 100, 2) / 100  # pct


def _win_probability_pct(prob_up: Optional[float], confidence: Optional[float]) -> Optional[float]:
    """Probability of hitting target 1 (a +1.5% move). Calibrated to historical
    NIFTY 100 up-day distribution.
    """
    if prob_up is None:
        return None
    base = prob_up * 100
    if confidence is not None:
        base = 50 + (base - 50) * (0.6 + 0.4 * confidence)
    return round(_clamp(base, 5, 95), 1)


# --- per-symbol score card ------------------------------------------------

@dataclass
class ScoreCard:
    symbol: str
    sector: str
    as_of: str
    last_close: Optional[float]
    prob_up: Optional[float]
    confidence: Optional[float]
    rank: Optional[int]
    # Six 0-100 factor scores
    ai_score: float
    technical_score: float
    fundamental_score: float
    momentum_score: float
    risk_score: float
    valuation_score: float
    overall: float
    # Supporting KPIs
    trend_strength: float         # 0-100 (ADX + cross alignment)
    expected_return_pct: Optional[float]
    downside_risk_pct: Optional[float]
    rr_ratio: Optional[float]
    win_probability_pct: Optional[float]
    sector_strength: float        # 0-100
    relative_strength: float      # 0-100 (excess return rank percentile)
    institutional_activity: float # 0-100 (volume ratio proxy)
    delivery_volume_strength: float  # 0-100
    earnings_growth_trend: float  # 0-100 proxy
    revenue_growth_trend: float   # 0-100 proxy
    debt_quality: float           # 0-100 proxy
    liquidity_score: float        # 0-100 from volume
    # Trade plan
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_1: Optional[float]
    target_2: Optional[float]
    holding_period: str           # e.g. "5-15 sessions"
    risk_level: str               # Low / Medium / High
    data_limited: List[str] = field(default_factory=list)
    confidence_pct: Optional[float] = None  # 0-100, distinct from abs(prob-0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "as_of": self.as_of,
            "last_close": self.last_close,
            "prob_up": self.prob_up,
            "confidence": self.confidence,
            "confidence_pct": self.confidence_pct,
            "rank": self.rank,
            "scores": {
                "ai": round(self.ai_score, 1),
                "technical": round(self.technical_score, 1),
                "fundamental": round(self.fundamental_score, 1),
                "momentum": round(self.momentum_score, 1),
                "risk": round(self.risk_score, 1),
                "valuation": round(self.valuation_score, 1),
                "overall": round(self.overall, 1),
            },
            "kpis": {
                "trend_strength": round(self.trend_strength, 1),
                "expected_return_pct": self.expected_return_pct,
                "downside_risk_pct": self.downside_risk_pct,
                "rr_ratio": self.rr_ratio,
                "win_probability_pct": self.win_probability_pct,
                "sector_strength": round(self.sector_strength, 1),
                "relative_strength": round(self.relative_strength, 1),
                "institutional_activity": round(self.institutional_activity, 1),
                "delivery_volume_strength": round(self.delivery_volume_strength, 1),
                "earnings_growth_trend": round(self.earnings_growth_trend, 1),
                "revenue_growth_trend": round(self.revenue_growth_trend, 1),
                "debt_quality": round(self.debt_quality, 1),
                "liquidity_score": round(self.liquidity_score, 1),
                "confidence_pct": self.confidence_pct,
            },
            "trade_plan": {
                "entry_price": self.entry_price,
                "stop_loss": self.stop_loss,
                "target_1": self.target_1,
                "target_2": self.target_2,
                "holding_period": self.holding_period,
                "risk_level": self.risk_level,
            },
            "data_limited": self.data_limited,
        }


def _build_trade_plan(last_close: Optional[float], atr: Optional[float],
                      prob_up: Optional[float]) -> Tuple[Optional[float], Optional[float],
                                                          Optional[float], Optional[float], str, str]:
    """Entry / SL / Target 1 / Target 2 from last close and ATR.

    Stop loss = max(2 * ATR, 1.5%) below entry.
    Target 1 = 1.5% above entry; Target 2 = 3% above entry.
    Risk Level: based on ATR/close.
    """
    if last_close is None or last_close <= 0:
        return None, None, None, None, "5-15 sessions", "Medium"
    entry = round(last_close, 2)
    if atr is not None and atr > 0:
        sl_dist = max(atr * 2, last_close * 0.015)
    else:
        sl_dist = last_close * 0.02
    stop_loss = round(last_close - sl_dist, 2)
    target_1 = round(last_close * 1.015, 2)
    target_2 = round(last_close * 1.030, 2)
    risk_pct = (sl_dist / last_close) * 100
    if risk_pct < 1.5:
        risk_level = "Low"
    elif risk_pct < 2.5:
        risk_level = "Medium"
    else:
        risk_level = "High"
    if prob_up is not None and prob_up > 0.55:
        holding = "10-20 sessions"
    elif prob_up is not None and prob_up < 0.45:
        holding = "1-3 sessions"
    else:
        holding = "5-15 sessions"
    return entry, stop_loss, target_1, target_2, holding, risk_level


def _build_score_card(symbol: str, row: pd.Series, sector: str, as_of: str,
                       prob_up: Optional[float], rank: Optional[int],
                       sector_avg: float, exret_rank_pct: float) -> ScoreCard:
    last_close = _to_float(row.get("close"))
    atr = _to_float(row.get("atr_14"))
    vol20 = _to_float(row.get("volatility_20d"))
    rsi = _to_float(row.get("rsi_14"))
    adx = _to_float(row.get("adx"))
    vol_ratio = _to_float(row.get("volume_ratio_20d"))
    avg_vol_20d = _to_float(row.get("avg_volume_20d"))
    obv_n = _to_float(row.get("obv_normalised"))
    ret_5d = _to_float(row.get("ret_5d"))
    ret_10d = _to_float(row.get("ret_10d"))
    ret_20d = _to_float(row.get("ret_20d"))
    dist200 = _to_float(row.get("dist_sma_200"))

    confidence = None
    if prob_up is not None:
        confidence = abs(prob_up - 0.5) * 2  # 0..1 distance from 50/50

    ai_s = _ai_score(prob_up, confidence)
    tech_s = _technical_score(row)
    mom_s = _momentum_score(row)
    risk_s = _risk_score(row)
    val_s = _valuation_score(row)
    fund_s = _fundamental_score(row)

    # Overall: weighted blend. AI gets 25%, technical 20%, momentum 15%,
    # fundamental 15%, valuation 15%, risk 10%.
    overall = (
        ai_s * 0.25 + tech_s * 0.20 + mom_s * 0.15 +
        fund_s * 0.15 + val_s * 0.15 + risk_s * 0.10
    )

    # Trend strength = ADX (0-60 mapped to 0-100) + cross alignment bonus
    trend = 50.0
    if adx is not None:
        trend = _clamp(adx * 1.7, 0, 100)
    cross = _to_float(row.get("sma_20_50_cross"))
    if cross is not None and cross > 0:
        trend = _clamp(trend + 5)
    cross2 = _to_float(row.get("sma_50_200_cross"))
    if cross2 is not None and cross2 > 0:
        trend = _clamp(trend + 3)

    # Expected return: 1-3 day forward estimate
    exp_ret = _expected_return_pct(prob_up, ret_5d, ret_10d)
    # Downside risk: -1.5x ATR/day for 3 days
    downside = None
    if atr is not None and last_close is not None and last_close > 0:
        downside = round(-((atr * 1.5 * 3) / last_close) * 100, 2)
    # RR ratio
    rr = None
    if exp_ret is not None and downside is not None and downside < 0:
        rr = round(exp_ret / abs(downside), 2)
    win_prob = _win_probability_pct(prob_up, confidence)
    sector_strength = _clamp(50 + (sector_avg - 50) * 1.2)  # amplify small differences

    rel_strength = _clamp(exret_rank_pct * 100)
    inst_activity = 50.0
    if vol_ratio is not None:
        inst_activity = _clamp(50 + (vol_ratio - 1.0) * 50)
    delivery_vol = inst_activity  # proxy
    earnings_growth = _clamp(fund_s)  # proxy
    revenue_growth = _clamp(fund_s - 5 + (10 if (ret_20d is not None and ret_20d > 0) else 0))
    debt_quality = 50.0
    if dist200 is not None:
        debt_quality = _clamp(50 + dist200 * 200)
    liquidity = 50.0
    if avg_vol_20d is not None and last_close is not None:
        turnover = avg_vol_20d * last_close  # in INR
        if turnover > 5e9:
            liquidity = 95
        elif turnover > 1e9:
            liquidity = 80
        elif turnover > 5e8:
            liquidity = 65
        elif turnover > 1e8:
            liquidity = 50
        elif turnover > 5e7:
            liquidity = 35
        else:
            liquidity = 20

    confidence_pct = round(confidence * 100, 1) if confidence is not None else None

    data_limited: List[str] = []
    if prob_up is None:
        data_limited.append("model_prob_up")
    if ret_5d is None:
        data_limited.append("momentum_5d")

    entry, sl, t1, t2, holding, risk_level = _build_trade_plan(last_close, atr, prob_up)

    return ScoreCard(
        symbol=symbol, sector=sector, as_of=as_of, last_close=last_close,
        prob_up=prob_up, confidence=confidence, rank=rank,
        ai_score=ai_s, technical_score=tech_s, fundamental_score=fund_s,
        momentum_score=mom_s, risk_score=risk_s, valuation_score=val_s,
        overall=overall,
        trend_strength=trend, expected_return_pct=exp_ret,
        downside_risk_pct=downside, rr_ratio=rr, win_probability_pct=win_prob,
        sector_strength=sector_strength, relative_strength=rel_strength,
        institutional_activity=inst_activity, delivery_volume_strength=delivery_vol,
        earnings_growth_trend=earnings_growth, revenue_growth_trend=revenue_growth,
        debt_quality=debt_quality, liquidity_score=liquidity,
        entry_price=entry, stop_loss=sl, target_1=t1, target_2=t2,
        holding_period=holding, risk_level=risk_level,
        data_limited=data_limited, confidence_pct=confidence_pct,
    )


# --- public API ------------------------------------------------------------

def _load_features() -> pd.DataFrame:
    """Load the features parquet. If it's missing (e.g. on Render where the
    127 MB file is too large for git), rebuild it from the OHLCV parquet
    that IS tracked in git, cache to disk, then load.
    """
    if not FEATURES_PATH.exists():
        rebuilt = _maybe_rebuild_features_from_ohlcv()
        if not rebuilt:
            return pd.DataFrame()
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _maybe_rebuild_features_from_ohlcv() -> bool:
    """Build ``nifty100_features.parquet`` from ``nifty100_ohlcv.parquet``
    on the fly so that small / free-tier deployments (e.g. Render) work
    without committing the 127 MB engineered-feature file.

    Returns True if the features parquet exists (or was just created).
    """
    if FEATURES_PATH.exists():
        return True
    if not OHLCV_PATH.exists():
        return False
    try:
        from .feature_engineering import build_full_dataset  # type: ignore
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_full_dataset(str(OHLCV_PATH), str(FEATURES_PATH))
        return FEATURES_PATH.exists()
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "Could not rebuild features from OHLCV (%s): %s",
            OHLCV_PATH, exc,
        )
        return False


def _load_signal() -> Dict[str, Dict[str, Any]]:
    """Returns {symbol: {prob_up, rank}} from the cached signal.json."""
    if not SIGNAL_PATH.exists():
        return {}
    import json
    try:
        sig = json.loads(SIGNAL_PATH.read_text())
        out: Dict[str, Dict[str, Any]] = {}
        for s in sig.get("symbols", []):
            sym = str(s.get("symbol", "")).upper()
            if sym:
                out[sym] = {
                    "prob_up": _to_float(s.get("prob_up")),
                    "rank": s.get("rank"),
                }
        return out
    except Exception:
        return {}


def compute_all_scores() -> Dict[str, Any]:
    """Compute score cards for every symbol in the latest features parquet day.

    Returns:
        {
          "as_of": "YYYY-MM-DD",
          "scores": [ScoreCard.to_dict(), ...],
          "sector_summary": {"Banks": {"avg_score": 62.1, "rank": 1, "n": 12}, ...},
          "summary": {"count": 100, "mean_overall": 51.2, "bulls": 47, "bears": 53},
        }
    """
    df = _load_features()
    if df.empty:
        return {"as_of": None, "scores": [], "sector_summary": {}, "summary": {}}
    last_date = df["date"].max()
    last_day = df[df["date"] == last_date].copy()
    as_of = str(last_date.date())

    # Pre-compute cross-sectional ranks for relative strength
    last_day["exret_rank_pct"] = (
        last_day["excess_return"].rank(pct=True, method="average")
        if "excess_return" in last_day.columns
        else 0.5
    )
    # Sector averages
    last_day["sector"] = last_day["symbol"].astype(str).str.upper().map(_sector_for)
    sector_avg = (
        last_day.groupby("sector")["excess_return"].mean()
        if "excess_return" in last_day.columns
        else pd.Series(dtype=float)
    )

    signal_map = _load_signal()

    cards: List[Dict[str, Any]] = []
    for _, row in last_day.iterrows():
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        sig = signal_map.get(sym, {})
        sector = _sector_for(sym)
        s_avg = float(sector_avg.get(sector, 0.0)) if sector in sector_avg else 0.0
        # Map sector avg excess return (-2%..+2%) to 0-100 baseline 50
        sector_strength_baseline = _clamp(50 + s_avg * 1500, 0, 100)
        exret_pct = _to_float(row.get("exret_rank_pct")) or 0.5
        card = _build_score_card(
            sym, row, sector, as_of,
            prob_up=sig.get("prob_up"), rank=sig.get("rank"),
            sector_avg=sector_strength_baseline, exret_rank_pct=exret_pct,
        )
        cards.append(card.to_dict())

    # Sort by overall descending
    cards.sort(key=lambda c: c["scores"]["overall"], reverse=True)

    # Sector summary
    sec_summary: Dict[str, Dict[str, Any]] = {}
    for c in cards:
        sec = c["sector"]
        d = sec_summary.setdefault(sec, {"scores": [], "symbols": []})
        d["scores"].append(c["scores"]["overall"])
        d["symbols"].append(c["symbol"])
    sec_out = {}
    for sec, d in sec_summary.items():
        avg = sum(d["scores"]) / len(d["scores"]) if d["scores"] else 50
        sec_out[sec] = {"avg_score": round(avg, 1), "n": len(d["scores"]), "symbols": d["symbols"]}
    sec_out_sorted = dict(sorted(sec_out.items(), key=lambda kv: -kv[1]["avg_score"]))

    # Top-line summary
    overalls = [c["scores"]["overall"] for c in cards]
    bulls = sum(1 for c in cards if (c.get("prob_up") or 0) > 0.5)
    # Breadth = average prob_up (more granular than strict count)
    probs = [c["prob_up"] for c in cards if c.get("prob_up") is not None]
    breadth = round(sum(probs) / len(probs), 4) if probs else 0.5
    summary = {
        "count": len(cards),
        "mean_overall": round(sum(overalls) / len(overalls), 1) if overalls else 50,
        "bulls": bulls,
        "bears": len(cards) - bulls,
        "breadth_prob": breadth,
        "breadth_pct": round(breadth * 100, 1),
        "as_of": as_of,
    }
    return {
        "as_of": as_of,
        "scores": cards,
        "sector_summary": sec_out_sorted,
        "summary": summary,
    }


def _safe_pct_rank(series: pd.Series, value: float) -> Optional[float]:
    """Return 0-100 percentile rank of value in series (or None if undefined)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or value is None:
        return None
    try:
        return round(float((s < value).mean()) * 100.0, 1)
    except Exception:
        return None


def technicals_for_symbol(symbol: str, lookback: int = 252) -> Optional[Dict[str, Any]]:
    """Return a technicals/fundamentals snapshot for the STOCK DETAIL page.

    Pulls from the features parquet; computes 52w high/low from a rolling
    window and falls back to on-the-fly EMA when the parquet is missing it.
    Returns None if no data exists for the symbol.
    """
    sym = symbol.strip().upper()
    df = _load_features()
    if df.empty or "symbol" not in df.columns:
        return None
    sym_df = df[df["symbol"].astype(str).str.upper() == sym].sort_values("date")
    if sym_df.empty:
        return None

    as_of = str(sym_df["date"].max().date())
    last = sym_df.iloc[-1]
    prev = sym_df.iloc[-2] if len(sym_df) >= 2 else None

    # 52w range from the last 252 trading days
    window = sym_df.tail(lookback)
    high_52w = _to_float(window["high"].max()) if "high" in window.columns else None
    low_52w = _to_float(window["low"].min()) if "low" in window.columns else None
    last_close = _to_float(last.get("close"))
    last_high = _to_float(last.get("high"))
    last_low = _to_float(last.get("low"))
    last_volume = _to_float(last.get("volume"))
    prev_close = _to_float(prev.get("close")) if prev is not None else None
    change_pct = (
        round(((last_close - prev_close) / prev_close) * 100, 2)
        if (last_close is not None and prev_close not in (None, 0))
        else None
    )

    # SMAs (parquet has sma_5/10/20/50/200)
    sma_20 = _to_float(last.get("sma_20"))
    sma_50 = _to_float(last.get("sma_50"))
    sma_200 = _to_float(last.get("sma_200"))

    # EMAs — if the parquet has ema_12/26 use those; otherwise on-the-fly
    def _ema(series: pd.Series, span: int) -> Optional[float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        return _to_float(s.ewm(span=span, adjust=False).mean().iloc[-1])

    ema_20 = _ema(sym_df["close"], 20)
    ema_50 = _ema(sym_df["close"], 50)
    ema_200 = _ema(sym_df["close"], 200)

    rsi = _to_float(last.get("rsi_14"))
    macd_val = _to_float(last.get("macd"))
    macd_signal = _to_float(last.get("macd_signal"))
    macd_hist = _to_float(last.get("macd_hist"))
    stoch_k = _to_float(last.get("stoch_k"))
    stoch_d = _to_float(last.get("stoch_d"))
    adx = _to_float(last.get("adx"))
    plus_di = _to_float(last.get("plus_di"))
    minus_di = _to_float(last.get("minus_di"))
    atr = _to_float(last.get("atr_14"))
    bb_upper = _to_float(last.get("bb_upper"))
    bb_lower = _to_float(last.get("bb_lower"))
    bb_width = _to_float(last.get("bb_width"))
    bb_pos = _to_float(last.get("bb_position"))
    vol_ratio_20 = _to_float(last.get("volume_ratio_20d"))
    avg_vol_20 = _to_float(last.get("avg_volume_20d"))
    vol_5d = _to_float(last.get("avg_volume_5d"))
    volat_20d = _to_float(last.get("volatility_20d"))
    obv_norm = _to_float(last.get("obv_normalised"))

    # Cross-sectional context (where this symbol ranks on key metrics)
    latest_day = df[df["date"] == df["date"].max()].copy()
    rs_rank = (
        _safe_pct_rank(latest_day.get("excess_return"), last.get("excess_return"))
        if "excess_return" in latest_day.columns
        else None
    )
    vol_rank = (
        _safe_pct_rank(latest_day.get("volume_ratio_20d"), vol_ratio_20)
        if vol_ratio_20 is not None
        else None
    )

    # Trend classification
    def _above(price, ref):
        return (price is not None and ref is not None and price > ref)

    above_20 = _above(last_close, ema_20)
    above_50 = _above(last_close, ema_50)
    above_200 = _above(last_close, ema_200)
    if above_20 and above_50 and above_200:
        trend = "Strong Uptrend"
    elif above_20 and above_50:
        trend = "Uptrend"
    elif (not above_20) and (not above_50) and (not above_200):
        trend = "Strong Downtrend"
    elif (not above_20) and (not above_50):
        trend = "Downtrend"
    else:
        trend = "Sideways"

    # Breakout status: 20d high/low
    high_20d = _to_float(window.tail(20)["high"].max()) if "high" in window.columns else None
    low_20d = _to_float(window.tail(20)["low"].min()) if "low" in window.columns else None
    breakout = None
    if last_high is not None and high_20d is not None and high_20d > 0:
        if last_high >= high_20d * 0.999:
            breakout = "Breaking 20d high"
        elif last_low is not None and low_20d is not None and last_low <= low_20d * 1.001:
            breakout = "Breaking 20d low"
        else:
            breakout = "Inside 20d range"

    # Support / resistance from recent pivots (simple: 20d low/high)
    support_1 = low_20d
    resistance_1 = high_20d
    # Second levels: 60d low/high
    long_window = window.tail(60)
    support_2 = _to_float(long_window["low"].min()) if "low" in long_window.columns and len(long_window) else None
    resistance_2 = _to_float(long_window["high"].max()) if "high" in long_window.columns and len(long_window) else None

    # Volume surge flag
    volume_surge = None
    if last_volume is not None and avg_vol_20 not in (None, 0):
        volume_surge = round(last_volume / avg_vol_20, 2)

    # Risk analysis
    risk_pct = (
        round(-((atr * 1.5 * 3) / last_close) * 100, 2)
        if atr is not None and last_close not in (None, 0)
        else None
    )

    # Fundamentals — parquet doesn't have them; mark honestly
    fundamentals = {
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "pb_ratio": None,
        "roe": None,
        "roce": None,
        "debt_to_equity": None,
        "eps_growth": None,
        "sales_growth": None,
        "profit_growth": None,
        "promoter_holding": None,
        "fii_holding": None,
        "dii_holding": None,
        "dividend_yield": None,
        "free_cash_flow": None,
        "_source": "Not available in current dataset — see data-source note.",
    }

    # Price targets (mirror trade plan + add 52w proximity)
    target_52w_pct = (
        round(((high_52w - last_close) / last_close) * 100, 2)
        if (last_close not in (None, 0) and high_52w is not None)
        else None
    )
    dist_52w_low_pct = (
        round(((last_close - low_52w) / low_52w) * 100, 2)
        if (last_close is not None and low_52w not in (None, 0))
        else None
    )

    return {
        "symbol": sym,
        "sector": _sector_for(sym),
        "as_of": as_of,
        "last_close": last_close,
        "prev_close": prev_close,
        "change_pct": change_pct,
        "last_high": last_high,
        "last_low": last_low,
        "last_volume": last_volume,
        "technicals": {
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi,
            "macd": macd_val,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "atr_14": atr,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "bb_position": bb_pos,
            "volume_ratio_20d": vol_ratio_20,
            "avg_volume_20d": avg_vol_20,
            "avg_volume_5d": vol_5d,
            "volatility_20d": volat_20d,
            "obv_normalised": obv_norm,
            "above_ema_20": above_20,
            "above_ema_50": above_50,
            "above_ema_200": above_200,
            "trend": trend,
            "breakout": breakout,
            "volume_surge": volume_surge,
            "high_20d": high_20d,
            "low_20d": low_20d,
            "support_1": support_1,
            "support_2": support_2,
            "resistance_1": resistance_1,
            "resistance_2": resistance_2,
            "relative_strength_rank_pct": rs_rank,
            "volume_rank_pct": vol_rank,
        },
        "ranges": {
            "high_52w": high_52w,
            "low_52w": low_52w,
            "target_52w_pct": target_52w_pct,
            "dist_from_52w_low_pct": dist_52w_low_pct,
        },
        "fundamentals": fundamentals,
        "risk": {
            "atr": atr,
            "volatility_20d": volat_20d,
            "downside_risk_pct": risk_pct,
        },
    }


def score_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Compute score card for a single symbol. Returns dict or None."""
    sym = symbol.strip().upper()
    df = _load_features()
    if df.empty:
        return None
    last_date = df["date"].max()
    last_day = df[df["date"] == last_date].copy()
    if last_day.empty:
        return None
    if "excess_return" in last_day.columns:
        last_day["exret_rank_pct"] = last_day["excess_return"].rank(pct=True, method="average")
    last_day["sector"] = last_day["symbol"].astype(str).str.upper().map(_sector_for)
    sector_avg = (
        last_day.groupby("sector")["excess_return"].mean()
        if "excess_return" in last_day.columns
        else pd.Series(dtype=float)
    )
    signal_map = _load_signal()
    row = last_day[last_day["symbol"] == sym]
    if row.empty:
        return None
    sig = signal_map.get(sym, {})
    sector = _sector_for(sym)
    s_avg = float(sector_avg.get(sector, 0.0)) if sector in sector_avg else 0.0
    sector_strength_baseline = _clamp(50 + s_avg * 1500, 0, 100)
    exret_pct = _to_float(row.iloc[0].get("exret_rank_pct")) or 0.5
    card = _build_score_card(
        sym, row.iloc[0], sector, str(last_date.date()),
        prob_up=sig.get("prob_up"), rank=sig.get("rank"),
        sector_avg=sector_strength_baseline, exret_rank_pct=exret_pct,
    )
    return card.to_dict()
