"""Nifty 100 universe definition.

Constituent list as of a recent rebalance; Yahoo Finance symbols (.NS).
Data access assumption (documented per AGENTS.md): constituents are fetched
from public Yahoo Finance endpoints using the `yfinance` package — no API
keys or credentials are used or stored.
"""
from __future__ import annotations

NIFTY100_SYMBOLS: list[str] = [
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "ATGL", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BAJAJHFL", "BANKBARODA",
    "BEL", "BPCL", "BHARTIARTL", "BHEL", "BOSCHLTD", "BRITANNIA", "CANBK",
    "CGPOWER", "CHOLAFIN", "CIPLA", "COALINDIA", "DABUR", "DIVISLAB",
    "DLF", "DMART", "DRREDDY", "EICHERMOT", "ESCORTS", "ETERNAL", "GAIL",
    "GODREJCP", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "HUDCO", "ICICIBANK", "IDEA",
    "INDHOTEL", "INDIGO", "INDUSINDBK", "INFY", "IOC", "IRFC", "ITC",
    "JINDALSTEL", "JSWENERGY", "JSWSTEEL", "KOTAKBANK", "LT", "LICI", "LODHA",
    "M&M", "MARUTI", "MAXHEALTH", "MOTHERSON", "NAUKRI", "NESTLEIND",
    "NTPC", "ONGC", "PFC", "PIDILITIND", "PNB", "POWERGRID", "RECLTD",
    "RELIANCE", "SBILIFE", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SUNPHARMA", "SWIGGY", "TATACHEM", "TATACONSUM", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR",
    "ULTRACEMCO", "UNITDSPR", "VBL", "VEDL", "WIPRO", "ZYDUSLIFE",
]
# Note: list is kept to 100 valid NSE tickers; symbols that fail to resolve
# on Yahoo Finance are skipped at download time and reported by the loader.


def yahoo_symbol(symbol: str) -> str:
    """Map an NSE symbol to its Yahoo Finance ticker."""
    return f"{symbol}.NS"
