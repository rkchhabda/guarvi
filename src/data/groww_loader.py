"""Groww API live data loader for real-time market data.

This module provides functionality to fetch live market data from Groww API.
Requires:
- Groww API key and secret (from Groww developer portal)
- Static IP whitelisted in Groww developer portal
- Access token generated from API key + secret

Usage:
    from src.data.groww_loader import GrowwLiveLoader
    
    loader = GrowwLiveLoader()
    live_data = loader.get_live_quote("RELIANCE")
    live_ohlcv = loader.get_live_ohlcv("RELIANCE", interval="1minute")
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    """Load a project-root .env file so GROWW_* variables are available.

    Uses python-dotenv when installed; silently skips otherwise (values can
    still be provided via real environment variables).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # project root
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            return


_load_project_env()

# Canonical schema for live data
LIVE_QUOTE_COLUMNS = [
    "symbol", "timestamp", "last_price", "open", "high", "low", "close",
    "volume", "avg_price", "oi", "change", "change_percent"
]

LIVE_OHLCV_COLUMNS = [
    "symbol", "timestamp", "open", "high", "low", "close", "volume"
]

# Endpoints used to detect this machine's outbound (public) IP for Groww's
# static-IP whitelisting. Tried in order until one responds.
PUBLIC_IP_SERVICES = [
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://checkip.amazonaws.com",
    "https://ipinfo.io/ip",
]


class GrowwLiveLoader:
    """Live data loader using Groww API."""
    
    def __init__(self, config_path: str | Path = "config/data.yaml"):
        """Initialize the Groww live data loader.
        
        Args:
            config_path: Path to the data configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.groww_config = self.config.get("groww", {})
        self.api_key = self.groww_config.get("api_key") or os.getenv("GROWW_API_KEY", "")
        self.secret = self.groww_config.get("secret") or os.getenv("GROWW_SECRET", "")
        self.static_ip = self.groww_config.get("static_ip") or os.getenv("GROWW_STATIC_IP", "")
        self.access_token = self.groww_config.get("access_token") or os.getenv("GROWW_ACCESS_TOKEN", "")
        self.live_dir = Path(self.config.get("paths", {}).get("live_dir", "data/live/nse"))
        self.live_dir.mkdir(parents=True, exist_ok=True)
        
        self._groww_api = None
        self._initialized = False
    
    def _load_config(self, config_path: str | Path) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    
    def get_public_ip(self, timeout: float = 5.0) -> str | None:
        """Detect this machine's outbound (public) IP address.
        
        Tries PUBLIC_IP_SERVICES in order and returns the first successful
        response. Uses only the standard library, so it works even before
        any credentials are configured. Honors HTTPS_PROXY if set (useful
        when routing Groww traffic through a static-IP proxy/VPN).
        
        Args:
            timeout: Per-request timeout in seconds.
            
        Returns:
            Public IP string, or None if all services fail.
        """
        last_error: Exception | None = None
        for url in PUBLIC_IP_SERVICES:
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    ip = resp.read().decode("utf-8").strip()
                if ip:
                    return ip
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            logger.debug(f"All public-IP services failed; last error: {last_error}")
        return None
    
    def verify_static_ip(self) -> bool | None:
        """Check that the outbound public IP matches the configured static IP.
        
        Groww whitelists the *source IP* of incoming API requests, so the IP
        configured in GROWW_STATIC_IP (and whitelisted in the Groww developer
        portal) must equal this machine's current public IP.
        
        Returns:
            True if IPs match, False on mismatch, None if the check was
            skipped (no static IP configured or detection failed).
        """
        if not self.static_ip:
            return None
        public_ip = self.get_public_ip()
        if not public_ip:
            logger.warning("Could not detect public IP; skipping static-IP verification.")
            return None
        if public_ip.strip() != self.static_ip.strip():
            logger.warning(
                "⚠️ Public IP %s does NOT match configured GROWW_STATIC_IP %s. "
                "Groww API calls may be rejected. Whitelist %s in the Groww "
                "developer portal, or run: python scripts/check_static_ip.py --set",
                public_ip, self.static_ip, public_ip,
            )
            return False
        logger.info("✅ Public IP %s matches whitelisted GROWW_STATIC_IP", public_ip)
        return True
    
    def initialize(self) -> bool:
        """Initialize Groww API connection.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True
            
        if not self.api_key or not self.secret:
            logger.error("Groww API key and secret are required. Set GROWW_API_KEY and GROWW_SECRET environment variables.")
            return False
        
        try:
            from growwapi import GrowwAPI
            
            # Generate access token if not provided
            if not self.access_token:
                logger.info("Generating access token from API key and secret...")
                self.access_token = GrowwAPI.get_access_token(
                    api_key=self.api_key,
                    secret=self.secret
                )
                logger.info("Access token generated successfully")
            
            # Initialize Groww API
            self._groww_api = GrowwAPI(self.access_token)
            self._initialized = True
            logger.info("✅ Groww API initialized successfully")
            
            # Best-effort pre-flight: warn if outbound IP is not the whitelisted one.
            self.verify_static_ip()
            return True
            
        except ImportError:
            logger.error("growwapi package not installed. Run: pip install growwapi")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Groww API: {e}")
            return False
    
    def get_live_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get live quote for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "TCS")
            
        Returns:
            Dictionary with live quote data or None if failed.
        """
        if not self.initialize():
            return None
        
        try:
            # Groww API uses get_quote for live quotes
            quote = self._groww_api.get_quote(trading_symbol=symbol)
            
            if quote:
                # Normalize to our schema
                normalized = {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp.now(),
                    "last_price": quote.get("last_price", 0),
                    "open": quote.get("open", 0),
                    "high": quote.get("high", 0),
                    "low": quote.get("low", 0),
                    "close": quote.get("close", 0),
                    "volume": quote.get("volume", 0),
                    "avg_price": quote.get("average_price", 0),
                    "oi": quote.get("oi", 0),
                    "change": quote.get("change", 0),
                    "change_percent": quote.get("change_percent", 0),
                }
                return normalized
            return None
            
        except Exception as e:
            logger.error(f"Failed to get live quote for {symbol}: {e}")
            return None
    
    def get_live_ohlcv(
        self,
        symbol: str,
        interval: str = "1minute",
        from_date: str | None = None,
        to_date: str | None = None
    ) -> pd.DataFrame | None:
        """Get live OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "TCS")
            interval: Candle interval (e.g., "1minute", "5minute", "15minute", "30minute", "60minute", "day")
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data or None if failed.
        """
        if not self.initialize():
            return None
        
        try:
            # Groww API uses get_historical_data for historical candles
            # For live data, we can use recent historical data
            from_date = from_date or pd.Timestamp.now().strftime("%Y-%m-%d")
            to_date = to_date or pd.Timestamp.now().strftime("%Y-%m-%d")
            
            # Map interval to Groww API format
            interval_map = {
                "1minute": "1minute",
                "5minute": "5minute",
                "15minute": "15minute",
                "30minute": "30minute",
                "60minute": "60minute",
                "day": "day",
            }
            groww_interval = interval_map.get(interval, "1minute")
            
            data = self._groww_api.get_historical_data(
                trading_symbol=symbol,
                interval=groww_interval,
                from_date=from_date,
                to_date=to_date
            )
            
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                # Normalize column names
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Ensure required columns
                rename_map = {
                    "timestamp": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
                df = df.rename(columns=rename_map)
                
                # Add symbol column
                df["symbol"] = symbol
                
                # Convert timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Select and order columns
                available_cols = [c for c in LIVE_OHLCV_COLUMNS if c in df.columns]
                df = df[available_cols]
                
                return df.sort_values("timestamp").reset_index(drop=True)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get live OHLCV for {symbol}: {e}")
            return None
    
    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get live quotes for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbol to quote data
        """
        results = {}
        for symbol in symbols:
            quote = self.get_live_quote(symbol)
            if quote:
                results[symbol] = quote
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        return results
    
    def save_live_data(self, data: pd.DataFrame | dict, symbol: str, data_type: str = "ohlcv") -> Path | None:
        """Save live data to CSV file.
        
        Args:
            data: DataFrame or dict with live data
            symbol: Trading symbol
            data_type: Type of data ("ohlcv" or "quote")
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{data_type}_{timestamp}.csv"
            filepath = self.live_dir / filename
            
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            df.to_csv(filepath, index=False)
            logger.info(f"Saved live {data_type} data for {symbol} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save live data for {symbol}: {e}")
            return None
    
    def place_order(
        self,
        trading_symbol: str,
        quantity: int,
        transaction_type: str,  # "BUY" or "SELL"
        order_type: str = "MARKET",  # "MARKET" or "LIMIT"
        price: float = 0,
        validity: str = "DAY",
        exchange: str = "NSE",
        segment: str = "CASH",
        product: str = "MIS"
    ) -> dict | None:
        """Place an order via Groww API.
        
        Args:
            trading_symbol: Trading symbol
            quantity: Number of shares
            transaction_type: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            price: Limit price (required for LIMIT orders)
            validity: "DAY" or "IOC"
            exchange: "NSE" or "BSE"
            segment: "CASH" or "FNO"
            product: "MIS", "CNC", "NRML"
            
        Returns:
            Order response dict or None if failed
        """
        if not self.initialize():
            return None
        
        try:
            # Map to Groww API constants
            txn_type = getattr(self._groww_api, f"TRANSACTION_TYPE_{transaction_type.upper()}", transaction_type)
            ord_type = getattr(self._groww_api, f"ORDER_TYPE_{order_type.upper()}", order_type)
            val = getattr(self._groww_api, f"VALIDITY_{validity.upper()}", validity)
            exch = getattr(self._groww_api, f"EXCHANGE_{exchange.upper()}", exchange)
            seg = getattr(self._groww_api, f"SEGMENT_{segment.upper()}", segment)
            prod = getattr(self._groww_api, f"PRODUCT_{product.upper()}", product)
            
            order_id = self._groww_api.place_order(
                trading_symbol=trading_symbol,
                quantity=quantity,
                validity=val,
                exchange=exch,
                segment=seg,
                product=prod,
                order_type=ord_type,
                transaction_type=txn_type,
                price=price if order_type.upper() == "LIMIT" else 0
            )
            
            logger.info(f"Order placed for {trading_symbol}: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place order for {trading_symbol}: {e}")
            return None
    
    def get_order_book(self) -> list[dict] | None:
        """Get order book from Groww API.
        
        Returns:
            List of orders or None if failed
        """
        if not self.initialize():
            return None
        
        try:
            orders = self._groww_api.get_order_book()
            return orders
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return None
    
    def get_positions(self) -> list[dict] | None:
        """Get current positions from Groww API.
        
        Returns:
            List of positions or None if failed
        """
        if not self.initialize():
            return None
        
        try:
            positions = self._groww_api.get_positions()
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return None
    
    def get_holdings(self) -> list[dict] | None:
        """Get holdings from Groww API.
        
        Returns:
            List of holdings or None if failed
        """
        if not self.initialize():
            return None
        
        try:
            holdings = self._groww_api.get_holdings()
            return holdings
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return None
    
    def get_funds(self) -> dict | None:
        """Get available funds from Groww API.
        
        Returns:
            Funds dict or None if failed
        """
        if not self.initialize():
            return None
        
        try:
            funds = self._groww_api.get_funds()
            return funds
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return None


def create_groww_loader(config_path: str | Path = "config/data.yaml") -> GrowwLiveLoader:
    """Factory function to create a GrowwLiveLoader instance.
    
    Args:
        config_path: Path to the data configuration YAML file.
        
    Returns:
        GrowwLiveLoader instance
    """
    return GrowwLiveLoader(config_path)


if __name__ == "__main__":
    # Demo usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    loader = create_groww_loader()
    
    if not loader.initialize():
        print("Failed to initialize Groww API. Check your credentials.")
        sys.exit(1)
    
    # Test with a symbol
    symbol = "RELIANCE"
    print(f"\nFetching live quote for {symbol}...")
    quote = loader.get_live_quote(symbol)
    if quote:
        print(f"Quote: {quote}")
        loader.save_live_data(quote, symbol, "quote")
    
    print(f"\nFetching live OHLCV for {symbol}...")
    ohlcv = loader.get_live_ohlcv(symbol, interval="5minute")
    if ohlcv is not None:
        print(f"OHLCV shape: {ohlcv.shape}")
        print(ohlcv.tail())
        loader.save_live_data(ohlcv, symbol, "ohlcv")
    
    print("\nDone!")