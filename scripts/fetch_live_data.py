#!/usr/bin/env python
"""Script to fetch live market data using Groww API.

This script demonstrates how to use the GrowwLiveLoader to fetch
real-time market data for symbols in your universe.

Usage:
    python scripts/fetch_live_data.py --symbols RELIANCE TCS INFY
    python scripts/fetch_live_data.py --universe nifty100 --interval 5minute
    python scripts/fetch_live_data.py --symbols RELIANCE --quote-only
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.groww_loader import create_groww_loader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_symbols_from_config(config_path: str = "config/data.yaml") -> list[str]:
    """Load symbols from the data configuration."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    
    symbols_file = Path(config.get("symbols_file", "config/nifty100_symbols.csv"))
    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
    
    import pandas as pd
    df = pd.read_csv(symbols_file)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    symbols = df[col].dropna().astype(str).str.strip().tolist()
    return [s for s in symbols if s]


def main():
    parser = argparse.ArgumentParser(description="Fetch live market data from Groww API")
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Trading symbols to fetch (e.g., RELIANCE TCS INFY)"
    )
    parser.add_argument(
        "--universe", "-u",
        choices=["nifty100"],
        help="Predefined universe to fetch"
    )
    parser.add_argument(
        "--interval", "-i",
        default="5minute",
        choices=["1minute", "5minute", "15minute", "30minute", "60minute", "day"],
        help="Candle interval for OHLCV data"
    )
    parser.add_argument(
        "--quote-only", "-q",
        action="store_true",
        help="Fetch only live quotes (no OHLCV)"
    )
    parser.add_argument(
        "--ohlcv-only", "-o",
        action="store_true",
        help="Fetch only OHLCV data (no quotes)"
    )
    parser.add_argument(
        "--save", "-S",
        action="store_true",
        help="Save data to CSV files"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/data.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Determine symbols to fetch
    if args.symbols:
        symbols = args.symbols
    elif args.universe == "nifty100":
        symbols = load_symbols_from_config(args.config)
        logger.info(f"Loaded {len(symbols)} symbols from NIFTY100 universe")
    else:
        parser.error("Either --symbols or --universe must be specified")
    
    # Create loader
    loader = create_groww_loader(args.config)
    
    # Initialize
    if not loader.initialize():
        logger.error("Failed to initialize Groww API. Check your credentials.")
        sys.exit(1)
    
    logger.info(f"Fetching live data for {len(symbols)} symbols...")
    
    # Fetch quotes
    if not args.ohlcv_only:
        logger.info("Fetching live quotes...")
        quotes = loader.get_multiple_quotes(symbols)
        logger.info(f"Successfully fetched quotes for {len(quotes)} symbols")
        
        for symbol, quote in quotes.items():
            print(f"\n{symbol}:")
            print(f"  Last Price: {quote['last_price']}")
            print(f"  Change: {quote['change']} ({quote['change_percent']:.2f}%)")
            print(f"  Volume: {quote['volume']}")
            print(f"  Open: {quote['open']}, High: {quote['high']}, Low: {quote['low']}, Close: {quote['close']}")
            
            if args.save:
                loader.save_live_data(quote, symbol, "quote")
    
    # Fetch OHLCV
    if not args.quote_only:
        logger.info(f"Fetching live OHLCV data (interval: {args.interval})...")
        for symbol in symbols:
            ohlcv = loader.get_live_ohlcv(symbol, interval=args.interval)
            if ohlcv is not None and len(ohlcv) > 0:
                logger.info(f"{symbol}: {len(ohlcv)} candles fetched")
                print(f"\n{symbol} OHLCV (last 5 candles):")
                print(ohlcv.tail().to_string(index=False))
                
                if args.save:
                    loader.save_live_data(ohlcv, symbol, "ohlcv")
            else:
                logger.warning(f"{symbol}: No OHLCV data available")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()