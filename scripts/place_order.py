#!/usr/bin/env python
"""Script to place orders using Groww API.

This script demonstrates how to place buy/sell orders using the GrowwLiveLoader.
Based on the sample script provided by Groww.

Usage:
    python scripts/place_order.py --symbol RELIANCE --quantity 1 --side BUY
    python scripts/place_order.py --symbol IDEA --quantity 10 --side SELL --order-type LIMIT --price 15.50
"""

import argparse
import logging
import os
import sys
import time
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


def main():
    parser = argparse.ArgumentParser(description="Place orders via Groww API")
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading symbol (e.g., RELIANCE, IDEA, TCS)"
    )
    parser.add_argument(
        "--quantity", "-q",
        type=int,
        required=True,
        help="Number of shares to buy/sell"
    )
    parser.add_argument(
        "--side", "-S",
        choices=["BUY", "SELL"],
        required=True,
        help="Transaction side: BUY or SELL"
    )
    parser.add_argument(
        "--order-type", "-t",
        choices=["MARKET", "LIMIT"],
        default="MARKET",
        help="Order type: MARKET or LIMIT (default: MARKET)"
    )
    parser.add_argument(
        "--price", "-p",
        type=float,
        default=0,
        help="Limit price (required for LIMIT orders)"
    )
    parser.add_argument(
        "--validity", "-v",
        choices=["DAY", "IOC"],
        default="DAY",
        help="Order validity: DAY or IOC (default: DAY)"
    )
    parser.add_argument(
        "--exchange", "-e",
        choices=["NSE", "BSE"],
        default="NSE",
        help="Exchange: NSE or BSE (default: NSE)"
    )
    parser.add_argument(
        "--segment", "-g",
        choices=["CASH", "FNO"],
        default="CASH",
        help="Segment: CASH or FNO (default: CASH)"
    )
    parser.add_argument(
        "--product", "-P",
        choices=["MIS", "CNC", "NRML"],
        default="MIS",
        help="Product type: MIS, CNC, or NRML (default: MIS)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/data.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Simulate order without actually placing it"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Validate LIMIT order requires price
    if args.order_type == "LIMIT" and args.price <= 0:
        parser.error("Limit price (--price) is required for LIMIT orders")
    
    # Create loader
    loader = create_groww_loader(args.config)
    
    # Initialize
    if not loader.initialize():
        logger.error("Failed to initialize Groww API. Check your credentials.")
        sys.exit(1)
    
    # Check funds before placing order
    if args.side == "BUY":
        funds = loader.get_funds()
        if funds:
            logger.info(f"Available funds: {funds}")
    
    if args.dry_run:
        logger.info("DRY RUN - Order details:")
        logger.info(f"  Symbol: {args.symbol}")
        logger.info(f"  Quantity: {args.quantity}")
        logger.info(f"  Side: {args.side}")
        logger.info(f"  Order Type: {args.order_type}")
        logger.info(f"  Price: {args.price if args.order_type == 'LIMIT' else 'Market'}")
        logger.info(f"  Validity: {args.validity}")
        logger.info(f"  Exchange: {args.exchange}")
        logger.info(f"  Segment: {args.segment}")
        logger.info(f"  Product: {args.product}")
        print("\n✅ Dry run complete. No order was placed.")
        return
    
    # Place order
    logger.info(f"Placing {args.order_type} {args.side} order for {args.symbol}...")
    
    order_result = loader.place_order(
        trading_symbol=args.symbol,
        quantity=args.quantity,
        transaction_type=args.side,
        order_type=args.order_type,
        price=args.price,
        validity=args.validity,
        exchange=args.exchange,
        segment=args.segment,
        product=args.product
    )
    
    if order_result:
        order_id = order_result.get('groww_order_id', order_result)
        logger.info(f"✅ Order placed successfully!")
        logger.info(f"   Order ID: {order_id}")
        logger.info(f"   Symbol: {args.symbol}")
        logger.info(f"   Quantity: {args.quantity}")
        logger.info(f"   Side: {args.side}")
        logger.info(f"   Type: {args.order_type}")
        if args.order_type == "LIMIT":
            logger.info(f"   Price: {args.price}")
    else:
        logger.error("❌ Failed to place order")
        sys.exit(1)
    
    # Optionally check order book
    logger.info("\nFetching order book...")
    orders = loader.get_order_book()
    if orders:
        logger.info(f"Current orders: {len(orders)}")
        for order in orders[-5:]:  # Show last 5 orders
            logger.info(f"  {order}")


if __name__ == "__main__":
    main()