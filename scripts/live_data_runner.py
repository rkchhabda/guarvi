#!/usr/bin/env python
"""Continuous live-data fetcher for NSE market hours (systemd-friendly).

Runs as a daemon on a VPS whose static IP is whitelisted for the Groww API.
While the NSE cash market is open (Mon-Fri, 09:15-15:30 IST by default) it
polls live quotes and OHLCV candles at a fixed interval and appends them to
daily CSV files under data/live/nse/. Outside market hours it sleeps until
the next session. The Groww access token is renewed automatically each day.

Note: NSE trading holidays are not special-cased; on a holiday the API
simply returns the previous session's data and nothing harmful is saved.

Usage:
    python scripts/live_data_runner.py --universe nifty100 --candle-interval 5minute
    python scripts/live_data_runner.py --symbols RELIANCE TCS --poll-seconds 60
    python scripts/live_data_runner.py --once      # one fetch cycle, ignore market hours
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.groww_loader import create_groww_loader  # noqa: E402

log = logging.getLogger("live_data_runner")


# ---------------------------------------------------------------------------
# Market-hours helpers
# ---------------------------------------------------------------------------

def parse_hhmm(value: str) -> tuple[int, int]:
    """Parse 'HH:MM' into an (hour, minute) tuple."""
    try:
        h_str, m_str = value.split(":")
        hour, minute = int(h_str), int(m_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected HH:MM, got {value!r}") from exc
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise argparse.ArgumentTypeError(f"Time out of range: {value!r}")
    return hour, minute


def next_session(
    now: datetime, start_hm: tuple[int, int], end_hm: tuple[int, int]
) -> tuple[datetime, datetime]:
    """Return (open, close) of the next NSE session at or after `now`.

    Sessions run Mon-Fri (weekends skipped). If `now` is mid-session the
    current session is returned; if markets have closed for the day the next
    weekday's session is returned.
    """
    day = now
    for _ in range(8):  # covers any weekend
        start = day.replace(hour=start_hm[0], minute=start_hm[1], second=0, microsecond=0)
        end = day.replace(hour=end_hm[0], minute=end_hm[1], second=0, microsecond=0)
        if day.weekday() < 5 and now <= end:
            return start, end
        day = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    raise RuntimeError("Could not find the next market session within 8 days.")


# ---------------------------------------------------------------------------
# Symbols + persistence
# ---------------------------------------------------------------------------

def load_symbols(
    config_path: str, symbols: list[str] | None, universe: str | None
) -> list[str]:
    """Resolve the symbol list from CLI args or the configured universe."""
    if symbols:
        return [s.strip().upper() for s in symbols if s.strip()]
    if universe == "nifty100":
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        symbols_file = Path(config.get("symbols_file", "config/nifty100_symbols.csv"))
        if not symbols_file.exists():
            raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
        df = pd.read_csv(symbols_file)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        out = df[col].dropna().astype(str).str.strip().tolist()
        return [s for s in out if s]
    raise ValueError("Provide --symbols or --universe nifty100")


def append_csv(df_new: pd.DataFrame, path: Path, dedupe_cols: list[str]) -> int:
    """Append rows to a CSV file, deduplicating on `dedupe_cols` (keep newest).

    If the existing file is unreadable or incompatible (e.g. missing the
    dedupe columns), it is recreated from `df_new` instead of crashing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df_new
    if path.exists():
        try:
            df_old = pd.read_csv(path)
            missing = [c for c in dedupe_cols if c not in df_old.columns]
            if missing:
                raise ValueError(f"existing file missing dedupe columns {missing}")
            merged = pd.concat([df_old, df_new], ignore_index=True)
            merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
            df = merged.sort_values(dedupe_cols)
        except Exception as exc:  # unreadable/incompatible file: start over
            log.warning("Could not merge into %s (%s); recreating it.", path, exc)
            df = df_new
    df.to_csv(path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Fetch cycle
# ---------------------------------------------------------------------------

def fetch_cycle(loader, symbols: list[str], live_dir: Path, args) -> None:
    """Fetch one snapshot of quotes + candles and append them to daily CSVs."""
    today = datetime.now(ZoneInfo(args.timezone)).strftime("%Y%m%d")

    if not args.no_quotes:
        quotes = loader.get_multiple_quotes(symbols)
        if quotes:
            quote_df = pd.DataFrame([{**q, "symbol": sym} for sym, q in quotes.items()])
            preferred = ["symbol", "timestamp", "last_price", "open", "high", "low",
                         "close", "volume", "avg_price", "oi", "change", "change_percent"]
            ordered = [c for c in preferred if c in quote_df.columns]
            quote_df = quote_df[ordered + [c for c in quote_df.columns if c not in ordered]]
            name = f"quotes_{today}.csv"
            n = append_csv(quote_df, live_dir / name, ["symbol", "timestamp"])
            log.info("Saved %d quotes (%d rows in %s)", len(quotes), n, name)

    if not args.no_candles:
        for sym in symbols:
            ohlcv = loader.get_live_ohlcv(sym, interval=args.candle_interval)
            if ohlcv is None or len(ohlcv) == 0:
                log.warning("%s: no candle data returned", sym)
                continue
            ts_col = "timestamp" if "timestamp" in ohlcv.columns else ohlcv.columns[0]
            rel = Path("ohlcv") / f"{sym}_{today}.csv"
            n = append_csv(ohlcv, live_dir / rel, [ts_col])
            log.info("%s: %d candle rows on disk (%s)", sym, n, args.candle_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NSE market-hours live data fetcher (Groww API)"
    )
    parser.add_argument("--config", "-c", default="config/data.yaml",
                        help="Path to data configuration YAML")
    parser.add_argument("--symbols", "-s", nargs="+",
                        help="Symbols to fetch (overrides --universe)")
    parser.add_argument("--universe", "-u", choices=["nifty100"], default="nifty100",
                        help="Predefined universe to fetch (default: nifty100)")
    parser.add_argument("--candle-interval", default="5minute",
                        choices=["1minute", "5minute", "15minute", "30minute",
                                 "60minute", "day"],
                        help="Candle interval for OHLCV (default: 5minute)")
    parser.add_argument("--poll-seconds", type=int, default=300,
                        help="Seconds between fetch cycles (default: 300)")
    parser.add_argument("--start", type=parse_hhmm, default=parse_hhmm("09:15"),
                        help="Session start HH:MM in --timezone (default: 09:15)")
    parser.add_argument("--end", type=parse_hhmm, default=parse_hhmm("15:30"),
                        help="Session end HH:MM in --timezone (default: 15:30)")
    parser.add_argument("--timezone", default="Asia/Kolkata",
                        help="IANA timezone for market hours (default: Asia/Kolkata)")
    parser.add_argument("--no-quotes", action="store_true", help="Skip live quotes")
    parser.add_argument("--no-candles", action="store_true", help="Skip OHLCV candles")
    parser.add_argument("--once", action="store_true",
                        help="Run a single fetch cycle now, ignoring market hours")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def run(args) -> int:
    tz = ZoneInfo(args.timezone)
    symbols = load_symbols(args.config, args.symbols, None if args.symbols else args.universe)
    preview = ", ".join(symbols[:5]) + (" ..." if len(symbols) > 5 else "")
    log.info("Tracking %d symbols: %s", len(symbols), preview)

    loader = None
    loader_day: str | None = None

    def ensure_loader() -> bool:
        """(Re)create the loader once per day so the access token stays fresh."""
        nonlocal loader, loader_day
        today = datetime.now(tz).strftime("%Y-%m-%d")
        if loader is not None and loader_day == today:
            return True
        loader = create_groww_loader(args.config)
        if not loader.initialize():
            loader = None
            return False
        loader_day = today
        return True

    if args.once:
        if not ensure_loader():
            log.error("Groww API init failed. Check GROWW_API_KEY / GROWW_SECRET "
                      "in .env and that this machine's IP is whitelisted.")
            return 1
        fetch_cycle(loader, symbols, loader.live_dir, args)
        log.info("Single cycle complete (--once).")
        return 0

    log.info("Market hours %02d:%02d-%02d:%02d %s, polling every %ds",
             *args.start, *args.end, args.timezone, args.poll_seconds)

    while True:
        now = datetime.now(tz)
        sess_start, sess_end = next_session(now, args.start, args.end)
        if now < sess_start:
            wait_min = (sess_start - now).total_seconds() / 60
            log.info("Market closed; sleeping %.0f min until %s", wait_min, sess_start)
            time.sleep(max(1.0, (sess_start - now).total_seconds()))
            continue

        if not ensure_loader():
            log.error("Groww API init failed; retrying in 60s "
                      "(check credentials / IP whitelist).")
            time.sleep(60)
            continue

        try:
            fetch_cycle(loader, symbols, loader.live_dir, args)
        except Exception:
            log.exception("Fetch cycle failed; continuing.")

        now = datetime.now(tz)
        wake = now + timedelta(seconds=args.poll_seconds)
        if wake >= sess_end:
            log.info("Session ended at %s", sess_end)
            continue  # loop recomputes the next session
        time.sleep((wake - now).total_seconds())


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
