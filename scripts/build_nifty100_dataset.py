"""Build a real historical OHLCV dataset for the Nifty 100 universe.

Usage:
    python scripts/build_nifty100_dataset.py --start 2016-01-01 --end 2026-08-25
        [--symbols SYM1,SYM2] [--output data/processed/nifty100_ohlcv]
        [--force] [--provider jugaad-data|yfinance] [--max-symbols N]

Downloads each symbol separately (resume-safe: existing per-symbol interim
files are kept unless --force), applies quality checks, and combines the
normalized files into <output>.parquet and <output>.csv. Also writes a
data-quality report next to the dataset.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.jugaad_loader import download_nse_stock, load_config, load_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

REQUEST_DELAY_SECONDS = 1.0  # be polite to NSE endpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: config)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol subset")
    parser.add_argument(
        "--output",
        default="data/processed/nifty100_ohlcv",
        help="Output path prefix (without extension)",
    )
    parser.add_argument("--force", action="store_true", help="Redownload existing symbols")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["jugaad-data", "yfinance"],
        help="Primary provider override",
    )
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit number of symbols")
    return parser.parse_args()


def check_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply quality checks; return cleaned frame + report dict."""
    report: dict = {}
    n0 = len(df)

    bad_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    report["invalid_dates_removed"] = int(bad_dates)

    price_cols = ["open", "high", "low", "close"]
    nonpositive = df[price_cols].le(0).any(axis=1) | df[price_cols].isna().any(axis=1)
    report["nonpositive_or_missing_prices_removed"] = int(nonpositive.sum())
    df = df[~nonpositive]

    bad_high_low = df["high"] < df["low"]
    report["high_lt_low_removed"] = int(bad_high_low.sum())
    df = df[~bad_high_low]

    # high should bound open/close where applicable
    bad_bounds = (df["high"] < df[["open", "close"]].max(axis=1))
    report["high_lt_open_or_close_removed"] = int(bad_bounds.sum())
    df = df[~bad_bounds]

    neg_volume = df["volume"] < 0
    report["negative_volume_removed"] = int(neg_volume.sum())
    df = df[~neg_volume]

    dupes = df.duplicated(subset=["symbol", "date"], keep="last")
    report["duplicate_rows_removed"] = int(dupes.sum())
    df = df[~dupes]

    report["rows_before"] = n0
    report["rows_after"] = len(df)
    return df.reset_index(drop=True), report


def missing_dates_report(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Count expected-but-missing trading dates per symbol vs NSE calendar proxy."""
    rows = []
    for sym, g in df.groupby("symbol"):
        idx = pd.DatetimeIndex(g["date"])
        full = pd.bdate_range(start, end)
        missing = len(full) - idx.normalize().nunique()
        rows.append({"symbol": sym, "rows": len(g), "approx_missing_bdays": max(missing, 0),
                     "first_date": idx.min().date(), "last_date": idx.max().date()})
    return pd.DataFrame(rows).sort_values("approx_missing_bdays", ascending=False)


def main() -> int:
    args = parse_args()
    cfg = load_config(ROOT / "config" / "data.yaml")

    start = args.start or cfg.get("date_range", {}).get("start", "2016-01-01")
    end = args.end or cfg.get("date_range", {}).get("end") or pd.Timestamp.today().strftime("%Y-%m-%d")

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols(cfg)
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    provider_cfg = cfg.get("provider", {})
    fallback = str(provider_cfg.get("fallback_provider", "none")).lower()
    retries = int(provider_cfg.get("retries", 3))
    retry_delay = float(provider_cfg.get("retry_delay_seconds", 2.0))

    raw_dir = ROOT / "data" / "raw" / "nse"
    interim_dir = ROOT / "data" / "interim" / "nse"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    succeeded, failed, skipped = [], [], []
    for i, sym in enumerate(symbols, 1):
        interim_path = interim_dir / f"{sym}.csv"
        if interim_path.exists() and interim_path.stat().st_size > 0 and not args.force:
            logger.info("[%d/%d] %s: cached, skipping (use --force to refresh)", i, len(symbols), sym)
            skipped.append(sym)
            frames.append(pd.read_csv(interim_path))
            continue
        try:
            norm = download_nse_stock(
                sym,
                start,
                end,
                retries=retries,
                retry_delay=retry_delay,
                fallback_provider=fallback if args.provider is None else ("yfinance" if args.provider == "yfinance" else "none"),
                raw_dir=raw_dir,
                interim_dir=interim_dir,
            )
            frames.append(norm)
            succeeded.append(sym)
            logger.info(
                "[%d/%d] %s: OK rows=%d range=%s..%s source=%s",
                i, len(symbols), sym, len(norm),
                norm["date"].min().date(), norm["date"].max().date(),
                norm["source"].iloc[0],
            )
        except Exception as exc:  # noqa: BLE001 - log failure, continue universe
            failed.append(sym)
            logger.error("[%d/%d] %s: FAILED (%s)", i, len(symbols), sym, exc)
        time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        logger.error("No data downloaded; aborting.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined, qreport = check_quality(combined)

    out_prefix = ROOT / args.output
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_prefix.with_suffix(".parquet"), index=False)
    combined.to_csv(out_prefix.with_suffix(".csv"), index=False)

    coverage = missing_dates_report(combined, start, end)
    min_rows = 250
    insufficient = coverage[coverage["rows"] < min_rows]

    report_lines = [
        "# Nifty 100 dataset build report",
        "",
        f"- Date range requested: {start} .. {end}",
        f"- Symbols requested: {len(symbols)}",
        f"- Downloaded OK: {len(succeeded)}",
        f"- Loaded from cache: {len(skipped)}",
        f"- Failed: {len(failed)} {failed if failed else ''}",
        f"- Combined rows: {len(combined)}",
        f"- Quality: {qreport}",
        "",
        "## Symbols with insufficient history (<%d rows)" % min_rows,
        "",
        insufficient.to_string(index=False) if not insufficient.empty else "(none)",
        "",
        "## Coverage by symbol (top 30 by approx missing business days)",
        "",
        coverage.head(30).to_string(index=False),
    ]
    report_path = out_prefix.parent / (out_prefix.stem + "_quality_report.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {out_prefix.with_suffix('.parquet')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {report_path}")
    if failed:
        print(f"Failed symbols ({len(failed)}): {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
