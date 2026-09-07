"""Build leakage-safe feature dataset from frozen Nifty 100 OHLCV.

Usage:
    python scripts/build_features.py
        [--input data/processed/nifty100_ohlcv.parquet]
        [--output data/processed/nifty100_features]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "nifty100_ohlcv.parquet"),
        help="Input frozen OHLCV parquet path",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "nifty100_features"),
        help="Output path prefix (without extension)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Building features from %s", args.input)

    from market_ml.feature_engineering import build_full_dataset, ensure_dataframe

    result = build_full_dataset(args.input, args.output)
    features = ensure_dataframe(result, context='build_features')

    logger.info("Feature build complete:")
    logger.info("  Rows: %d", result["row_count"])
    logger.info("  Symbols: %d", result["symbol_count"])
    logger.info("  Features: %d", result["feature_count"])
    logger.info("  Target balance: %s", result["target_balance"])
    logger.info("  DataFrame shape: %s", features.shape)

    if not result["removal_log"].empty:
        logger.info("  Removed rows by symbol:")
        for _, row in result["removal_log"].iterrows():
            logger.info("    %s: %s (%d rows)", row["symbol"], row["reason"], row["rows_removed"])

    # Write quality report
    report_path = ROOT / "reports" / "feature_quality_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Feature Quality Report",
        "",
        f"**Dataset version:** nifty100_features_v1_2026-08-26",
        f"**Source:** nifty100_ohlcv_v1_2026-08-26",
        f"**Generated:** 2026-08-26",
        "",
        "## Summary",
        "",
        f"- Total rows: {result['row_count']:,}",
        f"- Symbols: {result['symbol_count']}",
        f"- Feature columns: {result['feature_count']}",
        f"- Target: `target_direction` (1=up, 0=down)",
        "",
        "## Target Balance",
        "",
        f"- Valid rows: {result['target_balance']['total']:,}",
        f"- Up (1): {result['target_balance']['up']:,} ({result['target_balance']['pct_up']}%)",
        f"- Down (0): {result['target_balance']['down']:,} ({100 - result['target_balance']['pct_up']}%)",
        f"- NaN (last row per symbol): {result['row_count'] - result['target_balance']['total']:,}",
        "",
        "## Missing Values",
        "",
    ]

    if result["missing_summary"]:
        lines.append("| Feature | Missing Count |")
        lines.append("|---------|--------------|")
        for feat, cnt in sorted(result["missing_summary"].items(), key=lambda x: -x[1]):
            lines.append(f"| {feat} | {cnt:,} |")
    else:
        lines.append("(no missing values in non-target columns)")

    lines.extend([
        "",
        "## Removed Rows by Symbol",
        "",
    ])

    if not result["removal_log"].empty:
        lines.append("| Symbol | Reason | Rows Removed |")
        lines.append("|--------|--------|-------------|")
        for _, row in result["removal_log"].sort_values("rows_removed", ascending=False).iterrows():
            lines.append(f"| {row['symbol']} | {row['reason']} | {row['rows_removed']} |")
    else:
        lines.append("(no rows removed)")

    lines.extend([
        "",
        "## Per-Symbol Statistics",
        "",
        "| Symbol | Rows | First Date | Last Date | Target NaNs |",
        "|--------|------|-----------|-----------|-------------|",
    ])
    for _, row in result["symbol_stats"].sort_values("rows", ascending=False).iterrows():
        lines.append(
            f"| {row['symbol']} | {row['rows']} | "
            f"{row['first_date'].date()} | {row['last_date'].date()} | "
            f"{row['target_nulls']} |"
        )

    lines.extend([
        "",
        "## Leakage Prevention",
        "",
        "- All features computed per-symbol using only past/present data",
        "- No future OHLC prices used in feature computation",
        "- Cross-sectional features use only same-date data",
        "- Target uses `shift(-1)` within each symbol only",
        "- Last row per symbol has NaN target (excluded from training)",
        "",
        "## Assumptions",
        "",
        "- Stochastic oscillator uses 14-period lookback (standard)",
        "- ADX uses 14-period (Wilder's standard)",
        "- OBV normalised by 20-day rolling std for cross-symbol comparability",
        "- MACD/signal/histogram normalised by price level for cross-symbol comparability",
        "- SMA 200 requires 200+ trading days of history; newly listed stocks "
        "will have NaN for early rows",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote quality report to %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
