"""Unit tests for scripts/refresh_live_features.py (no network / no API)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import refresh_live_features as rlf  # noqa: E402
from market_ml.feature_engineering import (  # noqa: E402
    build_cross_sectional, build_symbol_features)


# ---------------------------------------------------------------------------
# aggregate_intraday_to_daily
# ---------------------------------------------------------------------------

def _write_intraday_csv(path: Path, symbol: str, date_str: str,
                        opens, highs, lows, closes, volumes) -> None:
    """Write a synthetic 5-min candle file in the same shape as Groww dumps."""
    rows = []
    for i, (o, h, l, c, v) in enumerate(
            zip(opens, highs, lows, closes, volumes)):
        rows.append({
            "timestamp": f"{date_str}T{9 + i // 60:02d}:{i % 60:02d}:00+05:30",
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_aggregate_empty_live_dir(tmp_path):
    """No live files => empty DataFrame with the expected schema."""
    out = rlf.aggregate_intraday_to_daily(tmp_path)
    assert out.empty
    assert list(out.columns) == ["date", "symbol", "open", "high", "low",
                                 "close", "volume", "source"]


def test_aggregate_two_symbols_three_days(tmp_path):
    """Two symbols × three days of 5-min candles should collapse to 6 rows
    with open=first, high=max, low=min, close=last, volume=sum."""
    for date in ("2026-08-25", "2026-08-26", "2026-08-27"):
        _write_intraday_csv(
            tmp_path / f"MAXHEALTH_{date}.csv", "MAXHEALTH", date,
            opens=[100, 101, 102], highs=[103, 104, 105],
            lows=[99, 100, 101], closes=[102, 103, 104],
            volumes=[10, 20, 30],
        )
        _write_intraday_csv(
            tmp_path / f"IDEA_{date}.csv", "IDEA", date,
            opens=[50, 51, 52], highs=[53, 54, 55],
            lows=[49, 50, 51], closes=[52, 53, 54],
            volumes=[100, 200, 300],
        )
    out = rlf.aggregate_intraday_to_daily(tmp_path)
    assert len(out) == 6
    assert set(out["symbol"].unique()) == {"MAXHEALTH", "IDEA"}
    assert list(out["date"].dt.strftime("%Y-%m-%d").unique()) == [
        "2026-08-25", "2026-08-26", "2026-08-27"]

    # MAXHEALTH 2026-08-25: open=100 (first), high=105 (max), low=99 (min),
    # close=104 (last), vol=60 (sum)
    row = out[(out["symbol"] == "MAXHEALTH") &
              (out["date"] == pd.Timestamp("2026-08-25"))].iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 105.0
    assert row["low"] == 99.0
    assert row["close"] == 104.0
    assert row["volume"] == 60.0
    assert row["source"] == "groww_live"


def test_aggregate_skips_unparseable_files(tmp_path):
    """A file with a non-numeric symbol/date stem should be ignored."""
    (tmp_path / "BADFILE.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-08-25T09:15:00+05:30,1,2,1,2,10\n", encoding="utf-8")
    _write_intraday_csv(
        tmp_path / "TCS_2026-08-25.csv", "TCS", "2026-08-25",
        opens=[200], highs=[201], lows=[199], closes=[201], volumes=[10])
    out = rlf.aggregate_intraday_to_daily(tmp_path)
    assert list(out["symbol"].unique()) == ["TCS"]


# ---------------------------------------------------------------------------
# refresh() — end-to-end with a minimal historical dataset
# ---------------------------------------------------------------------------

def _make_historical_ohlcv(tmp_path: Path, symbols: list[str],
                           n_days: int = 300) -> Path:
    """Build a tiny but feature-warm historical OHLCV (SMA-200 needs ~252d)."""
    rng = np.random.default_rng(42)
    rows = []
    base_dates = pd.bdate_range(end="2026-08-24", periods=n_days)
    for sym in symbols:
        px = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
        for d, p in zip(base_dates, px):
            rows.append({
                "date": pd.Timestamp(d).normalize(),
                "symbol": sym,
                "open": float(p),
                "high": float(p + 1),
                "low": float(p - 1),
                "close": float(p),
                "volume": 1000.0,
                "source": "historical",
            })
    path = tmp_path / "nifty100_ohlcv.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _make_historical_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Build a feature dataset that mirrors the production schema, derived
    from the synthetic historical OHLCV. Used to seed the features parquet
    so refresh_live_features has a schema + warmup to project against."""
    frames = []
    for sym, grp in hist.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        feats = build_symbol_features(grp)
        out = pd.concat([grp.reset_index(drop=True),
                         feats.reset_index(drop=True)], axis=1)
        frames.append(out)
    full = pd.concat(frames, ignore_index=True)
    cs = build_cross_sectional(full)
    if not cs.empty:
        full = pd.concat([full, cs.reset_index(drop=True)], axis=1)
    if "target_direction" not in full.columns:
        full["target_direction"] = np.nan
    return full


def test_refresh_appends_new_rows_and_is_idempotent(tmp_path):
    """First run appends 6 rows, second run adds 0 (idempotent)."""
    ohlcv_path = _make_historical_ohlcv(tmp_path, ["MAXHEALTH", "IDEA"])
    hist = pd.read_parquet(ohlcv_path)
    seed = _make_historical_features(hist)
    features_path = tmp_path / "nifty100_features.parquet"
    seed.to_parquet(features_path, index=False)
    live_dir = tmp_path / "live" / "nse" / "ohlcv"
    live_dir.mkdir(parents=True)

    for date in ("2026-08-25", "2026-08-26", "2026-08-27"):
        _write_intraday_csv(
            live_dir / f"MAXHEALTH_{date}.csv", "MAXHEALTH", date,
            opens=[105.0, 106.0, 107.0], highs=[108.0, 109.0, 110.0],
            lows=[104.0, 105.0, 106.0], closes=[107.0, 108.0, 109.0],
            volumes=[100, 200, 300],
        )
        _write_intraday_csv(
            live_dir / f"IDEA_{date}.csv", "IDEA", date,
            opens=[55.0, 56.0, 57.0], highs=[58.0, 59.0, 60.0],
            lows=[54.0, 55.0, 56.0], closes=[57.0, 58.0, 59.0],
            volumes=[1000, 2000, 3000],
        )

    summary1 = rlf.refresh(ohlcv_path, features_path, live_dir)
    assert summary1["refreshed_rows"] == 6
    assert summary1["live_dates"] == ["2026-08-25", "2026-08-26",
                                       "2026-08-27"]
    assert summary1["symbols"] == 2

    df1 = pd.read_parquet(features_path)
    assert df1["date"].max() == pd.Timestamp("2026-08-27")
    assert df1["date"].isna().sum() == 0

    # Second call should be a no-op (idempotent).
    summary2 = rlf.refresh(ohlcv_path, features_path, live_dir)
    assert summary2["refreshed_rows"] == 0
    assert summary2["live_dates"] == []
    df2 = pd.read_parquet(features_path)
    assert len(df2) == len(df1)
    assert df2["date"].max() == pd.Timestamp("2026-08-27")


def test_refresh_no_live_files_is_noop(tmp_path):
    """With an empty live dir, refresh should return refreshed_rows=0 and
    leave the features parquet untouched (or create it empty if missing)."""
    ohlcv_path = _make_historical_ohlcv(tmp_path, ["TCS"])
    hist = pd.read_parquet(ohlcv_path)
    seed = _make_historical_features(hist)
    features_path = tmp_path / "nifty100_features.parquet"
    seed.to_parquet(features_path, index=False)
    live_dir = tmp_path / "live" / "nse" / "ohlcv"
    live_dir.mkdir(parents=True)

    summary = rlf.refresh(ohlcv_path, features_path, live_dir)
    assert summary["refreshed_rows"] == 0


def test_refresh_targets_post_features_max_dates(tmp_path):
    """Live dates <= features parquet max date should NOT be appended (the
    cutoff is the features max, not the OHLCV max)."""
    ohlcv_path = _make_historical_ohlcv(tmp_path, ["RELIANCE"])
    hist = pd.read_parquet(ohlcv_path)
    seed = _make_historical_features(hist)
    features_path = tmp_path / "nifty100_features.parquet"
    seed.to_parquet(features_path, index=False)
    live_dir = tmp_path / "live" / "nse" / "ohlcv"
    live_dir.mkdir(parents=True)

    # Use a date that's already inside the features window (e.g. 2026-08-20)
    _write_intraday_csv(
        live_dir / "RELIANCE_2026-08-20.csv", "RELIANCE", "2026-08-20",
        opens=[100], highs=[101], lows=[99], closes=[100], volumes=[10])
    summary = rlf.refresh(ohlcv_path, features_path, live_dir)
    assert summary["refreshed_rows"] == 0
    assert summary["live_dates"] == []
