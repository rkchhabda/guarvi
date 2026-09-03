"""Validation tests for leakage-safe feature engineering.

Tests the 9 required validations plus additional quality checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_ml.feature_engineering import (
    build_full_dataset,
    build_symbol_features,
    build_target,
    build_cross_sectional,
)

FEATURES_PARQUET = Path(__file__).resolve().parents[1] / "data" / "processed" / "nifty100_features.parquet"
OHLCV_PARQUET = Path(__file__).resolve().parents[1] / "data" / "processed" / "nifty100_ohlcv.parquet"


@pytest.fixture(scope="module")
def features_df():
    """Load the built feature dataset (module-scoped for speed)."""
    if not FEATURES_PARQUET.exists():
        pytest.skip("Feature dataset not built yet; run scripts/build_features.py")
    return pd.read_parquet(FEATURES_PARQUET)


@pytest.fixture(scope="module")
def ohlcv_df():
    """Load the frozen OHLCV dataset."""
    if not OHLCV_PARQUET.exists():
        pytest.skip("OHLCV dataset not found")
    return pd.read_parquet(OHLCV_PARQUET)


# ---------------------------------------------------------------------------
# Test 1: No duplicate date-symbol rows
# ---------------------------------------------------------------------------
class TestNoDuplicateRows:
    def test_no_duplicate_date_symbol(self, features_df):
        dupes = features_df.duplicated(subset=["date", "symbol"])
        assert not dupes.any(), f"Found {dupes.sum()} duplicate date-symbol rows"


# ---------------------------------------------------------------------------
# Test 2: Features are calculated separately by symbol
# ---------------------------------------------------------------------------
class TestPerSymbolCalculation:
    def test_per_symbol_features_differ(self, features_df):
        """Two symbols should have different feature values on the same date."""
        # Pick a date well into the dataset where features are populated
        mid_date = features_df["date"].sort_values().iloc[len(features_df) // 2]
        day_data = features_df[features_df["date"] == mid_date]
        syms = day_data["symbol"].unique()[:2]
        row0 = day_data[day_data["symbol"] == syms[0]].iloc[0]
        row1 = day_data[day_data["symbol"] == syms[1]].iloc[0]
        # At least one technical feature should differ
        tech_cols = ["rsi_14", "macd", "sma_20", "volatility_20d"]
        any_diff = any(row0[c] != row1[c] for c in tech_cols if pd.notna(row0[c]) and pd.notna(row1[c]))
        assert any_diff, "Features appear identical across symbols — cross-contamination?"

    def test_each_symbol_has_target_nan_at_end(self, features_df):
        """Last row per symbol must have NaN target."""
        last_rows = features_df.groupby("symbol").tail(1)
        assert last_rows["target_direction"].isna().all(), (
            "Some symbols do not have NaN target on their last row"
        )


# ---------------------------------------------------------------------------
# Test 3: No feature uses future rows (leakage check)
# ---------------------------------------------------------------------------
class TestNoFutureLeakage:
    def test_features_only_use_past(self, features_df):
        """For each symbol, features at date t must not depend on t+1..t+n."""
        # SMA at date t only uses closes up to t
        sym = features_df[features_df["symbol"] == "RELIANCE"].copy()
        sym = sym.sort_values("date").reset_index(drop=True)
        # Take a row in the middle
        mid = len(sym) // 2
        row = sym.iloc[mid]
        # SMA_20 at date t should equal the mean of close[t-19..t]
        closes = sym["close"].iloc[mid - 19 : mid + 1]
        expected_sma_20 = closes.mean()
        assert abs(row["sma_20"] - expected_sma_20) < 1e-6, (
            f"SMA_20 = {row['sma_20']}, expected {expected_sma_20}"
        )

    def test_target_uses_only_next_close(self, features_df):
        """target_direction at t should equal (close[t+1] > close[t])."""
        sym = features_df[features_df["symbol"] == "RELIANCE"].copy().sort_values("date")
        # Skip last row (NaN target)
        for i in range(len(sym) - 2, max(len(sym) - 10, 0), -1):
            row = sym.iloc[i]
            next_row = sym.iloc[i + 1]
            expected = 1.0 if next_row["close"] > row["close"] else 0.0
            actual = row["target_direction"]
            if pd.notna(actual):
                assert actual == expected, (
                    f"Row {i}: target={actual}, expected={expected}"
                )


# ---------------------------------------------------------------------------
# Test 4: Target equals next valid trading-session direction
# ---------------------------------------------------------------------------
class TestTargetDefinition:
    def test_target_direction_logic(self, features_df):
        """target_direction == 1 iff next session close > current close."""
        sym = features_df[features_df["symbol"] == "INFY"].copy().sort_values("date")
        valid = sym.dropna(subset=["target_direction"])
        # Check a sample of rows
        sample = valid.iloc[::100]  # every 100th row
        for idx, row in sample.iterrows():
            pos = sym.index.get_loc(idx)
            if pos < len(sym) - 1:
                next_close = sym.iloc[pos + 1]["close"]
                expected = 1.0 if next_close > row["close"] else 0.0
                assert row["target_direction"] == expected


# ---------------------------------------------------------------------------
# Test 5: Last row for each symbol has no target
# ---------------------------------------------------------------------------
class TestLastRowNoTarget:
    def test_last_row_nan_target(self, features_df):
        last = features_df.groupby("symbol").tail(1)
        assert last["target_direction"].isna().all()

    def test_second_last_has_target(self, features_df):
        second_last = features_df.groupby("symbol").nth(-2)
        assert second_last["target_direction"].notna().all()


# ---------------------------------------------------------------------------
# Test 6: No target column appears in feature list
# ---------------------------------------------------------------------------
class TestTargetNotInFeatures:
    def test_target_not_in_features(self, features_df):
        target_col = "target_direction"
        # The feature matrix should not have target_direction as a feature
        # (it's a separate column, but verify it's not accidentally used)
        meta_cols = {"date", "symbol", "open", "high", "low", "close", "volume", "source", target_col}
        feature_cols = [c for c in features_df.columns if c not in meta_cols]
        assert target_col not in feature_cols
        assert len(feature_cols) > 50, f"Expected many features, got {len(feature_cols)}"


# ---------------------------------------------------------------------------
# Test 7: Feature values are finite after cleaning
# ---------------------------------------------------------------------------
class TestFiniteValues:
    def test_no_infinite_features(self, features_df):
        meta_cols = {"date", "symbol", "open", "high", "low", "close", "volume", "source", "target_direction"}
        feature_cols = [c for c in features_df.columns if c not in meta_cols]
        inf_count = np.isinf(features_df[feature_cols].values).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in features"

    def test_no_nan_in_critical_features(self, features_df):
        """After warmup removal, key features should not be all-NaN."""
        # ret_1d should be available for all but the first row per symbol
        for sym, grp in features_df.groupby("symbol"):
            grp = grp.sort_values("date")
            # First row may be NaN for ret_1d
            # All others should be valid
            if len(grp) > 1:
                ret_1d_valid = grp["ret_1d"].iloc[1:].notna().all()
                assert ret_1d_valid, f"{sym}: ret_1d has NaN after first row"


# ---------------------------------------------------------------------------
# Test 8: Original OHLCV columns are unchanged
# ---------------------------------------------------------------------------
class TestOHLCVUnchanged:
    def test_ohlcv_columns_present(self, features_df):
        for col in ["date", "symbol", "open", "high", "low", "close", "volume", "source"]:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_ohlcv_values_match(self, features_df, ohlcv_df):
        """OHLCV values in features should match the frozen dataset."""
        merged = features_df[["date", "symbol", "open", "high", "low", "close", "volume"]].merge(
            ohlcv_df[["date", "symbol", "open", "high", "low", "close", "volume"]],
            on=["date", "symbol"],
            how="inner",
            suffixes=("_feat", "_orig"),
        )
        for col in ["open", "high", "low", "close", "volume"]:
            assert (merged[f"{col}_feat"] == merged[f"{col}_orig"]).all(), (
                f"Column {col} differs between features and OHLCV"
            )


# ---------------------------------------------------------------------------
# Test 9: Feature row dates do not exceed frozen dataset end date
# ---------------------------------------------------------------------------
class TestDateBounds:
    def test_dates_within_range(self, features_df, ohlcv_df):
        max_feat_date = features_df["date"].max()
        max_ohlcv_date = ohlcv_df["date"].max()
        assert max_feat_date <= max_ohlcv_date, (
            f"Feature max date {max_feat_date} exceeds OHLCV max date {max_ohlcv_date}"
        )

    def test_min_date_not_earlier(self, features_df, ohlcv_df):
        min_feat_date = features_df["date"].min()
        min_ohlcv_date = ohlcv_df["date"].min()
        assert min_feat_date >= min_ohlcv_date, (
            f"Feature min date {min_feat_date} is earlier than OHLCV min {min_ohlcv_date}"
        )


# ---------------------------------------------------------------------------
# Additional quality checks
# ---------------------------------------------------------------------------
class TestAdditionalQuality:
    def test_all_expected_features_exist(self, features_df):
        expected = [
            "ret_1d", "ret_5d", "ret_20d", "log_ret_1d", "log_ret_5d",
            "hl_range", "co_return", "gap_return",
            "sma_5", "sma_20", "sma_50", "sma_200",
            "dist_sma_20", "dist_ema_12",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d", "adx",
            "volatility_10d", "volatility_20d",
            "atr_14", "bb_position", "bb_width",
            "volume_ratio_20d", "obv",
            "rank_return", "excess_return",
            "target_direction",
        ]
        missing = [f for f in expected if f not in features_df.columns]
        assert not missing, f"Missing expected features: {missing}"

    def test_target_is_binary(self, features_df):
        valid = features_df["target_direction"].dropna()
        unique = set(valid.unique())
        assert unique <= {0.0, 1.0}, f"Target has unexpected values: {unique - {0.0, 1.0}}"

    def test_rsi_in_valid_range(self, features_df):
        valid_rsi = features_df["rsi_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), (
            f"RSI out of [0,100]: min={valid_rsi.min()}, max={valid_rsi.max()}"
        )

    def test_stoch_in_valid_range(self, features_df):
        valid = features_df["stoch_k"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), (
            f"Stochastic %K out of [0,100]: min={valid.min()}, max={valid.max()}"
        )
