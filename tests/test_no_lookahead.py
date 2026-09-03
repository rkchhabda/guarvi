"""Tests for the no-look-ahead feature audit.

We verify that:
  1. A clean, properly-built feature frame passes the assertion.
  2. A *planted* look-ahead column (a feature that equals the *next day's*
     value of some sidecar) is flagged loudly.
  3. A constant column passes trivially (no future data, no past data).
  4. The end-to-end composition + audit works on a tiny synthetic OHLCV
     and exogenous frame.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.market_ml.features_exogenous import (
    FeatureRegistry,
    FeatureSpec,
    assert_no_lookahead,
    build_exogenous_features,
    default_registry,
    fii_dii_net_flow,
    india_vix_change_5d,
    put_call_ratio,
)


# ----- Helpers --------------------------------------------------------------


def _synthetic_exo(n_days: int = 30) -> pd.DataFrame:
    """Sidecar DataFrame with a few canonical columns."""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "fii_net_cr": rng.normal(100, 500, n_days),
        "dii_net_cr": rng.normal(200, 400, n_days),
        "nifty100_adv_cr": np.full(n_days, 5000.0),
        "india_vix": rng.normal(15, 2, n_days).clip(10, 30),
        "pcr_oi": rng.normal(1.0, 0.2, n_days).clip(0.3, 2.0),
        "total_oi": rng.normal(1_000_000, 50_000, n_days),
        "us_futures_pct_change_overnight": rng.normal(0, 0.005, n_days),
        "earnings_event_flag": (rng.uniform(0, 1, n_days) < 0.05).astype(int),
    }, index=dates)


def _synthetic_ohlcv(n_days: int = 30, n_syms: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(7)
    rows = []
    for d in dates:
        for s in range(n_syms):
            close = 100 + rng.normal(0, 1)
            rows.append({"date": d, "symbol": f"S{s:02d}",
                         "open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1_000_000})
    return pd.DataFrame(rows)


# ----- (1) End-to-end on clean synthetic data ------------------------------


def test_end_to_end_clean_features_pass_assertion():
    ohlcv = _synthetic_ohlcv()
    exo = _synthetic_exo(ohlcv["date"].nunique())
    feats = build_exogenous_features(ohlcv, exo, default_registry())
    assert not feats.empty, "no features were built"
    # The default registry has 9 features; some may be absent if columns missing.
    # We require at least 5 to have materialised.
    feature_cols = [c for c in feats.columns if c not in ("date", "symbol")]
    assert len(feature_cols) >= 5, f"too few features: {feature_cols}"
    # And the audit should pass without raising
    report = assert_no_lookahead(feats, default_registry(), n_samples=10)
    assert isinstance(report, dict)
    assert all("FAIL" not in v for v in report.values()), f"unexpected failure: {report}"


# ----- (2) Planted look-ahead must be flagged ------------------------------


def test_planted_lookahead_is_detected():
    """Build a feature that, by construction, equals the *next day's* value
    of the sidecar. The audit must raise.
    """
    ohlcv = _synthetic_ohlcv()
    exo = _synthetic_exo(ohlcv["date"].nunique())

    # Build the clean baseline first
    feats = build_exogenous_features(ohlcv, exo, default_registry())
    # Now add a planted look-ahead column: feature[T] = exo.fii_net_cr[T+1]
    nxt = exo["fii_net_cr"].shift(-1)
    nxt.name = "planted_leak"
    feats = feats.merge(
        nxt.rename("planted_leak").reset_index().rename(columns={"index": "date"}),
        on="date", how="left",
    )

    # The audit must catch this
    with pytest.raises(AssertionError) as exc:
        assert_no_lookahead(feats, default_registry(), n_samples=15)
    assert "planted_leak" in str(exc.value) or "Look-ahead" in str(exc.value)


# ----- (3) A custom feature with explicit lookback is respected ------------


def test_explicit_lookback_5d_vix_is_safe():
    """india_vix_change_5d uses VIX[T] - VIX[T-5]. The audit must not flag it."""
    ohlcv = _synthetic_ohlcv()
    exo = _synthetic_exo(60)  # need enough history for a 5-day shift
    feats = build_exogenous_features(ohlcv, exo, default_registry())
    assert "india_vix_chg_5d" in feats.columns
    report = assert_no_lookahead(feats, default_registry(), n_samples=20)
    assert "india_vix_chg_5d" in report
    assert "FAIL" not in report["india_vix_chg_5d"]


# ----- (4) Constant / absent columns don't crash the audit -----------------


def test_constant_column_does_not_crash_audit():
    ohlcv = _synthetic_ohlcv()
    exo = _synthetic_exo()
    feats = build_exogenous_features(ohlcv, exo, default_registry())
    # Replace one column with a constant; the audit should pass.
    feats["fii_net_cr"] = 0.0
    report = assert_no_lookahead(feats, default_registry(), n_samples=10)
    assert "fii_net_cr" in report
    assert "FAIL" not in report["fii_net_cr"]


# ----- (5) Registry mechanics ----------------------------------------------


def test_registry_register_and_enable():
    reg = FeatureRegistry()
    reg.register(FeatureSpec(name="x", fn=fii_dii_net_flow, lookback_days=1))
    assert "x" in reg.names()
    reg.enable("x", False)
    assert "x" not in reg.names()
    reg.enable("x", True)
    assert "x" in reg.names()
    with pytest.raises(ValueError):
        reg.register(FeatureSpec(name="x", fn=put_call_ratio, lookback_days=1))


def test_registry_total_lookback():
    reg = FeatureRegistry()
    reg.register(FeatureSpec(name="a", fn=fii_dii_net_flow, lookback_days=1))
    reg.register(FeatureSpec(name="b", fn=india_vix_change_5d, lookback_days=5))
    reg.register(FeatureSpec(name="c", fn=put_call_ratio, lookback_days=1))
    assert reg.total_lookback_days() == 5
    reg.enable("b", False)
    assert reg.total_lookback_days() == 1


# ----- (6) Per-feature function tests --------------------------------------


def test_fii_dii_returns_series_for_real_column():
    exo = _synthetic_exo(10)
    s = fii_dii_net_flow(exo)
    assert isinstance(s, pd.Series)
    assert len(s) == 10
    assert not s.isna().all()


def test_fii_dii_returns_empty_when_column_missing():
    exo = pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=5))
    s = fii_dii_net_flow(exo)
    assert s.empty


def test_pcr_handles_missing_column():
    s = put_call_ratio(pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=5)))
    assert s.empty
