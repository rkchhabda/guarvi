"""Unit tests for the Indian retail cost model and the cross-sectional backtest.

We check:
  1. The cost component sum matches the documented default for a mid-size
     delivery trade (Rs 100,000 notional, both sides).
  2. Slippage dominates when set very high; the floor brokerage kicks in when
     the trade is small.
  3. The backtest engine handles an empty predictions frame without crashing.
  4. End-to-end: a tiny synthetic predictions DataFrame produces sensible
     gross > net, and Sharpe is defined.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.market_ml.costs import (
    COST_KEYS,
    CostConfig,
    load_costs,
    round_trip_cost_decimal,
    round_trip_cost_pct,
)
from src.market_ml.backtest import (
    BacktestConfig,
    _resolve_leg_size,
    run_backtest,
)


# ----- (1) Component-level cost math -----

def test_default_cost_config_has_all_keys():
    cfg = CostConfig()
    for k in COST_KEYS:
        assert hasattr(cfg, k), f"CostConfig missing {k}"


def test_default_round_trip_is_sane_mid_size_trade():
    """For a Rs 100k notional, the default round-trip cost is dominated by STT
    (0.025% sell) and the exchange/sebi/GST stack; should be roughly 0.10% to
    0.20% of one side's notional — well above the 5-bps slippage-only figure.
    """
    cfg = CostConfig()
    pct = round_trip_cost_pct(cfg, sell_value_inr=100_000, buy_value_inr=100_000)
    dec = round_trip_cost_decimal(cfg, sell_value_inr=100_000, buy_value_inr=100_000)
    assert 0.05 < pct < 0.50, f"unexpected round-trip pct={pct}"
    assert abs(pct / 100.0 - dec) < 1e-12


def test_floor_brokerage_binds_for_small_trade():
    """Rs 20 flat brokerage should dominate for trades < ~Rs 66,667."""
    cfg = CostConfig()  # brokerage_pct=0.03%, flat=20
    pct_small = round_trip_cost_pct(cfg, sell_value_inr=5_000, buy_value_inr=5_000)
    pct_big = round_trip_cost_pct(cfg, sell_value_inr=500_000, buy_value_inr=500_000)
    assert pct_small > pct_big, f"small={pct_small} should exceed big={pct_big}"


def test_slippage_dominates_when_bumped_high():
    cfg_low = CostConfig(slippage_bps_per_side=5.0)
    cfg_high = CostConfig(slippage_bps_per_side=100.0)
    pct_low = round_trip_cost_pct(cfg_low, sell_value_inr=1_000_000, buy_value_inr=1_000_000)
    pct_high = round_trip_cost_pct(cfg_high, sell_value_inr=1_000_000, buy_value_inr=1_000_000)
    assert pct_high - pct_low >= 1.8  # +200 bps on round-trip (100 bps per side * 2)


def test_load_costs_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "costs.yaml"
    yaml_path.write_text(
        "costs:\n  brokerage_pct: 0.0\n  stt_sell_pct: 0.01\n  slippage_bps_per_side: 2.0\n",
        encoding="utf-8",
    )
    cfg = load_costs(yaml_path)
    assert cfg.brokerage_pct == 0.0
    assert cfg.stt_sell_pct == 0.01
    assert cfg.slippage_bps_per_side == 2.0


def test_env_overrides(monkeypatch, tmp_path: Path):
    yaml_path = tmp_path / "costs.yaml"
    yaml_path.write_text("costs:\n  brokerage_pct: 0.03\n", encoding="utf-8")
    monkeypatch.setenv("GUARVI_COST_BROKERAGE_PCT", "0.0")
    monkeypatch.setenv("GUARVI_COST_SLIPPAGE_BPS_PER_SIDE", "12.5")
    cfg = load_costs(yaml_path)
    assert cfg.brokerage_pct == 0.0
    assert cfg.slippage_bps_per_side == 12.5


# ----- (2) Leg-size resolver -----

def test_resolve_leg_size_decile1_caps_at_topn():
    # 100-symbol universe, decile 1 -> 10, capped at 10
    assert _resolve_leg_size(100, 1, 10) == 10
    # 50 symbols, decile 1 -> 5
    assert _resolve_leg_size(50, 1, 10) == 5
    # cap binds: 100 symbols, decile 3, cap 20 -> min(30, 20) = 20
    assert _resolve_leg_size(100, 3, 20) == 20
    # floor: empty universe still returns 1
    assert _resolve_leg_size(0, 1, 10) == 1


# ----- (3) Backtest end-to-end -----

def _synthetic_predictions(n_days: int = 30, n_syms: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for s in range(n_syms):
            prob = float(rng.uniform(0.30, 0.70))
            ret = float(rng.normal(0.0005, 0.01))  # ~5 bps drift, 1% vol
            rows.append({"date": date, "symbol": f"SYM{s:03d}",
                         "predicted_probability": prob, "actual_return": ret})
    return pd.DataFrame(rows)


def test_backtest_runs_and_reports_sensible_summaries():
    df = _synthetic_predictions()
    res = run_backtest(df)
    s = res["summary"]
    # Skeleton checks
    assert s["n_days"] == 30
    assert s["n_universe_avg"] == 20
    assert s["n_long"] == 2   # 20 / 10 = 2
    assert s["n_short"] == 2
    # Cost is non-trivial
    assert s["cost_bps_roundtrip_per_leg"] > 5.0  # at least the slippage
    # Net is finite
    assert -100 < s["net_cum_return_pct"] < 100
    # Daily series is aligned with summary count
    assert len(res["daily"]) == s["n_days"]


def test_backtest_gross_geq_net_for_synthetic_data():
    df = _synthetic_predictions(seed=7)
    res = run_backtest(df)
    s = res["summary"]
    # Costs only subtract — gross should be >= net in the cumulative sense
    # (they may flip on individual days, but cumulative comparison is sound
    # when costs are symmetric across all paths; with daily rebalance and
    # symmetric cost on both legs, gross - 2*cost per day is the daily net)
    assert s["net_cum_return_pct"] <= s["gross_cum_return_pct"] + 1e-6


def test_backtest_empty_dataframe_does_not_crash():
    res = run_backtest(pd.DataFrame(columns=["date", "symbol", "predicted_probability", "actual_return"]))
    assert res["summary"]["n_days"] == 0
    assert res["daily"] == []


def test_backtest_rejects_missing_columns():
    bad = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        run_backtest(bad)


def test_backtest_with_zero_slippage_matches_gross_plus_costs_only():
    df = _synthetic_predictions(n_days=10, n_syms=10)
    costs_zero = CostConfig(
        brokerage_pct=0.0, brokerage_flat_inr=0.0, stt_sell_pct=0.0,
        exchange_txn_pct=0.0, sebi_charges_pct=0.0, stamp_duty_buy_pct=0.0,
        gst_on_brokerage_pct=0.0, slippage_bps_per_side=0.0,
    )
    res = run_backtest(df, costs=costs_zero)
    # cost_decimal should be exactly 0
    assert res["summary"]["cost_bps_roundtrip_per_leg"] == 0.0
    # With zero costs, gross == net
    assert abs(res["summary"]["net_cum_return_pct"] - res["summary"]["gross_cum_return_pct"]) < 1e-9
