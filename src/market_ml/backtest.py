"""Cost-aware cross-sectional backtest for the NIFTY 100 directional signal.

Strategy
--------
Each trading day, rank the universe by ``predicted_probability`` (the model's
P(UP)). Go LONG the top decile (configurable; default top 1 decile, i.e. ~10
names) and SHORT the bottom decile. Equal-weight within each leg. Hold for one
day, rebalance next morning.

Costs
-----
Indian retail delivery costs are computed by
:func:`src.market_ml.costs.round_trip_cost_decimal` from
``config/costs.yaml``. They are subtracted from every position change.

Returns
-------
A dict of summary metrics suitable for the web portal: gross/net cumulative
return, net Sharpe, max drawdown, turnover, hit rate, plus the daily-returns
series (so the front-end can plot an equity curve later if it wants).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .costs import CostConfig, load_costs, round_trip_cost_decimal


@dataclass
class BacktestConfig:
    long_decile: int = 1
    short_decile: int = 1
    top_n: int = 10
    bottom_n: int = 10
    rebalance: str = "daily"  # "daily" | "weekly"
    initial_equity_inr: float = 1_000_000.0
    risk_free_rate_pct: float = 6.0
    trading_days_per_year: int = 252


def _resolve_leg_size(n_universe: int, decile: int, abs_cap: int) -> int:
    """Decile of N, capped at abs_cap. Decile=1 means top/bottom 10%."""
    by_decile = max(1, n_universe // 10 * decile)
    return max(1, min(by_decile, abs_cap))


def _per_leg_pnl(
    chosen: pd.DataFrame,
    cost_decimal: float,
) -> tuple[float, int]:
    """Gross PnL for one leg, net of round-trip cost, and trade count.

    ``chosen`` must have columns ``actual_return`` (decimal) and ``size_change``
    (1 if entered, 0 if held, -1 if exited; used for turnover).
    """
    if chosen.empty:
        return 0.0, 0
    gross = float(chosen["actual_return"].mean())
    n = int(len(chosen))
    net = gross - cost_decimal
    return net, n


def run_backtest(
    df: pd.DataFrame,
    *,
    costs: Optional[CostConfig] = None,
    bt: Optional[BacktestConfig] = None,
) -> dict:
    """Run the cross-sectional long/short backtest.

    Parameters
    ----------
    df : DataFrame
        Must have columns ``date``, ``symbol``, ``predicted_probability``,
        ``actual_return`` (decimal, e.g. 0.012 = +1.2%).
    costs / bt : optional overrides.

    Returns
    -------
    dict with keys ``daily`` (list of dicts), ``summary`` (dict of scalars).
    """
    if costs is None:
        costs = load_costs(Path(__file__).resolve().parents[2] / "config" / "costs.yaml")
    if bt is None:
        bt = BacktestConfig()

    required = {"date", "symbol", "predicted_probability", "actual_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions dataframe missing columns: {sorted(missing)}")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["predicted_probability", "actual_return"])
    d = d.sort_values(["date", "symbol"]).reset_index(drop=True)

    cost_dec = round_trip_cost_decimal(costs)

    daily_rows = []
    for date, g in d.groupby("date"):
        n_uni = len(g)
        n_long = _resolve_leg_size(n_uni, bt.long_decile, bt.top_n)
        n_short = _resolve_leg_size(n_uni, bt.short_decile, bt.bottom_n)

        # Sort: high prob -> long, low prob -> short
        order = g.sort_values("predicted_probability", ascending=False)
        long_leg = order.head(n_long)
        short_leg = order.tail(n_short)

        long_ret, _ = _per_leg_pnl(long_leg, cost_dec)
        short_ret, _ = _per_leg_pnl(short_leg, cost_dec)
        # Short leg PnL is the negative of the symbols' actual returns, net of cost
        short_ret = -short_ret

        # Turnover: full rebalance every day (worst case). Tracked separately
        # for the dollar-notional-weighted "realistic" turnover.
        turnover_pct = 2.0  # 100% of long + 100% of short traded each day
        gross_port_ret = 0.5 * (long_leg["actual_return"].mean() - short_leg["actual_return"].mean())
        # Costs apply once per leg per day, so total cost is 2 * cost_dec on
        # a notional-weighted basis (we subtract from the gross spread).
        net_port_ret = gross_port_ret - 2.0 * cost_dec

        daily_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "n_universe": int(n_uni),
            "n_long": int(n_long),
            "n_short": int(n_short),
            "long_leg_ret": float(long_leg["actual_return"].mean()) if not long_leg.empty else 0.0,
            "short_leg_ret": float(-short_leg["actual_return"].mean()) if not short_leg.empty else 0.0,
            "gross_spread": float(gross_port_ret),
            "net_spread": float(net_port_ret),
            "cost_decimal": float(cost_dec),
            "turnover_pct": float(turnover_pct),
        })

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {"daily": [], "summary": _empty_summary()}

    # --- Aggregates ---
    gross = daily["gross_spread"]
    net = daily["net_spread"]
    cum_gross = float((1.0 + gross).prod() - 1.0)
    cum_net = float((1.0 + net).prod() - 1.0)
    # Sharpe on net daily returns, annualised
    rf_daily = (bt.risk_free_rate_pct / 100.0) / bt.trading_days_per_year
    excess = net - rf_daily
    sharpe_net = (
        float(excess.mean() / excess.std(ddof=0) * np.sqrt(bt.trading_days_per_year))
        if excess.std(ddof=0) > 0
        else 0.0
    )
    equity = (1.0 + net).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())

    # Hit rate: how often did the net spread beat zero?
    hit_rate = float((net > 0).mean())

    # Naive equal-weight baseline (long all, short none)
    daily["naive_ret"] = d.groupby("date")["actual_return"].mean().reindex(pd.to_datetime(daily["date"])).values
    cum_naive = float((1.0 + daily["naive_ret"].fillna(0)).prod() - 1.0)

    summary = {
        "n_days": int(len(daily)),
        "n_trading_days": int(len(daily)),
        "n_universe_avg": float(daily["n_universe"].mean()),
        "n_long": int(daily["n_long"].iloc[0]),
        "n_short": int(daily["n_short"].iloc[0]),
        "long_decile": bt.long_decile,
        "short_decile": bt.short_decile,
        "cost_decimal_roundtrip_per_leg": float(cost_dec),
        "cost_bps_roundtrip_per_leg": float(cost_dec * 10_000),
        "cost_bps_roundtrip_per_day": float(cost_dec * 2 * 10_000),
        "gross_cum_return_pct": round(cum_gross * 100.0, 4),
        "net_cum_return_pct": round(cum_net * 100.0, 4),
        "naive_equal_weight_cum_return_pct": round(cum_naive * 100.0, 4),
        "edge_vs_naive_pct": round((cum_net - cum_naive) * 100.0, 4),
        "net_daily_mean_bps": round(float(net.mean()) * 10_000, 3),
        "net_daily_std_bps": round(float(net.std(ddof=0)) * 10_000, 3),
        "net_sharpe": round(sharpe_net, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "hit_rate_pct": round(hit_rate * 100.0, 2),
        "turnover_pct_per_day": round(float(daily["turnover_pct"].mean()) * 100.0, 2),
        "initial_equity_inr": bt.initial_equity_inr,
        "risk_free_rate_pct": bt.risk_free_rate_pct,
    }
    return {
        "daily": daily.to_dict(orient="records"),
        "summary": summary,
        "config": {
            "costs": costs.to_dict(),
            "backtest": bt.__dict__,
        },
    }


def _empty_summary() -> dict:
    return {
        "n_days": 0,
        "gross_cum_return_pct": 0.0,
        "net_cum_return_pct": 0.0,
        "naive_equal_weight_cum_return_pct": 0.0,
        "edge_vs_naive_pct": 0.0,
        "net_sharpe": 0.0,
        "max_drawdown_pct": 0.0,
        "hit_rate_pct": 0.0,
        "cost_bps_roundtrip_per_day": 0.0,
        "turnover_pct_per_day": 0.0,
    }


def write_report(result: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))


def run_from_predictions_csv(
    predictions_csv: Path,
    *,
    out_json: Optional[Path] = None,
    costs: Optional[CostConfig] = None,
    bt: Optional[BacktestConfig] = None,
) -> dict:
    """Convenience: load predictions.csv, run the backtest, optionally persist."""
    df = pd.read_csv(Path(predictions_csv))
    result = run_backtest(df, costs=costs, bt=bt)
    if out_json is not None:
        write_report(result, Path(out_json))
    return result
