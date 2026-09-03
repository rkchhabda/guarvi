"""Indian retail trading cost model.

Loads configurable cost constants from ``config/costs.yaml`` and computes the
per-side (and round-trip) cost of an equity delivery trade. The math here is
isolated from the backtest engine so it can be unit-tested in isolation.

All percentage values are in *percent* (e.g. 0.03 means 0.03%, NOT 3%).
To convert a percent to a decimal: ``pct / 100.0``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover — yaml is in the requirements file
    yaml = None  # type: ignore


COST_KEYS = (
    "brokerage_pct",
    "brokerage_flat_inr",
    "stt_sell_pct",
    "exchange_txn_pct",
    "sebi_charges_pct",
    "stamp_duty_buy_pct",
    "gst_on_brokerage_pct",
    "slippage_bps_per_side",
)


@dataclass
class CostConfig:
    brokerage_pct: float = 0.03
    brokerage_flat_inr: float = 20.0
    stt_sell_pct: float = 0.025
    exchange_txn_pct: float = 0.00325
    sebi_charges_pct: float = 0.0001
    stamp_duty_buy_pct: float = 0.015
    gst_on_brokerage_pct: float = 18.0
    slippage_bps_per_side: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


def _to_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_costs(
    config_path: Optional[Path] = None,
    *,
    env_prefix: str = "GUARVI_COST_",
) -> CostConfig:
    """Load cost config from YAML, then overlay any env-var overrides.

    Env-var names mirror the YAML keys with the prefix, e.g.
    ``GUARVI_COST_BROKERAGE_PCT=0.0`` to set brokerage to zero.
    """
    cfg = CostConfig()
    if config_path and Path(config_path).exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        costs = (data.get("costs") or {})
        for k in COST_KEYS:
            if k in costs and costs[k] is not None:
                setattr(cfg, k, _to_float(costs[k], getattr(cfg, k)))

    for k in COST_KEYS:
        env_name = env_prefix + k.upper()
        if env_name in os.environ:
            setattr(cfg, k, _to_float(os.environ[env_name], getattr(cfg, k)))

    return cfg


def round_trip_cost_pct(
    cfg: CostConfig,
    *,
    side_turnover: float = 1.0,
    sell_value_inr: float = 0.0,
    buy_value_inr: float = 0.0,
) -> float:
    """Total cost in *percent* of one side's notional.

    Parameters
    ----------
    cfg : CostConfig
        Cost configuration.
    side_turnover : float
        Fraction of one side's notional that is traded. Default 1.0 (full leg).
    sell_value_inr / buy_value_inr : float
        Optional notional values (used to decide whether the flat-Rs brokerage
        floor binds; otherwise the floor is ignored and the percentage figure
        alone is used). Both default to 0 (percentage-only model).

    Returns
    -------
    float
        Cost as a percentage of one side's notional, *including* slippage.
    """
    p = side_turnover

    brokerage_pct = cfg.brokerage_pct * p
    # Rs 20 floor binds only if the notional on the relevant side is large
    # enough that 0.03% of it would exceed Rs 20. With the default
    # brokerage_pct=0.03% the threshold is 20 / 0.0003 ≈ Rs 66,667.
    if sell_value_inr > 0 and cfg.brokerage_pct * p * sell_value_inr / 100.0 < cfg.brokerage_flat_inr:
        brokerage_pct = max(brokerage_pct, cfg.brokerage_flat_inr * 100.0 / max(sell_value_inr, 1.0))
    if buy_value_inr > 0 and cfg.brokerage_pct * p * buy_value_inr / 100.0 < cfg.brokerage_flat_inr:
        brokerage_pct = max(brokerage_pct, cfg.brokerage_flat_inr * 100.0 / max(buy_value_inr, 1.0))

    stt_sell_pct = cfg.stt_sell_pct * p
    exchange_pct = cfg.exchange_txn_pct * 2.0 * p  # both sides
    sebi_pct = cfg.sebi_charges_pct * 2.0 * p
    stamp_pct = cfg.stamp_duty_buy_pct * p
    gst_pct = (cfg.gst_on_brokerage_pct / 100.0) * brokerage_pct
    # Conservative: also add GST on the exchange txn and SEBI fees
    gst_pct += (cfg.gst_on_brokerage_pct / 100.0) * (cfg.exchange_txn_pct * 2.0 * p)
    gst_pct += (cfg.gst_on_brokerage_pct / 100.0) * (cfg.sebi_charges_pct * 2.0 * p)
    slip_pct = cfg.slippage_bps_per_side * 2.0 * p / 100.0  # both sides

    return brokerage_pct + stt_sell_pct + exchange_pct + sebi_pct + stamp_pct + gst_pct + slip_pct


def round_trip_cost_decimal(cfg: CostConfig, **kw) -> float:
    """Same as :func:`round_trip_cost_pct` but returns a decimal (0.0005 = 5 bps)."""
    return round_trip_cost_pct(cfg, **kw) / 100.0
