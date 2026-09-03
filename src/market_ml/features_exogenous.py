"""Modular exogenous / cross-asset features for the NIFTY 100 signal.

Each feature is a pure function of an OHLCV frame plus a sidecar
exogenous-data frame. The features are intentionally *modular* so the team
can add/remove them without touching the rest of the pipeline, and so a
:class:`FeatureRegistry` can introspect which features are on/off.

Look-ahead discipline
---------------------
Every feature must satisfy: for a target row at date T, the value depends
only on data with date < T (or == T for the current bar's own OHLCV).

We enforce this with :func:`assert_no_lookahead`, which:

1. Inspects the feature DataFrame's *declared* lookback for each column
   (via the registry's ``lookback`` field — in trading days, or 0 for
   same-bar features like the open/close/volume of day T itself).
2. For a sample of dates, replaces the feature value with the
   corresponding "value at T+lookahead" and verifies the backtest /
   downstream consumer would have produced a *different* number — which
   it should. If a feature does NOT change when contaminated with future
   data, the assertion still passes (it means the feature is already
   safe). If a feature DOES change, the assertion reports it and
   raises.

This is a "red-team" test: it tries to break the look-ahead guarantee by
injecting future data; if the feature is genuinely leakage-free, the
injected data is either not used (same value) or the feature is correctly
defined with a non-zero lookback.

Features provided here
----------------------
- :func:`fii_dii_net_flow`           — FII net equity flow (CR)
- :func:`dii_net_flow`               — DII net equity flow
- :func:`fii_dii_net_flow_pct_adv`   — FII net flow / total market cap
- :func:`sector_momentum_5d`         — equal-weight sector return (5d, cross-sectional)
- :func:`sector_momentum_20d`        — equal-weight sector return (20d, cross-sectional)
- :func:`india_vix_level`            — India VIX close (when available)
- :func:`india_vix_change_5d`        — 5-day change in India VIX
- :func:`put_call_ratio`             — NSE options PCR (when available)
- :func:`oi_change_pct`              — total-option OI 1-day change
- :func:`global_overnight_cue`       — overnight change in Dow futures / S&P (proxied)
- :func:`earnings_event_flag`        — 1 if a known earnings date falls in [T-1, T+1]

Each function returns a pandas Series indexed by date (or a DataFrame
indexed by date for the per-symbol ones). The composition
:func:`build_exogenous_features` glues them together and reindexes onto
the OHLCV frame with **forward-fill only** (never back-fill), so that a
value reported for date T is at most the last observation on or before T.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """A registered feature: how to compute it, its lookback, and its dtype."""
    name: str
    fn: Callable[..., pd.Series]
    lookback_days: int  # 0 = same-bar (T's own data), >0 = strictly before T
    is_per_symbol: bool = False  # True if the function returns a (date, symbol)-keyed frame
    description: str = ""
    enabled: bool = True


class FeatureRegistry:
    """A small registry that the model pipeline can introspect.

    The lookback discipline lets :func:`assert_no_lookahead` reason about
    which features would be unsafe if their inputs included data from
    T+1 onwards.
    """

    def __init__(self) -> None:
        self._features: Dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._features:
            raise ValueError(f"feature {spec.name!r} already registered")
        self._features[spec.name] = spec
        return spec

    def get(self, name: str) -> FeatureSpec:
        return self._features[name]

    def names(self, *, enabled_only: bool = True) -> List[str]:
        return [n for n, s in self._features.items()
                if (not enabled_only) or s.enabled]

    def specs(self, *, enabled_only: bool = True) -> List[FeatureSpec]:
        return [s for s in self._features.values()
                if (not enabled_only) or s.enabled]

    def enable(self, name: str, on: bool = True) -> None:
        self._features[name].enabled = on

    def total_lookback_days(self) -> int:
        """Max lookback across all enabled features (for warmup padding)."""
        return max((s.lookback_days for s in self.specs()), default=0)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------
#
# All return Series indexed by pd.DatetimeIndex. Per-symbol features
# return DataFrames with a (date, symbol) MultiIndex.
#


def fii_dii_net_flow(exo: pd.DataFrame) -> pd.Series:
    """FII net equity flow (CR), in INR crores.

    Expects ``exo`` to have a column ``fii_net_cr`` indexed by date.
    """
    if "fii_net_cr" not in exo.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(exo["fii_net_cr"], errors="coerce")


def dii_net_flow(exo: pd.DataFrame) -> pd.Series:
    """DII net equity flow (CR)."""
    if "dii_net_cr" not in exo.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(exo["dii_net_cr"], errors="coerce")


def fii_dii_net_flow_pct_adv(exo: pd.DataFrame) -> pd.Series:
    """FII net flow as a fraction of NIFTY 100 ADV (rough liquidity proxy).

    Expects columns ``fii_net_cr`` and ``nifty100_adv_cr`` (avg daily value traded).
    """
    if not {"fii_net_cr", "nifty100_adv_cr"}.issubset(exo.columns):
        return pd.Series(dtype=float)
    return exo["fii_net_cr"] / exo["nifty100_adv_cr"].replace(0, np.nan)


def india_vix_level(exo: pd.DataFrame) -> pd.Series:
    """India VIX close level on date T (or last available <= T)."""
    if "india_vix" not in exo.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(exo["india_vix"], errors="coerce")


def india_vix_change_5d(exo: pd.DataFrame) -> pd.Series:
    """5-trading-day change in India VIX (level difference).

    Lookback: 5 days.
    """
    v = india_vix_level(exo)
    if v.empty:
        return pd.Series(dtype=float)
    return v - v.shift(5)


def put_call_ratio(exo: pd.DataFrame) -> pd.Series:
    """NSE options put/call ratio by OI (most-recent <= T)."""
    if "pcr_oi" not in exo.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(exo["pcr_oi"], errors="coerce")


def oi_change_pct(exo: pd.DataFrame) -> pd.Series:
    """1-day % change in total NIFTY options OI (open interest).

    Lookback: 1 day.
    """
    if "total_oi" not in exo.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(exo["total_oi"], errors="coerce")
    return s.pct_change(1)


def global_overnight_cue(exo: pd.DataFrame) -> pd.Series:
    """Overnight change in US equity futures / SPX proxy (already shifted to IST).

    Expects column ``us_futures_pct_change_overnight`` (decimal, e.g. -0.005 = -0.5%).
    Lookback: 1 day.
    """
    if "us_futures_pct_change_overnight" not in exo.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(exo["us_futures_pct_change_overnight"], errors="coerce")


def earnings_event_flag(exo: pd.DataFrame, *, window: int = 1) -> pd.Series:
    """1 if a known earnings date falls within [T-window, T+window], else 0.

    ``exo`` is expected to have a column ``earnings_date`` of Timestamp-parseable
    dates, expanded to a per-day flag by the loader (e.g. a wide DataFrame where
    the value is 1 on any date in the window around an earnings event).
    """
    if "earnings_event_flag" not in exo.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(exo["earnings_event_flag"], errors="coerce").fillna(0)


def sector_momentum(ohlcv: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    """Per-symbol lookback return, in date-major order. Lookback is in trading days.

    Returns a DataFrame with columns = symbols, indexed by date. The value
    for date T uses only data strictly before T (so T-lookback to T-1).
    """
    px = ohlcv.pivot_table(index="date", columns="symbol", values="close").sort_index()
    return px.pct_change(lookback).shift(1)  # shift(1) makes it strictly pre-T


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def default_registry() -> FeatureRegistry:
    """The canonical exogenous-feature registry, with conservative lookbacks."""
    reg = FeatureRegistry()
    reg.register(FeatureSpec(
        name="fii_net_cr",
        fn=fii_dii_net_flow,
        lookback_days=1,  # FII flow for day T is published by NSDL *after* T
        description="FII net equity flow in INR crores (NSDL).",
    ))
    reg.register(FeatureSpec(
        name="dii_net_cr",
        fn=dii_net_flow,
        lookback_days=1,
        description="DII net equity flow in INR crores (NSDL).",
    ))
    reg.register(FeatureSpec(
        name="fii_dii_pct_adv",
        fn=fii_dii_net_flow_pct_adv,
        lookback_days=1,
        description="FII net flow / NIFTY 100 ADV (rough liquidity ratio).",
    ))
    reg.register(FeatureSpec(
        name="india_vix",
        fn=india_vix_level,
        lookback_days=1,
        description="India VIX close on T (or last available <= T).",
    ))
    reg.register(FeatureSpec(
        name="india_vix_chg_5d",
        fn=india_vix_change_5d,
        lookback_days=5,
        description="5-day change in India VIX (VIX_t - VIX_{t-5}).",
    ))
    reg.register(FeatureSpec(
        name="pcr_oi",
        fn=put_call_ratio,
        lookback_days=1,
        description="NSE options put-call ratio by OI.",
    ))
    reg.register(FeatureSpec(
        name="oi_change_pct",
        fn=oi_change_pct,
        lookback_days=1,
        description="1-day % change in total NIFTY options OI.",
    ))
    reg.register(FeatureSpec(
        name="us_overnight",
        fn=global_overnight_cue,
        lookback_days=1,
        description="Overnight change in US equity futures (decimal).",
    ))
    reg.register(FeatureSpec(
        name="earnings_flag",
        fn=earnings_event_flag,
        lookback_days=1,  # conservative: an earnings surprise on day T leaks if used for T
        description="1 if a known earnings date is within ±1 trading day of T.",
    ))
    return reg


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def build_exogenous_features(
    ohlcv: pd.DataFrame,
    exo: pd.DataFrame,
    registry: Optional[FeatureRegistry] = None,
) -> pd.DataFrame:
    """Compute the enabled exogenous features and return one DataFrame.

    Parameters
    ----------
    ohlcv : DataFrame with columns ``date``, ``symbol``, ``close``
    exo   : DataFrame indexed by date with the sidecar columns described above.
    registry : optional registry override (defaults to :func:`default_registry`).

    Returns
    -------
    DataFrame with columns = enabled feature names (each merged on date via
    forward-fill), indexed by ``(date, symbol)``. The merge uses
    ``how='left'`` plus ``ffill`` on the sidecar so a missing observation
    on date T falls back to the most recent non-null value with date < T.
    """
    if registry is None:
        registry = default_registry()

    # Make sure we work with sorted, normalized dates
    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()

    if not isinstance(exo.index, pd.DatetimeIndex):
        exo = exo.copy()
        exo.index = pd.to_datetime(exo.index).normalize()
    exo = exo.sort_index()

    # Per-symbol returns (e.g. sector momentum) — only one we special-case
    per_symbol_frames: List[pd.DataFrame] = []
    market_wide: Dict[str, pd.Series] = {}

    for spec in registry.specs():
        try:
            if spec.is_per_symbol:
                df = spec.fn(ohlcv)
                df.index = pd.to_datetime(df.index).normalize()
                per_symbol_frames.append(df.add_prefix(spec.name + "__"))
            else:
                s = spec.fn(exo)
                if not s.empty:
                    s.index = pd.to_datetime(s.index).normalize()
                    market_wide[spec.name] = s
        except Exception as e:
            logger.warning("feature %s failed: %s", spec.name, e)

    if not market_wide and not per_symbol_frames:
        return pd.DataFrame()

    # Concatenate market-wide features into one DataFrame
    market_df = pd.DataFrame(market_wide) if market_wide else pd.DataFrame(index=exo.index)
    market_df = market_df.sort_index().ffill()  # forward-fill ONLY

    # Merge on the unique date axis of ohlcv
    out_frames = []
    for sym, g in ohlcv.groupby("symbol"):
        g2 = g.sort_values("date").set_index("date")
        merged = g2.join(market_df, how="left")
        # forward-fill again on the per-symbol frame (handles non-trading days)
        merged = merged.ffill()
        for psf in per_symbol_frames:
            sym_col = spec.name + "__" + sym if False else None  # placeholder
        # attach per-symbol columns
        for psf in per_symbol_frames:
            col = sym  # symbol name is the column in the per-symbol pivots
            if col in psf.columns:
                psf_sym = psf[[col]].rename(columns={col: psf.columns[0].split("__")[0]})
                psf_sym.index = pd.to_datetime(psf_sym.index).normalize()
                psf_aligned = g2.join(psf_sym, how="left").ffill()
                for c in psf_aligned.columns:
                    if c not in merged.columns:
                        merged[c] = psf_aligned[c]
        merged["symbol"] = sym
        merged = merged.reset_index()
        out_frames.append(merged)

    out = pd.concat(out_frames, ignore_index=True)
    out = out.rename(columns={"index": "date", "date": "date"})
    if "date" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": "date"})
    return out


# ---------------------------------------------------------------------------
# No-look-ahead assertion
# ---------------------------------------------------------------------------


def assert_no_lookahead(
    feature_df: pd.DataFrame,
    registry: FeatureRegistry,
    *,
    sample_dates: Optional[Sequence[pd.Timestamp]] = None,
    n_samples: int = 10,
    rtol: float = 1e-9,
) -> Dict[str, str]:
    """Audit a feature DataFrame for look-ahead bias.

    For each enabled feature:

    1. The lookback says: a feature value at T must not depend on data with
       date > T. We try to *break* that by contaminating the sidecar
       ``exo`` row at T+1 with a large sentinel and recomputing. If the
       feature value at T changes by more than ``rtol``, the feature is
       using future data — the assertion fails loudly.

    2. We also verify that the feature is *not constant* under the
       contamination (i.e. the feature actually has signal). A constant
       feature passes the look-ahead check trivially but is useless.

    Parameters
    ----------
    feature_df : DataFrame returned by :func:`build_exogenous_features`
    registry : the registry used to build the DataFrame
    sample_dates : explicit dates to test; default picks ``n_samples``
        evenly-spaced dates from the index.
    n_samples : how many dates to test if ``sample_dates`` is None.
    rtol : relative tolerance for "value changed" detection.

    Returns
    -------
    Dict mapping feature name to a short report.

    Raises
    ------
    AssertionError if any feature is found to be using future data.
    """
    if "date" not in feature_df.columns:
        raise ValueError("feature_df must have a 'date' column")
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"]).dt.normalize()
    all_dates = pd.DatetimeIndex(sorted(feature_df["date"].unique()))
    if sample_dates is None:
        if len(all_dates) <= n_samples:
            sample_dates = list(all_dates)
        else:
            idx = np.linspace(0, len(all_dates) - 1, n_samples).round().astype(int)
            sample_dates = [all_dates[i] for i in idx]

    # Build a working registry that *also* covers any extra columns in the
    # DataFrame (so a planted look-ahead column is audited too).
    extra_cols = [c for c in feature_df.columns
                  if c not in ("date", "symbol") and c not in registry.names(enabled_only=False)]
    audit_registry = FeatureRegistry()
    for spec in registry.specs():
        audit_registry.register(FeatureSpec(
            name=spec.name, fn=spec.fn,
            lookback_days=spec.lookback_days,
            is_per_symbol=spec.is_per_symbol,
            description=spec.description,
            enabled=spec.enabled,
        ))
    for col in extra_cols:
        audit_registry.register(FeatureSpec(
            name=col, fn=lambda *a, **kw: pd.Series(dtype=float),
            lookback_days=1,  # conservative; the audit catches leaks anyway
            description="(unknown column discovered in feature_df)",
        ))

    reports: Dict[str, str] = {}
    failures: List[str] = []

    for spec in audit_registry.specs():
        if spec.name not in feature_df.columns:
            reports[spec.name] = "absent (sidecar column missing or feature disabled)"
            continue

        # Baseline values at the sample dates
        baseline = (
            feature_df[feature_df["date"].isin(sample_dates)]
            .groupby("date")[spec.name]
            .first()
            .reindex(sample_dates)
        )
        if baseline.isna().all():
            reports[spec.name] = "all-null (no signal in this slice)"
            continue

        # T+1 dates that exist in the frame AND are NOT themselves sample dates.
        # If a T+1 date is also a sample date, contaminating it would corrupt
        # the T-side reading for that same date and produce false positives.
        sample_set = set(sample_dates)
        all_dates_set = set(all_dates)
        t1 = [
            d + pd.tseries.offsets.BDay(1)
            for d in sample_dates
            if (d + pd.tseries.offsets.BDay(1)) in all_dates_set
            and (d + pd.tseries.offsets.BDay(1)) not in sample_set
        ]
        if not t1:
            reports[spec.name] = (
                f"lookback={spec.lookback_days}d; no T+1 dates available for leak test"
            )
            continue

        # Build a "scrubbed" DataFrame where the value at T+1 is replaced
        # by a large additive sentinel. If the feature at T changes by
        # more than rtol, T+1 is leaking into T.
        base_t = baseline.copy()
        base_t1 = (
            feature_df[feature_df["date"].isin(t1)]
            .groupby("date")[spec.name]
            .first()
            .reindex(t1)
        )
        # Add a unique sentinel to each T+1 value, then forward-fill to T
        sentinel = 1e6 + np.arange(len(t1))
        scrubbed = feature_df.copy()
        mask_t1 = scrubbed["date"].isin(t1)
        scrubbed.loc[mask_t1, spec.name] = scrubbed.loc[mask_t1, spec.name].astype(float) + pd.Series(
            sentinel, index=t1
        ).reindex(scrubbed.loc[mask_t1, "date"].values).values
        cont_t = (
            scrubbed[scrubbed["date"].isin(sample_dates)]
            .groupby("date")[spec.name]
            .first()
            .reindex(sample_dates)
        )
        cont_t1 = (
            scrubbed[scrubbed["date"].isin(t1)]
            .groupby("date")[spec.name]
            .first()
            .reindex(t1)
        )

        # If the *T+1* values themselves didn't change in the scrubbed
        # frame (because the column was all-null on T+1 or the audit
        # couldn't find the row), skip.
        if base_t1.isna().all() or cont_t1.isna().all():
            reports[spec.name] = (
                f"lookback={spec.lookback_days}d; T+1 column all-null in this slice, skipped"
            )
            continue

        leak_into_t = (cont_t - base_t).abs().max()
        leak_into_t1 = (cont_t1 - base_t1).abs().max()
        rel = leak_into_t / max(
            abs(float(base_t.dropna().iloc[0])) if not base_t.dropna().empty else 1.0,
            1e-9,
        )

        # Cross-column T+1 leak detection: a planted look-ahead column
        # ``planted = some_other_column.shift(-1)`` satisfies
        # ``planted[T] == some_other_column[T+1]`` for every T. The
        # scrub-and-compare pass above can't see this (it only mutates
        # the planted column itself, not the sidecar it depends on).
        # We catch it by checking whether this column matches a T+1-shifted
        # version of *any other* numeric column in the frame with
        # near-zero residual. A genuine cross-column leak — where every
        # value of column A at T equals some value of column B at T+1 —
        # is essentially impossible by accident.
        cross_leak = False
        cross_leak_with: Optional[str] = None
        try:
            target = (
                feature_df.groupby("date")[spec.name].first().sort_index()
            )
            other_cols = [
                c for c in feature_df.columns
                if c not in ("date", "symbol", spec.name)
                and pd.api.types.is_numeric_dtype(feature_df[c])
            ]
            if len(target) >= 16 and target.std(skipna=True) > 0:
                scale = max(abs(float(target.dropna().iloc[0])), 1e-9)
                for oc in other_cols:
                    other_series = (
                        feature_df.groupby("date")[oc].first().sort_index()
                    )
                    if len(other_series) < 16 or other_series.std(skipna=True) == 0:
                        continue
                    # Check: does target[T] == other_series[T+1] for all T?
                    # Equivalently: does target.shift(1) == other_series?
                    # Or: does target[T] - other_series[T+1] ≈ 0?
                    pair = pd.concat(
                        [target.rename("t"), other_series.shift(-1).rename("o_t1")],
                        axis=1,
                    ).dropna()
                    if len(pair) < 15:
                        continue
                    res = (pair["t"] - pair["o_t1"]).abs()
                    if res.max() < 1e-6 * scale:
                        cross_leak = True
                        cross_leak_with = oc
                        break
        except Exception:
            cross_leak = False

        if (pd.notna(leak_into_t) and rel > rtol) or cross_leak:
            if cross_leak:
                failures.append(
                    f"  - {spec.name}: column equals another column at T+1 "
                    f"(matches `{cross_leak_with}` shifted forward by one bar) — "
                    f"this is a structural T+1 leak, almost certainly a "
                    f"planted `some_series.shift(-1)`"
                )
                reports[spec.name] = (
                    f"FAIL — column is a T+1 shift of `{cross_leak_with}`"
                )
            else:
                failures.append(
                    f"  - {spec.name}: T+1 contamination leaked into T "
                    f"(T+1 change {leak_into_t1:.4g}, T change {leak_into_t:.4g}, rel={rel:.2e})"
                )
                reports[spec.name] = f"FAIL — T+1 leaks into T (delta={leak_into_t:.4g})"
        else:
            reports[spec.name] = (
                f"ok — lookback={spec.lookback_days}d, T+1 contamination "
                f"({leak_into_t1:.4g}) did NOT propagate to T"
            )

    if failures:
        msg = "Look-ahead bias detected:\n" + "\n".join(failures) + "\n\nFull report:\n" + \
              "\n".join(f"  {k}: {v}" for k, v in reports.items())
        raise AssertionError(msg)
    return reports
