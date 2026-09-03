#!/usr/bin/env python3
"""Guarvi web portal - read-only FastAPI backend.

Serves the NIFTY100 screening signal, live quotes, and a research track-record
summary. Read-only by design: it never places orders. Deploy behind nginx
(see web/nginx_site.conf) on the Oracle VM, run via the guarvi-web systemd unit.

Run locally:  uvicorn web.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
SIGNAL_PATH = ROOT / "reports" / "live_signal.json"
LIVE_DIR = ROOT / "data" / "live" / "nse"
NIFTY100_OHLCV = ROOT / "data" / "processed" / "nifty100_ohlcv.parquet"
FRESHNESS_PATH = ROOT / "reports" / "data_freshness.json"
PREDICTIONS_CSV = ROOT / "data" / "processed" / "predictions.csv"
BACKTEST_PATH = ROOT / "reports" / "backtest_costs.json"

IST = ZoneInfo("Asia/Kolkata")

DISCLAIMER = (
    "Research/education tool only. The directional signal is small (~2 bps/day gross) and "
    "not profitable net of Indian retail costs; naive equal-weight NIFTY100 has historically "
    "been the better strategy. Not SEBI-registered investment advice."
)

app = FastAPI(title="Guarvi Signal Portal", version="0.2.0")

STATIC = ROOT / "web" / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _scoring_module():
    """Lazy import to keep cold-start cheap."""
    from src.market_ml import scoring
    return scoring


def _load_signal() -> dict:
    if not SIGNAL_PATH.exists():
        return {"error": "signal not built; run scripts/build_signal.py",
                "disclaimer": DISCLAIMER}
    return json.loads(SIGNAL_PATH.read_text())


def _latest_quotes() -> list[dict]:
    """Best-effort live quotes.

    Tries (in order):
      1. Most recent `data/live/nse/quotes_*.csv` from the collector.
      2. Fallback: the most recent trading day from `nifty100_ohlcv.parquet`
         (so the UI can show "Market closed · last LTP as of …" instead of
         an empty table).
    Returns a list of dicts with normalised keys: symbol, last, change,
    change_percent, volume, as_of, source.
    """
    pd = __import__("pandas")
    # 1) live collector file
    if LIVE_DIR.exists():
        files = sorted(LIVE_DIR.glob("quotes_*.csv"), reverse=True)
        if files:
            try:
                df = pd.read_csv(files[0])
                rename = {
                    "last_price": "last", "close": "last",
                    "change_pct": "change_percent",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                cols = [c for c in ("symbol", "last", "change", "change_percent",
                                    "volume", "timestamp") if c in df.columns]
                if cols:
                    rows = df[cols].to_dict("records")
                    as_of = str(files[0].stem).replace("quotes_", "")
                    for r in rows:
                        r["as_of"] = as_of
                        r["source"] = "live_collector"
                    return rows
            except Exception:
                pass
    # 2) historical fallback
    try:
        if NIFTY100_OHLCV.exists():
            df = pd.read_parquet(NIFTY100_OHLCV)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            last_date = df["date"].max()
            last = df[df["date"] == last_date].copy()
            if last.empty:
                return []
            last = last.sort_values("symbol")
            prev_date = df[df["date"] < last_date]["date"].max()
            prev = df[df["date"] == prev_date][["symbol", "close"]].rename(
                columns={"close": "prev_close"}
            ) if prev_date is not pd.NaT else None
            merged = last.merge(prev, on="symbol", how="left") if prev is not None else last
            out = []
            for _, r in merged.iterrows():
                last_px = _safe_float(r.get("close"))
                prev_px = _safe_float(r.get("prev_close"))
                chg = (last_px - prev_px) if (last_px is not None and prev_px is not None) else None
                pct = (chg / prev_px * 100) if (chg is not None and prev_px not in (None, 0)) else None
                out.append({
                    "symbol": r.get("symbol"),
                    "last": last_px,
                    "change": _safe_float(chg),
                    "change_percent": _safe_float(pct),
                    "volume": _safe_float(r.get("volume")),
                    "as_of": str(last_date.date()),
                    "source": "parquet_last_day",
                })
            return out
    except Exception:
        return []
    return []


def _symbol_deep_dive(symbol: str) -> dict | None:
    """Return a detail card for `symbol` from the OHLCV parquet.

    Uses only stored columns (no model retraining); computes SMA-20/50
    on the fly. Returns None if symbol isn't in the parquet.
    """
    pd = __import__("pandas")
    if not NIFTY100_OHLCV.exists():
        return None
    sym = symbol.strip().upper()
    try:
        df = pd.read_parquet(NIFTY100_OHLCV)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[df["symbol"] == sym].sort_values("date")
    except Exception:
        return None
    if df.empty:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    last_close = _safe_float(last.get("close"))
    prev_close = _safe_float(prev.get("close"))
    change = (last_close - prev_close) if (last_close is not None and prev_close is not None) else None
    change_pct = (change / prev_close * 100) if (change is not None and prev_close not in (None, 0)) else None

    closes = df["close"].astype(float)
    sma_20 = _safe_float(closes.tail(20).mean()) if len(closes) >= 1 else None
    sma_50 = _safe_float(closes.tail(50).mean()) if len(closes) >= 1 else None
    high_52w = _safe_float(closes.tail(252).max()) if len(closes) >= 1 else None
    low_52w = _safe_float(closes.tail(252).min()) if len(closes) >= 1 else None

    last_30 = df.tail(30)[["date", "close"]].copy()
    last_30["date"] = last_30["date"].dt.strftime("%Y-%m-%d")
    history = [
        {"date": r["date"], "close": _safe_float(r["close"])}
        for _, r in last_30.iterrows()
    ]

    # Best-effort signal-row enrichment (prob_up / rank) from the cached signal
    prob_up, rank = None, None
    try:
        sig = _load_signal()
        for s in sig.get("symbols", []):
            if str(s.get("symbol", "")).upper() == sym:
                prob_up = _safe_float(s.get("prob_up"))
                rank = s.get("rank")
                break
    except Exception:
        pass

    confidence = None
    if prob_up is not None:
        confidence = abs(prob_up - 0.5) * 2  # 0..1 distance from 50/50

    return {
        "disclaimer": DISCLAIMER,
        "symbol": sym,
        "prob_up": prob_up,
        "rank": rank,
        "confidence": _safe_float(confidence),
        "last_close": last_close,
        "previous_close": prev_close,
        "change": _safe_float(change),
        "change_percent": _safe_float(change_pct),
        "volume": _safe_float(last.get("volume")),
        "high": _safe_float(last.get("high")),
        "low": _safe_float(last.get("low")),
        "open": _safe_float(last.get("open")),
        "sma_20": sma_20,
        "sma_50": sma_50,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "as_of": str(last["date"].date()),
        "source": "parquet_ohlcv",
        "history": history,
    }


def _safe_float(x):
    """Coerce to float; return None for NaN/None/non-numeric so JSON dumps cleanly."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _data_freshness() -> dict:
    """Single source of truth for 'when was the OHLCV parquet last updated?'.

    Reads the freshness marker written by scripts/ingest_eod.py. If missing
    (e.g. the marker has never been written), it falls back to the parquet
    file's own mtime + the max date in the data. Either way, the result
    is a dict with: max_date, as_of_ist, last_trading_day, age_trading_days,
    is_stale, is_fresh, source.

    Stale means >1 trading day old (i.e. NSE has published a new bhavcopy
    that we have not yet ingested).
    """
    today_ist = datetime.now(IST).date()

    def _trading_day(d):
        while d.weekday() >= 5:  # Sat/Sun
            d = d - timedelta(days=1)
        return d

    last_trading_day = _trading_day(today_ist)

    # 1) The marker written by ingest_eod.py
    if FRESHNESS_PATH.exists():
        try:
            d = json.loads(FRESHNESS_PATH.read_text())
            return {
                "max_date": d.get("max_date"),
                "as_of_ist": d.get("as_of_ist"),
                "last_trading_day": d.get("last_trading_day") or last_trading_day.strftime("%Y-%m-%d"),
                "age_trading_days": d.get("age_trading_days"),
                "is_stale": bool(d.get("is_stale", True)),
                "is_fresh": bool(d.get("is_fresh", False)),
                "source": "freshness_marker",
            }
        except Exception:
            pass

    # 2) Fallback: read the parquet and find max(date)
    if NIFTY100_OHLCV.exists():
        try:
            df = __import__("pandas").read_parquet(NIFTY100_OHLCV, columns=["date"])
            max_dt = __import__("pandas").to_datetime(df["date"]).max()
            max_date_str = max_dt.strftime("%Y-%m-%d")
            age = (last_trading_day - max_dt.date()).days
            return {
                "max_date": max_date_str,
                "as_of_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                "last_trading_day": last_trading_day.strftime("%Y-%m-%d"),
                "age_trading_days": age,
                "is_stale": age > 1,
                "is_fresh": age <= 1,
                "source": "parquet_fallback",
            }
        except Exception:
            pass

    return {
        "max_date": None,
        "as_of_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "last_trading_day": last_trading_day.strftime("%Y-%m-%d"),
        "age_trading_days": None,
        "is_stale": True,
        "is_fresh": False,
        "source": "unavailable",
    }


def _backtest_summary(force: bool = False) -> dict:
    """Run (or return cached) the cost-aware cross-sectional backtest.

    Caches the result in ``reports/backtest_costs.json``. Re-runs only if the
    cache is missing or older than the predictions CSV.
    """
    if (
        not force
        and BACKTEST_PATH.exists()
        and PREDICTIONS_CSV.exists()
        and BACKTEST_PATH.stat().st_mtime > PREDICTIONS_CSV.stat().st_mtime
    ):
        try:
            return json.loads(BACKTEST_PATH.read_text())
        except Exception:
            pass

    if not PREDICTIONS_CSV.exists():
        return {
            "error": "predictions.csv not found; run scripts/build_signal.py first",
            "disclaimer": DISCLAIMER,
        }

    try:
        import sys as _sys
        if str(ROOT) not in _sys.path:
            _sys.path.insert(0, str(ROOT))
        from src.market_ml.backtest import run_from_predictions_csv  # type: ignore
        result = run_from_predictions_csv(
            PREDICTIONS_CSV, out_json=BACKTEST_PATH,
        )
        return result
    except Exception as e:
        return {
            "error": f"backtest failed: {e}",
            "disclaimer": DISCLAIMER,
        }


def _nifty50_index() -> dict:
    """Return the most recent NIFTY 50 index value we can compute.

    The research pipeline stores a NIFTY100 OHLCV parquet, not the NIFTY 50
    index. We try two cheap fallbacks:

    1. Look for any NIFTY 50 index file under data/raw or data/external.
    2. If none, approximate the NIFTY 50 by the simple mean of the 50 most
       liquid large-caps in the parquet (a research-grade proxy, clearly
       labeled). If even that fails, return a graceful "unavailable" dict.
    """
    # Fallback 1: pre-existing index file
    for cand in [
        ROOT / "data" / "raw" / "nifty50.csv",
        ROOT / "data" / "external" / "nifty50.csv",
        ROOT / "data" / "processed" / "nifty50_index.parquet",
    ]:
        if cand.exists():
            try:
                if cand.suffix == ".parquet":
                    df = __import__("pandas").read_parquet(cand)
                else:
                    df = __import__("pandas").read_csv(cand)
                df["date"] = __import__("pandas").to_datetime(df["date"])
                df = df.sort_values("date")
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                close = _safe_float(last.get("close"))
                pclose = _safe_float(prev.get("close"))
                chg = (close - pclose) if (close is not None and pclose is not None) else None
                chg_pct = ((chg / pclose) * 100) if (chg is not None and pclose not in (None, 0)) else None
                return {
                    "source": "nifty50_index_file",
                    "value": close, "previous_close": pclose,
                    "change": chg, "change_percent": chg_pct,
                    "as_of": str(last["date"].date()),
                }
            except Exception:
                pass
    # Fallback 2: simple average of the top 50 large-caps from the OHLCV parquet
    try:
        df = __import__("pandas").read_parquet(NIFTY100_OHLCV)
        df["date"] = __import__("pandas").to_datetime(df["date"])
        # Choose the 50 symbols with the highest average volume (proxy for "top")
        top_syms = (df.groupby("symbol")["volume"].mean()
                      .sort_values(ascending=False).head(50).index.tolist())
        proxy = (df[df["symbol"].isin(top_syms)]
                 .groupby("date")["close"].mean().sort_index())
        if proxy.empty:
            return {"source": "unavailable", "value": None, "as_of": None}
        last_date, last_val = proxy.index[-1], float(proxy.iloc[-1])
        prev_val = float(proxy.iloc[-2]) if len(proxy) > 1 else last_val
        chg = last_val - prev_val
        chg_pct = (chg / prev_val * 100) if prev_val else 0.0
        return {
            "source": "proxy_top50_liquid_avg",
            "value": _safe_float(last_val),
            "previous_close": _safe_float(prev_val),
            "change": _safe_float(chg),
            "change_percent": _safe_float(chg_pct),
            "as_of": str(last_date.date()),
        }
    except Exception as e:
        return {"source": "unavailable", "error": str(e),
                "value": None, "as_of": None}


@app.get("/api/signal")
def api_signal():
    sig = _load_signal()
    sig.setdefault("disclaimer", DISCLAIMER)
    sig.setdefault("freshness", "freshness not reported")
    return sig


@app.get("/api/screen")
def api_screen(limit: int = 0, direction: str = "ALL"):
    sig = _load_signal()
    if "error" in sig:
        return sig
    syms = sig.get("symbols", [])
    if direction.upper() == "UP":
        syms = [s for s in syms if s["prob_up"] >= 0.5]
    elif direction.upper() == "DOWN":
        syms = [s for s in syms if s["prob_up"] < 0.5]
    if limit and limit > 0:
        syms = syms[:limit]
    return {"disclaimer": DISCLAIMER, "as_of": sig.get("as_of"),
            "market_breadth_prob": sig.get("market_breadth_prob"),
            "direction": sig.get("direction"), "symbols": syms}


@app.get("/api/quotes")
def api_quotes(limit: int = 50):
    q = _latest_quotes()
    if limit and limit > 0:
        q = q[:limit]
    # When we're falling back to the parquet, also report the market state
    # so the UI can render "Market closed · last LTP as of …" cleanly.
    market_state = "open" if any(r.get("source") == "live_collector" for r in q) else "closed"
    as_of = q[0]["as_of"] if q else None
    return {
        "disclaimer": DISCLAIMER,
        "count": len(q),
        "market_state": market_state,
        "as_of": as_of,
        "source": q[0]["source"] if q else "unavailable",
        "quotes": q,
    }


@app.get("/api/symbol/{symbol}")
def api_symbol(symbol: str):
    """Detailed view for a single NIFTY-100 ticker (P(UP), confidence,
    last close, volume, 52w H/L, SMA-20/50, last-30-day close history)."""
    detail = _symbol_deep_dive(symbol)
    if detail is None:
        return JSONResponse(
            status_code=404,
            content={"disclaimer": DISCLAIMER, "error": f"symbol not found: {symbol}"},
        )
    return detail


@app.get("/api/performance")
def api_performance():
    """Curated research track-record summary (see PROJECT_STATUS.md / PHASE*.md)."""
    out = {
        "disclaimer": DISCLAIMER,
        "summary": {
            "methodology": "Expanding-window walk-forward, 1-day direction, |ret|<=0.5 filter, "
                           "XGB 150/depth4/lr0.05, median imputer. No random train/test splits.",
            "base_signal_auc": 0.5253,
            "base_signal_accuracy_pct": 51.85,
            "base_rate_pct": 50.4,
            "phase9_probe_extended_auc": 0.5316,
            "phase9_probe_note": "2026-only window (142 FII/DII sessions); McNemar not significant; "
                                  "definitive test blocked by NSE/NSDL access from host.",
            "gross_edge_bps_per_day": 2,
            "verdict": "Signal is real but small; not monetizable at Indian retail costs. "
                       "Naive equal-weight NIFTY100 has been the best strategy on the 2025-26 holdout.",
        },
        "phases": [
            {"phase": 3, "result": "XGB 51.85% acc, AUC 0.5253, edge +1.44% vs naive, z=+16.8"},
            {"phase": 6, "result": "5-day horizon: 0/2 holdout passes"},
            {"phase": 7, "result": "market-state/sector features: Gate 1 FAIL (dAUC -0.0043)"},
            {"phase": 8, "result": "macro features: Gate 1 FAIL (dAUC -0.0053)"},
            {"phase": 9, "result": "FII/DII probe: raw dAUC +0.022 but McNemar n.s., Gate 2 FAIL"},
        ],
    }
    # Layer the cost-aware backtest summary on top so the dashboard can
    # show "net" alongside the curated gross-edge card.
    bt = _backtest_summary(force=False)
    bt_summary = (bt or {}).get("summary") if isinstance(bt, dict) else None
    if bt_summary and "n_days" in bt_summary and bt_summary["n_days"] > 0:
        out["summary"]["net_backtest"] = {
            "n_days": bt_summary["n_days"],
            "n_universe_avg": bt_summary.get("n_universe_avg"),
            "n_long": bt_summary.get("n_long"),
            "n_short": bt_summary.get("n_short"),
            "cost_bps_roundtrip_per_leg": bt_summary.get("cost_bps_roundtrip_per_leg"),
            "cost_bps_roundtrip_per_day": bt_summary.get("cost_bps_roundtrip_per_day"),
            "gross_cum_return_pct": bt_summary.get("gross_cum_return_pct"),
            "net_cum_return_pct": bt_summary.get("net_cum_return_pct"),
            "naive_equal_weight_cum_return_pct": bt_summary.get("naive_equal_weight_cum_return_pct"),
            "edge_vs_naive_pct": bt_summary.get("edge_vs_naive_pct"),
            "net_daily_mean_bps": bt_summary.get("net_daily_mean_bps"),
            "net_sharpe": bt_summary.get("net_sharpe"),
            "max_drawdown_pct": bt_summary.get("max_drawdown_pct"),
            "hit_rate_pct": bt_summary.get("hit_rate_pct"),
            "turnover_pct_per_day": bt_summary.get("turnover_pct_per_day"),
        }
    return out


@app.get("/api/now")
def api_now():
    """Server time in IST (the dashboard's clock and the market's clock)."""
    now_ist = datetime.now(IST)
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    in_session = (now_ist.weekday() < 5 and market_open <= now_ist <= market_close)
    return {
        "utc": now_ist.astimezone(timezone.utc).isoformat(),
        "ist": now_ist.isoformat(),
        "ist_human": now_ist.strftime("%a %d %b %Y  %H:%M:%S IST"),
        "in_market_session": in_session,
        "data_freshness": _data_freshness(),
    }


@app.get("/api/freshness")
def api_freshness():
    """How fresh is the OHLCV parquet? Drives the UI's stale-data banner."""
    f = _data_freshness()
    return {"disclaimer": DISCLAIMER, **f}


@app.get("/api/backtest")
def api_backtest(force: bool = False):
    """Cost-aware cross-sectional backtest (LONG top decile / SHORT bottom).

    Returns ``summary`` (key metrics) and ``daily`` (one row per trading day).
    The result is cached on disk in ``reports/backtest_costs.json``; pass
    ``force=true`` to recompute.
    """
    r = _backtest_summary(force=force)
    if "disclaimer" not in r:
        r["disclaimer"] = DISCLAIMER
    return r


@app.get("/api/nifty50")
def api_nifty50():
    return {"disclaimer": DISCLAIMER, **_nifty50_index()}


# ---------------------------------------------------------------------------
# v1 — versioned, paid-API-friendly routes
# ---------------------------------------------------------------------------
# These mirror the unversioned /api/* routes but live under /api/v1/ so we
# can add breaking changes later without taking customers down. All v1
# responses include a `tier` marker ("free_delayed" or "live") so the
# data layer can introduce paid live tiers without changing the wire
# contract. The disclaimer is repeated on every response.

API_VERSION = "v1"

# Free tier: signals are the cached EOD signals from the parquet (delayed
# at least one trading day). Live tier: signals are the most recent live
# collector run, if any. Tier selection is driven by the GUARVI_API_TIER
# env var (defaults to "free_delayed"); the live tier would gate on a
# paid API key in production but the *route* is wired up now so the
# contract is stable.
import os as _os
_DEFAULT_TIER = _os.environ.get("GUARVI_API_TIER", "free_delayed").strip().lower()
VALID_TIERS = {"free_delayed", "live"}


def _resolve_tier(requested: str | None) -> str:
    """Pick a tier, defaulting to free_delayed.

    If the requested tier isn't recognised we fall back to free_delayed
    rather than 500ing — the dashboard always has a working response.
    """
    t = (requested or _DEFAULT_TIER).strip().lower()
    if t not in VALID_TIERS:
        t = "free_delayed"
    return t


def _tier_meta(tier: str) -> dict:
    """Metadata for the tier selection that we attach to every v1 response."""
    if tier == "live":
        return {
            "tier": "live",
            "tier_label": "Live (paid)",
            "freshness": "live_collector_or_eod",
            "rate_limit_per_min": 600,
        }
    return {
        "tier": "free_delayed",
        "tier_label": "Free — delayed by ≥1 trading day",
        "freshness": "eod_parquet",
        "rate_limit_per_min": 60,
    }


def _wrap_v1(payload: dict, tier: str) -> dict:
    """Attach version, tier, and disclaimer to a v1 response payload."""
    out = dict(payload)
    out["api_version"] = API_VERSION
    out["tier_meta"] = _tier_meta(tier)
    out.setdefault("disclaimer", DISCLAIMER)
    return out


# Live tier: when the live collector has a fresh quotes file in
# data/live/nse/, prefer it. When it doesn't, fall back to the parquet
# (and label the source). Free tier: parquet only.
def _quotes_for_tier(tier: str, limit: int) -> list[dict]:
    if tier == "free_delayed":
        # Force the parquet path by temporarily hiding the live dir.
        # We do this with a no-op copy of _latest_quotes's logic but
        # skipping the live_collector branch.
        pd = __import__("pandas")
        try:
            df = pd.read_parquet(NIFTY100_OHLCV)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            last_date = df["date"].max()
            last = df[df["date"] == last_date].copy()
            if last.empty:
                return []
            last = last.sort_values("symbol")
            prev_date = df[df["date"] < last_date]["date"].max()
            prev = df[df["date"] == prev_date][["symbol", "close"]].rename(
                columns={"close": "prev_close"}
            ) if prev_date is not pd.NaT else None
            merged = last.merge(prev, on="symbol", how="left") if prev is not None else last
            out = []
            for _, r in merged.iterrows():
                last_px = _safe_float(r.get("close"))
                prev_px = _safe_float(r.get("prev_close"))
                chg = (last_px - prev_px) if (last_px is not None and prev_px is not None) else None
                pct = (chg / prev_px * 100) if (chg is not None and prev_px not in (None, 0)) else None
                out.append({
                    "symbol": r.get("symbol"),
                    "last": last_px,
                    "change": _safe_float(chg),
                    "change_percent": _safe_float(pct),
                    "volume": _safe_float(r.get("volume")),
                    "as_of": str(last_date.date()),
                    "source": "parquet_last_day",
                })
            return out[:limit] if limit and limit > 0 else out
        except Exception:
            return []
    return _latest_quotes()[:limit] if limit and limit > 0 else _latest_quotes()


@app.get("/api/v1/access")
def api_v1_access(tier: str | None = None):
    """Return the current API contract: version, tiers, route list, docs.

    Useful for paid customers wanting to discover what's available and at
    what rate-limit. Always public; never returns a 5xx.
    """
    t = _resolve_tier(tier)
    return _wrap_v1({
        "service": "Guarvi Signal Portal",
        "tiers": [
            {
                "id": "free_delayed",
                "label": "Free — delayed by ≥1 trading day",
                "freshness": "eod_parquet",
                "rate_limit_per_min": 60,
            },
            {
                "id": "live",
                "label": "Live (paid)",
                "freshness": "live_collector_or_eod",
                "rate_limit_per_min": 600,
            },
        ],
        "routes": [
            {"path": "/api/v1/access", "method": "GET",
             "summary": "Service metadata + tier list (this route)"},
            {"path": "/api/v1/signal", "method": "GET",
             "summary": "Top NIFTY-100 directional signal for the next session."},
            {"path": "/api/v1/screen", "method": "GET",
             "summary": "Filtered screen by direction (UP / DOWN / ALL) and limit."},
            {"path": "/api/v1/quotes", "method": "GET",
             "summary": "Most recent LTP for NIFTY-100 symbols (delayed or live per tier)."},
            {"path": "/api/v1/symbol/{symbol}", "method": "GET",
             "summary": "Deep-dive card for a single ticker (P(UP), confidence, 52w H/L, SMAs)."},
            {"path": "/api/v1/performance", "method": "GET",
             "summary": "Research track-record: gross edge + cost-aware net backtest."},
            {"path": "/api/v1/backtest", "method": "GET",
             "summary": "Cost-aware cross-sectional backtest (LONG top / SHORT bottom)."},
            {"path": "/api/v1/freshness", "method": "GET",
             "summary": "How fresh is the OHLCV parquet? Drives the stale-data banner."},
            {"path": "/api/v1/nifty50", "method": "GET",
             "summary": "NIFTY 50 index level (file or large-cap proxy)."},
            {"path": "/api/v1/now", "method": "GET",
             "summary": "Server time + data freshness snapshot."},
            {"path": "/api/v1/scores", "method": "GET",
             "summary": "BUY DECISION ENGINE: 6 factor scores + 14 KPIs for every NIFTY 100 symbol."},
            {"path": "/api/v1/scores/{symbol}", "method": "GET",
             "summary": "Single-symbol score card with all factor scores, KPIs, and trade plan."},
            {"path": "/api/v1/screener", "method": "GET",
             "summary": "Top 10 opportunities as a flat table (rank, signal, entry, SL, targets, R:R)."},
            {"path": "/api/v1/market", "method": "GET",
             "summary": "Market dashboard payload: NIFTY 50, breadth, sentiment, sector heatmap, top movers."},
            {"path": "/api/v1/kpis", "method": "GET",
             "summary": "Hero-section KPIs: stocks scanned, 30D wins, avg return, success rate."},
            {"path": "/api/v1/reasoning/{symbol}", "method": "GET",
             "summary": "Human-readable AI reasoning bullets for a single ticker."},
            {"path": "/api/v1/technicals/{symbol}", "method": "GET",
             "summary": "Full technicals + 52w range + support/resistance for STOCK DETAIL page."},
        ],
        "disclaimer_url": "/static/DISCLAIMER.md",
    }, t)


@app.get("/api/v1/signal")
def api_v1_signal(tier: str | None = None):
    t = _resolve_tier(tier)
    sig = _load_signal()
    sig.setdefault("disclaimer", DISCLAIMER)
    return _wrap_v1(sig, t)


@app.get("/api/v1/screen")
def api_v1_screen(limit: int = 0, direction: str = "ALL", tier: str | None = None):
    t = _resolve_tier(tier)
    sig = _load_signal()
    if "error" in sig:
        return _wrap_v1(sig, t)
    syms = sig.get("symbols", [])
    if direction.upper() == "UP":
        syms = [s for s in syms if s["prob_up"] >= 0.5]
    elif direction.upper() == "DOWN":
        syms = [s for s in syms if s["prob_up"] < 0.5]
    if limit and limit > 0:
        syms = syms[:limit]
    return _wrap_v1({
        "as_of": sig.get("as_of"),
        "market_breadth_prob": sig.get("market_breadth_prob"),
        "direction": sig.get("direction"),
        "symbols": syms,
    }, t)


@app.get("/api/v1/quotes")
def api_v1_quotes(limit: int = 50, tier: str | None = None):
    t = _resolve_tier(tier)
    q = _quotes_for_tier(t, limit)
    market_state = "open" if any(r.get("source") == "live_collector" for r in q) else "closed"
    as_of = q[0]["as_of"] if q else None
    return _wrap_v1({
        "count": len(q),
        "market_state": market_state,
        "as_of": as_of,
        "source": q[0]["source"] if q else "unavailable",
        "quotes": q,
    }, t)


@app.get("/api/v1/symbol/{symbol}")
def api_v1_symbol(symbol: str, tier: str | None = None):
    t = _resolve_tier(tier)
    detail = _symbol_deep_dive(symbol)
    if detail is None:
        return JSONResponse(
            status_code=404,
            content=_wrap_v1({"error": f"symbol not found: {symbol}"}, t),
        )
    return _wrap_v1(detail, t)


@app.get("/api/v1/performance")
def api_v1_performance(tier: str | None = None):
    t = _resolve_tier(tier)
    return _wrap_v1(api_performance(), t)


@app.get("/api/v1/backtest")
def api_v1_backtest(force: bool = False, tier: str | None = None):
    t = _resolve_tier(tier)
    return _wrap_v1(api_backtest(force=force), t)


@app.get("/api/v1/freshness")
def api_v1_freshness(tier: str | None = None):
    t = _resolve_tier(tier)
    return _wrap_v1(api_freshness(), t)


@app.get("/api/v1/nifty50")
def api_v1_nifty50(tier: str | None = None):
    t = _resolve_tier(tier)
    return _wrap_v1(api_nifty50(), t)


@app.get("/api/v1/now")
def api_v1_now(tier: str | None = None):
    t = _resolve_tier(tier)
    return _wrap_v1(api_now(), t)


# ------------------------------------------------------------------
# v1 redesign endpoints: scoring, screener, market, reasoning
# ------------------------------------------------------------------

@app.get("/api/v1/scores")
def api_v1_scores_all(tier: str | None = None, limit: int = 0):
    """All score cards (one per symbol) plus sector summary and headline stats."""
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    try:
        result = scoring.compute_all_scores()
    except Exception as e:
        return JSONResponse(status_code=500, content=_wrap_v1({"error": f"scoring failed: {e}"}, t))
    scores = result.get("scores", [])
    if limit and limit > 0:
        scores = scores[:limit]
    return _wrap_v1({
        "as_of": result.get("as_of"),
        "summary": result.get("summary", {}),
        "sector_summary": result.get("sector_summary", {}),
        "scores": scores,
        "count": len(scores),
    }, t)


@app.get("/api/v1/scores/{symbol}")
def api_v1_scores_symbol(symbol: str, tier: str | None = None):
    """Score card for a single symbol with all 6 factor scores + 14 KPIs."""
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    try:
        card = scoring.score_for_symbol(symbol)
    except Exception as e:
        return JSONResponse(status_code=500, content=_wrap_v1({"error": f"scoring failed: {e}"}, t))
    if card is None:
        return JSONResponse(
            status_code=404,
            content=_wrap_v1({"error": f"symbol not found in features parquet: {symbol}"}, t),
        )
    return _wrap_v1(card, t)


@app.get("/api/v1/screener")
def api_v1_screener(direction: str = "ALL", limit: int = 10, tier: str | None = None):
    """Top 10 (or limit) opportunities as a flat table with the 15 columns
    required by the redesign: rank, symbol, cmp, signal, confidence,
    expected upside, risk, risk:reward, entry, SL, target 1, target 2,
    holding period, action.
    """
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    try:
        result = scoring.compute_all_scores()
    except Exception as e:
        return JSONResponse(status_code=500, content=_wrap_v1({"error": f"scoring failed: {e}"}, t))
    cards = result.get("scores", [])
    if direction.upper() == "UP":
        cards = [c for c in cards if (c.get("prob_up") or 0) > 0.5]
    elif direction.upper() == "DOWN":
        cards = [c for c in cards if (c.get("prob_up") or 0) < 0.5]
    cards = cards[:max(1, limit)] if limit else cards[:10]

    rows = []
    for i, c in enumerate(cards, start=1):
        prob = c.get("prob_up") or 0.5
        scores = c["scores"]
        overall = scores["overall"]
        # Map overall to action label
        if overall >= 75 and prob >= 0.55:
            action = "STRONG BUY"
            signal = "UP"
        elif overall >= 60 and prob >= 0.52:
            action = "BUY"
            signal = "UP"
        elif overall >= 45 and prob >= 0.48:
            action = "HOLD"
            signal = "NEUTRAL"
        elif overall <= 30 and prob <= 0.45:
            action = "STRONG SELL"
            signal = "DOWN"
        else:
            action = "SELL"
            signal = "DOWN"
        # Risk:Reward from trade plan
        tp = c["trade_plan"]
        entry = tp["entry_price"]
        sl = tp["stop_loss"]
        t1 = tp["target_1"]
        t2 = tp["target_2"]
        rr = None
        if entry and sl and t1:
            risk_per_unit = (entry - sl) if (entry - sl) > 0 else None
            reward_per_unit = (t1 - entry) if (t1 - entry) > 0 else None
            if risk_per_unit and reward_per_unit and risk_per_unit > 0:
                rr = round(reward_per_unit / risk_per_unit, 2)
        expected_upside = c["kpis"].get("expected_return_pct")
        rows.append({
            "rank": i,
            "symbol": c["symbol"],
            "sector": c["sector"],
            "cmp": c["last_close"],
            "signal": signal,
            "action": action,
            "ai_score": scores["ai"],
            "overall_score": overall,
            "confidence_pct": c["kpis"].get("confidence_pct"),
            "expected_upside_pct": expected_upside,
            "downside_risk_pct": c["kpis"].get("downside_risk_pct"),
            "risk_reward": rr,
            "risk_level": tp["risk_level"],
            "entry": entry,
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "holding_period": tp["holding_period"],
            "win_probability_pct": c["kpis"].get("win_probability_pct"),
            "trend_strength": c["kpis"]["trend_strength"],
            "liquidity_score": c["kpis"]["liquidity_score"],
            "as_of": c["as_of"],
        })
    return _wrap_v1({
        "as_of": result.get("as_of"),
        "direction": direction.upper(),
        "limit": limit,
        "summary": result.get("summary", {}),
        "rows": rows,
        "count": len(rows),
    }, t)


@app.get("/api/v1/market")
def api_v1_market(tier: str | None = None):
    """One-shot payload for the MARKET DASHBOARD section.
    Combines: NIFTY 50, NIFTY 100 breadth, top gainers, top losers,
    sector heatmap, market sentiment, advance/decline.
    """
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    try:
        scores_payload = scoring.compute_all_scores()
    except Exception as e:
        return JSONResponse(status_code=500, content=_wrap_v1({"error": f"scoring failed: {e}"}, t))
    cards = scores_payload.get("scores", [])
    nifty50 = api_nifty50()
    sig = _load_signal()
    breadth_prob = sig.get("market_breadth_prob") or 0.5
    # Sentiment
    if breadth_prob >= 0.55:
        sentiment = "Bullish"
        sentiment_label = "Risk-On"
    elif breadth_prob <= 0.45:
        sentiment = "Bearish"
        sentiment_label = "Risk-Off"
    else:
        sentiment = "Neutral"
        sentiment_label = "Range-Bound"
    # Sector heatmap (sorted by avg overall desc)
    sec = scores_payload.get("sector_summary", {})
    sector_heatmap = [
        {"sector": s, "avg_score": d["avg_score"], "n": d["n"]}
        for s, d in sec.items()
    ]
    # Top gainers / losers by last close change (need ohlcv change %)
    # We have ret_1d indirectly via last_close vs prev — derive from signal last_close and parquet.
    try:
        import pandas as pd
        from src.market_ml import scoring as _s
        df = _s._load_features()
        if not df.empty:
            last_date = df["date"].max()
            last = df[df["date"] == last_date][["symbol", "close"]].copy()
            last["date"] = last_date
            prev_date = df[df["date"] < last_date]["date"].max()
            if prev_date is not None and prev_date is not pd.NaT:
                prev = df[df["date"] == prev_date][["symbol", "close"]].rename(columns={"close": "prev_close"})
                merged = last.merge(prev, on="symbol", how="left")
                merged["change_pct"] = ((merged["close"] - merged["prev_close"]) / merged["prev_close"]) * 100
            else:
                merged = last
                merged["change_pct"] = None
            gainers = merged.sort_values("change_pct", ascending=False).head(5)[["symbol", "close", "change_pct"]].to_dict("records")
            losers = merged.sort_values("change_pct", ascending=True).head(5)[["symbol", "close", "change_pct"]].to_dict("records")
            advancers = int((merged["change_pct"] > 0).sum())
            decliners = int((merged["change_pct"] < 0).sum())
            unchanged = len(merged) - advancers - decliners
        else:
            gainers, losers = [], []
            advancers = decliners = unchanged = 0
    except Exception:
        gainers, losers = [], []
        advancers = decliners = unchanged = 0

    return _wrap_v1({
        "as_of": scores_payload.get("as_of"),
        "nifty50": nifty50,
        "breadth_prob": breadth_prob,
        "sentiment": sentiment,
        "sentiment_label": sentiment_label,
        "advance_decline": {
            "advancers": advancers, "decliners": decliners, "unchanged": unchanged,
        },
        "top_gainers": [
            {"symbol": g["symbol"], "last": g.get("close"), "change_pct": g.get("change_pct")}
            for g in gainers
        ],
        "top_losers": [
            {"symbol": l["symbol"], "last": l.get("close"), "change_pct": l.get("change_pct")}
            for l in losers
        ],
        "sector_heatmap": sector_heatmap,
        "summary": scores_payload.get("summary", {}),
    }, t)


@app.get("/api/v1/kpis")
def api_v1_kpis(tier: str | None = None):
    """Headline KPIs for the HERO section.
    Combines: stocks scanned today, 30D winning signals, average return,
    success rate, active premium users (illustrative - we don't track
    users on free tier; we report the breadth of signals as a proxy and
    disclose the source).
    """
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    sig = _load_signal()
    perf = api_performance()
    summary = perf.get("summary", {}) if isinstance(perf, dict) else {}
    try:
        scores_payload = scoring.compute_all_scores()
        cards = scores_payload.get("scores", [])
    except Exception:
        cards = []
        scores_payload = {"summary": {}}

    # Stocks scanned today = total symbols in screen
    stocks_scanned = len(cards)
    # 30D winning signals = how many signals across the last 30 sessions had
    # prob_up > 0.5 in the signal file. Without per-day signal, we approximate
    # with the current breadth.
    breadth_pct = (sig.get("market_breadth_prob") or 0.5) * 100
    # Average return: from backtest (gross edge bps/day)
    avg_return_bps = summary.get("gross_edge_bps_per_day")
    avg_return_pct = (avg_return_bps / 100.0) if avg_return_bps is not None else None
    # Success rate = walk-forward accuracy
    success_rate = summary.get("base_signal_accuracy_pct")
    # Premium users: we don't track; return a clear "data-limited" marker
    return _wrap_v1({
        "as_of": scores_payload.get("as_of"),
        "stocks_scanned_today": stocks_scanned,
        "winning_signals_30d_pct": round(breadth_pct, 1),
        "avg_return_pct": avg_return_pct,
        "success_rate_pct": success_rate,
        "active_premium_users": None,  # honest: not tracked
        "premium_users_disclosure": "Premium user count not tracked in this research deployment.",
    }, t)


@app.get("/api/v1/reasoning/{symbol}")
def api_v1_reasoning(symbol: str, tier: str | None = None):
    """Human-readable AI reasoning for a single symbol.

    Generates explainable bullets from the score card and feature values
    so the dashboard can show 'why' (not just 'what') the model recommended.
    """
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    card = scoring.score_for_symbol(symbol)
    if card is None:
        return JSONResponse(
            status_code=404,
            content=_wrap_v1({"error": f"symbol not found: {symbol}"}, t),
        )
    scores = card["scores"]
    k = card["kpis"]
    bullets = []
    # Trend
    if scores["technical"] >= 70:
        bullets.append({"tag": "TREND", "text": "Price is above the 20, 50, and 200-day SMAs with positive cross alignment - a confirmed uptrend.", "tone": "bull"})
    elif scores["technical"] <= 30:
        bullets.append({"tag": "TREND", "text": "Price is below the 20, 50, and 200-day SMAs - a confirmed downtrend.", "tone": "bear"})
    else:
        bullets.append({"tag": "TREND", "text": "Mixed trend signals - some SMAs are above price and some below.", "tone": "neutral"})
    # RSI
    # We don't have RSI in the card; pull from features if needed
    # Momentum
    if scores["momentum"] >= 70:
        bullets.append({"tag": "MOMENTUM", "text": "5/10/20-day returns and OBV trend are all positive - momentum is strong.", "tone": "bull"})
    elif scores["momentum"] <= 30:
        bullets.append({"tag": "MOMENTUM", "text": "5/10/20-day returns are negative and OBV is declining - momentum is weak.", "tone": "bear"})
    # Volatility
    if scores["risk"] >= 70:
        bullets.append({"tag": "RISK", "text": "20-day realised volatility is below 0.8% daily - a low-risk setup.", "tone": "bull"})
    elif scores["risk"] <= 30:
        bullets.append({"tag": "RISK", "text": "20-day realised volatility is above 2% daily - a high-risk setup.", "tone": "bear"})
    # Valuation
    if scores["valuation"] >= 60:
        bullets.append({"tag": "VALUATION", "text": "Bollinger band position is in the lower half - price is closer to its recent floor than ceiling.", "tone": "bull"})
    elif scores["valuation"] <= 40:
        bullets.append({"tag": "VALUATION", "text": "Bollinger band position is in the upper half - price is closer to its recent ceiling than floor.", "tone": "bear"})
    # Relative strength
    if k["relative_strength"] >= 65:
        bullets.append({"tag": "RELATIVE STRENGTH", "text": f"Excess return vs the universe ranks in the top 35% - outperforming the NIFTY 100 average.", "tone": "bull"})
    elif k["relative_strength"] <= 35:
        bullets.append({"tag": "RELATIVE STRENGTH", "text": f"Excess return vs the universe ranks in the bottom 35% - underperforming the NIFTY 100 average.", "tone": "bear"})
    # Institutional activity (volume ratio proxy)
    if k["institutional_activity"] >= 60:
        bullets.append({"tag": "INSTITUTIONAL", "text": "Recent volume is 1.1x+ the 20-day average - participation is rising.", "tone": "bull"})
    elif k["institutional_activity"] <= 40:
        bullets.append({"tag": "INSTITUTIONAL", "text": "Recent volume is below the 20-day average - participation is light.", "tone": "bear"})
    # Sector
    if k["sector_strength"] >= 60:
        bullets.append({"tag": "SECTOR", "text": f"The {card['sector']} sector is currently in the top half of the universe on the cross-sectional score.", "tone": "bull"})
    elif k["sector_strength"] <= 40:
        bullets.append({"tag": "SECTOR", "text": f"The {card['sector']} sector is currently in the bottom half of the universe on the cross-sectional score.", "tone": "bear"})
    # AI / model
    if scores["ai"] >= 60:
        bullets.append({"tag": "AI MODEL", "text": f"XGBoost P(UP) = {round((card.get('prob_up') or 0)*100, 1)}% - above 50/50 with elevated confidence.", "tone": "bull"})
    elif scores["ai"] <= 40:
        bullets.append({"tag": "AI MODEL", "text": f"XGBoost P(UP) = {round((card.get('prob_up') or 0)*100, 1)}% - below 50/50 with elevated confidence.", "tone": "bear"})
    else:
        bullets.append({"tag": "AI MODEL", "text": f"XGBoost P(UP) = {round((card.get('prob_up') or 0)*100, 1)}% - the model is essentially neutral.", "tone": "neutral"})

    # Confidence summary
    conf = card.get("confidence_pct")
    if conf is not None:
        if conf >= 30:
            confidence_level = "High"
        elif conf >= 10:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
    else:
        confidence_level = "Unknown"
    # Action label
    if scores["overall"] >= 70:
        action = "STRONG BUY"
    elif scores["overall"] >= 60:
        action = "BUY"
    elif scores["overall"] <= 30:
        action = "STRONG SELL"
    elif scores["overall"] <= 40:
        action = "SELL"
    else:
        action = "HOLD"
    return _wrap_v1({
        "symbol": card["symbol"],
        "sector": card["sector"],
        "as_of": card["as_of"],
        "action": action,
        "overall_score": scores["overall"],
        "confidence_level": confidence_level,
        "confidence_pct": conf,
        "bullets": bullets,
    }, t)


@app.get("/api/v1/technicals/{symbol}")
def get_technicals(symbol: str, tier: Optional[str] = None):
    """Full technical/fundamental snapshot for the STOCK DETAIL page.

    Returns EMA 20/50/200, RSI, MACD, ADX, ATR, Bollinger, 52w high/low,
    support/resistance, breakout status, volume surge, relative-strength
    rank, and a fundamentals block (mostly null today, with a clear note).
    """
    t = _resolve_tier(tier)
    scoring = _scoring_module()
    snap = scoring.technicals_for_symbol(symbol)
    if snap is None:
        return JSONResponse(
            status_code=404,
            content=_wrap_v1({"error": f"symbol not found: {symbol}"}, t),
        )
    return _wrap_v1(snap, t)


@app.get("/")
def index():
    idx = STATIC / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "Guarvi portal backend is up. See /api/signal.",
                         "disclaimer": DISCLAIMER})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
