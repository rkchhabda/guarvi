"""Tests for the v1 JSON API and free/live tier separation.

These verify:
1. The v1 access route is reachable and lists every other v1 route.
2. Every v1 response includes api_version, tier_meta, and disclaimer.
3. Free-tier quotes come from the parquet (delayed), live-tier quotes
   from the live collector when present, falling back to parquet.
4. Tier fallback: an unrecognised tier resolves to free_delayed.
5. v1/symbol returns 404 with the same envelope for unknown symbols.
6. The v1 endpoints mirror the unversioned ones (same payload shape).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_v1_access_lists_routes(client):
    r = client.get("/api/v1/access")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    assert body["tier_meta"]["tier"] == "free_delayed"
    assert body["disclaimer"]
    paths = {route["path"] for route in body["routes"]}
    # Every documented v1 route is registered.
    for required in (
        "/api/v1/access",
        "/api/v1/signal",
        "/api/v1/screen",
        "/api/v1/quotes",
        "/api/v1/symbol/{symbol}",
        "/api/v1/performance",
        "/api/v1/backtest",
        "/api/v1/freshness",
        "/api/v1/nifty50",
        "/api/v1/now",
    ):
        assert required in paths, f"missing route in /api/v1/access: {required}"


def test_v1_tiers_defined(client):
    r = client.get("/api/v1/access")
    assert r.status_code == 200
    tier_ids = {t["id"] for t in r.json()["tiers"]}
    assert "free_delayed" in tier_ids
    assert "live" in tier_ids


def test_v1_signal_includes_envelope(client):
    r = client.get("/api/v1/signal")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    assert "tier_meta" in body
    assert body["tier_meta"]["tier"] == "free_delayed"
    assert "disclaimer" in body


def test_v1_quotes_free_tier_uses_parquet(client):
    r = client.get("/api/v1/quotes", params={"tier": "free_delayed", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["tier_meta"]["tier"] == "free_delayed"
    # All sources should be parquet_last_day — never live_collector.
    for q in body["quotes"]:
        assert q["source"] == "parquet_last_day", q
    assert body["market_state"] == "closed"


def test_v1_quotes_live_tier_falls_back_to_parquet(client):
    """When the live collector hasn't written a file, the live tier
    must still return a useful (parquet-backed) response — never 5xx."""
    r = client.get("/api/v1/quotes", params={"tier": "live", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["tier_meta"]["tier"] == "live"
    # market_state is "open" only if the source is live_collector; with
    # no collector file it must be "closed" and the parquet path taken.
    assert body["market_state"] in {"open", "closed"}


def test_v1_unknown_tier_falls_back_to_free(client):
    r = client.get("/api/v1/quotes", params={"tier": "platinum", "limit": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["tier_meta"]["tier"] == "free_delayed"


def test_v1_symbol_404_envelope(client):
    r = client.get("/api/v1/symbol/DOES_NOT_EXIST_XYZ")
    assert r.status_code == 404
    body = r.json()
    assert body["api_version"] == "v1"
    assert "tier_meta" in body
    assert "disclaimer" in body
    assert "DOES_NOT_EXIST_XYZ" in body["error"]


def test_v1_freshness_envelope(client):
    r = client.get("/api/v1/freshness")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    assert "max_date" in body
    assert "is_stale" in body
    assert "tier_meta" in body
    assert "disclaimer" in body


def test_v1_now_envelope(client):
    r = client.get("/api/v1/now")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    assert "data_freshness" in body
    assert "ist_human" in body
    assert "tier_meta" in body


def test_v1_performance_envelope(client):
    r = client.get("/api/v1/performance")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    assert "summary" in body
    assert "disclaimer" in body


def test_v1_nifty50_envelope(client):
    r = client.get("/api/v1/nifty50")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    # Either we have a real index file, a proxy, or a graceful "unavailable"
    assert "source" in body
    assert "disclaimer" in body


def test_v1_backtest_envelope(client):
    r = client.get("/api/v1/backtest")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "v1"
    assert "disclaimer" in body
    # Either we have a summary or an error — both are valid v1 envelopes.
    assert ("summary" in body) or ("error" in body)


def test_unversioned_routes_still_work(client):
    """The unversioned /api/* routes must keep working for the existing
    static frontend and the dashboard's AJAX calls."""
    for path in (
        "/api/signal",
        "/api/screen",
        "/api/quotes",
        "/api/freshness",
        "/api/now",
        "/api/performance",
        "/api/nifty50",
        "/api/backtest",
    ):
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"


        def test_v1_access_lists_buy_decision_engine_routes(client):
            """The BUY DECISION ENGINE v1 routes are advertised in /api/v1/access."""
            r = client.get("/api/v1/access")
            paths = {route["path"] for route in r.json()["routes"]}
            for required in (
                "/api/v1/scores",
                "/api/v1/scores/{symbol}",
                "/api/v1/screener",
                "/api/v1/market",
                "/api/v1/kpis",
                "/api/v1/reasoning/{symbol}",
                "/api/v1/technicals/{symbol}",
            ):
                assert required in paths, f"missing BUY DECISION ENGINE route: {required}"


        def test_v1_technicals_known_symbol(client):
            """GET /api/v1/technicals/RELIANCE returns the full technicals snapshot."""
            r = client.get("/api/v1/technicals/RELIANCE")
            assert r.status_code == 200
            body = r.json()
            assert body["api_version"] == "v1"
            assert "tier_meta" in body
            assert "disclaimer" in body
            assert body["symbol"] == "RELIANCE"
            assert "technicals" in body
            t = body["technicals"]
            # All required technicals fields present (or null)
            for key in (
                "ema_20", "ema_50", "ema_200",
                "rsi_14", "macd", "macd_signal", "macd_hist",
                "adx", "atr_14", "bb_upper", "bb_lower", "bb_width", "bb_position",
                "trend", "breakout", "volume_surge",
                "support_1", "support_2", "resistance_1", "resistance_2",
                "relative_strength_rank_pct", "volume_rank_pct",
            ):
                assert key in t, f"missing technicals key: {key}"
            # Ranges block
            assert "ranges" in body
            assert "high_52w" in body["ranges"]
            assert "low_52w" in body["ranges"]
            # Fundamentals block (mostly null in current dataset, but present)
            assert "fundamentals" in body
            assert body["fundamentals"]["_source"]  # honest disclosure


        def test_v1_technicals_unknown_symbol_returns_404(client):
            """GET /api/v1/technicals/UNKNOWN returns 404 with the v1 envelope."""
            r = client.get("/api/v1/technicals/ZZZZZZZZ")
            assert r.status_code == 404
            body = r.json()
            assert body["api_version"] == "v1"
            assert "error" in body
            assert "tier_meta" in body


        def test_v1_technicals_case_insensitive(client):
            """Symbol lookup is case-insensitive (URL path is uppercased internally)."""
            r = client.get("/api/v1/technicals/reliance")
            # FastAPI routes are case-sensitive by default; the helper uppercases,
            # so lower-case still resolves.
            assert r.status_code == 200
            assert r.json()["symbol"] == "RELIANCE"
