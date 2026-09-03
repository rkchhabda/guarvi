"""Unit tests for scripts/live_data_runner.py (no network / no API needed)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import live_data_runner as ldr  # noqa: E402

TZ = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# parse_hhmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("09:15", (9, 15)),
    ("15:30", (15, 30)),
    ("00:00", (0, 0)),
    ("23:59", (23, 59)),
])
def test_parse_hhmm_valid(value, expected):
    assert ldr.parse_hhmm(value) == expected


@pytest.mark.parametrize("value", ["25:00", "12:60", "abc", "12", "", "1x:30"])
def test_parse_hhmm_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        ldr.parse_hhmm(value)


# ---------------------------------------------------------------------------
# next_session
# ---------------------------------------------------------------------------

def sess(now):
    return ldr.next_session(now, (9, 15), (15, 30))


def test_next_session_mid_session():
    now = datetime(2026, 8, 31, 10, 0, tzinfo=TZ)  # Monday
    start, end = sess(now)
    assert (start, end) == (
        datetime(2026, 8, 31, 9, 15, tzinfo=TZ),
        datetime(2026, 8, 31, 15, 30, tzinfo=TZ),
    )


def test_next_session_before_open_same_day():
    now = datetime(2026, 8, 31, 8, 0, tzinfo=TZ)  # Monday, pre-open
    start, _ = sess(now)
    assert start == datetime(2026, 8, 31, 9, 15, tzinfo=TZ)


def test_next_session_after_close_next_weekday():
    now = datetime(2026, 8, 31, 16, 0, tzinfo=TZ)  # Monday after close
    start, _ = sess(now)
    assert start == datetime(2026, 9, 1, 9, 15, tzinfo=TZ)  # Tuesday


def test_next_session_friday_after_close_skips_weekend():
    now = datetime(2026, 9, 4, 16, 0, tzinfo=TZ)  # Friday after close
    start, _ = sess(now)
    assert start == datetime(2026, 9, 7, 9, 15, tzinfo=TZ)  # Monday


def test_next_session_weekend():
    now = datetime(2026, 8, 29, 12, 0, tzinfo=TZ)  # Saturday
    start, _ = sess(now)
    assert start == datetime(2026, 8, 31, 9, 15, tzinfo=TZ)  # Monday


# ---------------------------------------------------------------------------
# load_symbols
# ---------------------------------------------------------------------------

def test_load_symbols_from_cli():
    assert ldr.load_symbols("ignored.yaml", [" reliance ", "TCS"], None) == \
        ["RELIANCE", "TCS"]


def test_load_symbols_from_universe(tmp_path):
    symbols_csv = tmp_path / "symbols.csv"
    symbols_csv.write_text("symbol\nRELIANCE\nTCS\n\nINFY\n", encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(
        {"symbols_file": str(symbols_csv)}), encoding="utf-8")
    assert ldr.load_symbols(str(config), None, "nifty100") == \
        ["RELIANCE", "TCS", "INFY"]


def test_load_symbols_missing_file(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(
        {"symbols_file": str(tmp_path / "nope.csv")}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ldr.load_symbols(str(config), None, "nifty100")


# ---------------------------------------------------------------------------
# append_csv
# ---------------------------------------------------------------------------

def test_append_csv_dedupes_and_keeps_newest(tmp_path):
    path = tmp_path / "out.csv"
    df1 = pd.DataFrame({"timestamp": ["t1", "t2"], "px": [1, 2]})
    df2 = pd.DataFrame({"timestamp": ["t2", "t3"], "px": [99, 3]})  # t2 updated
    assert ldr.append_csv(df1, path, ["timestamp"]) == 2
    assert ldr.append_csv(df2, path, ["timestamp"]) == 3
    out = pd.read_csv(path)
    assert list(out["timestamp"]) == ["t1", "t2", "t3"]
    assert out.loc[out["timestamp"] == "t2", "px"].item() == 99


def test_append_csv_recreates_corrupt_file(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("not,a,valid\n,,csv", encoding="utf-8")
    df = pd.DataFrame({"timestamp": ["t1"], "px": [1]})
    assert ldr.append_csv(df, path, ["timestamp"]) == 1


# ---------------------------------------------------------------------------
# fetch_cycle with a fake loader (no network)
# ---------------------------------------------------------------------------

class FakeLoader:
    """Mimics GrowwLiveLoader for fetch_cycle."""

    def __init__(self, quotes, candles):
        self._quotes, self._candles = quotes, candles
        self.quote_calls, self.candle_calls = [], []

    def get_multiple_quotes(self, symbols):
        self.quote_calls.append(list(symbols))
        return {s: self._quotes.get(s) for s in symbols if self._quotes.get(s)}

    def get_live_ohlcv(self, symbol, interval="1minute"):
        self.candle_calls.append((symbol, interval))
        return self._candles.get(symbol)


def make_args(**over):
    base = dict(timezone="Asia/Kolkata", no_quotes=False, no_candles=False,
                candle_interval="5minute")
    base.update(over)
    return argparse.Namespace(**base)


def test_fetch_cycle_writes_quotes_and_candles(tmp_path):
    today = datetime.now(TZ).strftime("%Y%m%d")
    quotes = {"RELIANCE": {"symbol": "RELIANCE", "timestamp": "2026-08-31 10:00:00",
                           "last_price": 100.0, "volume": 5}}
    candles = pd.DataFrame({"timestamp": ["09:15", "09:20"], "close": [1.0, 1.1]})
    loader = FakeLoader(quotes, {"RELIANCE": candles})
    live_dir = tmp_path / "live"

    ldr.fetch_cycle(loader, ["RELIANCE"], live_dir, make_args())

    qfile = live_dir / f"quotes_{today}.csv"
    cfile = live_dir / "ohlcv" / f"RELIANCE_{today}.csv"
    assert qfile.exists() and cfile.exists()
    qdf = pd.read_csv(qfile)
    assert qdf.loc[0, "symbol"] == "RELIANCE"
    assert qdf.loc[0, "last_price"] == 100.0
    assert pd.read_csv(cfile).shape[0] == 2
    assert loader.candle_calls == [("RELIANCE", "5minute")]


def test_fetch_cycle_repeat_run_does_not_duplicate_rows(tmp_path):
    today = datetime.now(TZ).strftime("%Y%m%d")
    quotes = {"TCS": {"symbol": "TCS", "timestamp": "2026-08-31 10:00:00",
                      "last_price": 3000.0}}
    loader = FakeLoader(quotes, {})
    live_dir = tmp_path / "live"
    args = make_args(no_candles=True)

    ldr.fetch_cycle(loader, ["TCS"], live_dir, args)
    ldr.fetch_cycle(loader, ["TCS"], live_dir, args)  # same snapshot again

    qdf = pd.read_csv(live_dir / f"quotes_{today}.csv")
    assert len(qdf) == 1  # deduped on (symbol, timestamp)


def test_fetch_cycle_handles_missing_data(tmp_path):
    today = datetime.now(TZ).strftime("%Y%m%d")
    loader = FakeLoader({}, {})
    ldr.fetch_cycle(loader, ["NOPE"], tmp_path / "live",
                    make_args())  # must not raise
    assert not (tmp_path / "live" / f"quotes_{today}.csv").exists()
