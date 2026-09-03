"""Unit tests for the jugaad-data loader (no live NSE requests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.jugaad_loader import SCHEMA_COLUMNS, _normalize, download_nse_stock


def _fake_stock_df(symbol: str = "TEST", from_date=None, to_date=None, series: str = "EQ") -> pd.DataFrame:
    """Mimic jugaad_data.nse.stock_df output (descending dates, 'mva' close)."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "DATE": [d.strftime("%d-%b-%Y") for d in reversed(dates)],
            "SERIES": ["EQ"] * 5,
            "PREV_CLOSE": [100.0] * 5,
            "OPEN": list(range(101, 106)),
            "HIGH": list(range(102, 107)),
            "LOW": list(range(99, 104)),
            "mva": list(range(100, 105)),
            "VWAP": [100.0] * 5,
            "VOLUME": [1_000_000.0] * 5,
        }
    )


class TestNormalize:
    def test_schema_columns(self):
        raw = _fake_stock_df("TEST")
        norm = _normalize(raw, "TEST", source="jugaad-data")
        assert list(norm.columns) == SCHEMA_COLUMNS + ["source"]

    def test_dates_timezone_naive(self):
        raw = _fake_stock_df("TEST")
        norm = _normalize(raw, "TEST", source="jugaad-data")
        assert pd.api.types.is_datetime64_any_dtype(norm["date"])
        assert norm["date"].dt.tz is None

    def test_numeric_dtypes(self):
        raw = _fake_stock_df("TEST")
        for col in ("open", "high", "low", "close", "volume"):
            assert pd.api.types.is_numeric_dtype(_normalize(raw, "TEST", source="jugaad-data")[col])

    def test_sorted_by_symbol_and_date(self):
        raw = _fake_stock_df("TEST")  # arrives in descending date order
        norm = _normalize(raw, "TEST", source="jugaad-data")
        assert norm["date"].is_monotonic_increasing
        assert (norm["symbol"] == "TEST").all()

    def test_duplicates_removed(self):
        raw = pd.concat([_fake_stock_df("TEST"), _fake_stock_df("TEST")], ignore_index=True)
        norm = _normalize(raw, "TEST", source="jugaad-data")
        assert not norm.duplicated(subset=["symbol", "date"]).any()
        assert len(norm) == 5

    def test_duplicate_dates_keeps_high_volume_row(self):
        """Regression: jugaad-data sometimes returns two rows for the same date —
        one real trading row and one low-volume adjustment row. The normalizer
        must keep the high-volume row."""
        dates = ["2025-01-06"] * 2  # same date
        raw = pd.DataFrame({
            "DATE": dates,
            "SERIES": ["EQ", "EQ"],
            "PREV_CLOSE": [100.0, 100.0],
            "OPEN": [100.0, 100.5],
            "HIGH": [105.0, 100.5],
            "LOW": [99.0, 100.5],
            "mva": [103.0, 100.5],
            "VWAP": [102.0, 100.5],
            "VOLUME": [5_000_000.0, 1.0],  # real row vs adjustment
        })
        norm = _normalize(raw, "TEST", source="jugaad-data")
        assert len(norm) == 1
        row = norm.iloc[0]
        # Must keep the high-volume row (high=105, close=103)
        assert row["high"] == 105.0
        assert row["close"] == 103.0
        assert row["volume"] == 5_000_000.0

    def test_mva_mapped_to_close(self):
        raw = _fake_stock_df("TEST")
        norm = _normalize(raw, "TEST", source="jugaad-data")
        assert sorted(norm["close"].tolist()) == [100, 101, 102, 103, 104]

    def test_missing_column_raises(self):
        raw = _fake_stock_df("TEST").drop(columns=["mva"])
        with pytest.raises(ValueError):
            _normalize(raw, "TEST", source="jugaad-data")

    def test_source_column_marked(self):
        norm = _normalize(_fake_stock_df("TEST"), "TEST", source="jugaad-data")
        assert (norm["source"] == "jugaad-data").all()


class TestDownloadNseStock:
    def test_mocked_stock_df(self, tmp_path, monkeypatch):
        monkeypatch.setattr("jugaad_data.nse.stock_df", _fake_stock_df)
        norm = download_nse_stock(
            "TEST",
            "2024-01-01",
            "2024-01-10",
            retries=1,
            raw_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
        )
        assert len(norm) == 5
        assert set(SCHEMA_COLUMNS).issubset(norm.columns)
        assert (norm["source"] == "jugaad-data").all()
        assert (tmp_path / "raw" / "TEST.csv").exists()
        assert (tmp_path / "interim" / "TEST.csv").exists()

    def test_failure_raises_after_retries(self, tmp_path, monkeypatch):
        def boom(**kwargs):
            raise ConnectionError("NSE down")

        monkeypatch.setattr("jugaad_data.nse.stock_df", boom)
        with pytest.raises(RuntimeError, match="All providers failed"):
            download_nse_stock(
                "TEST",
                "2024-01-01",
                "2024-01-10",
                retries=2,
                retry_delay=0,
                fallback_provider="none",
                raw_dir=tmp_path / "raw",
                interim_dir=tmp_path / "interim",
            )

    def test_yfinance_fallback_used(self, tmp_path, monkeypatch):
        def boom(**kwargs):
            raise ConnectionError("NSE down")

        monkeypatch.setattr("jugaad_data.nse.stock_df", boom)

        yf_frame = pd.DataFrame(
            {
                "Date": pd.bdate_range("2024-01-02", periods=3),
                "Open": [1.0, 2.0, 3.0],
                "High": [1.5, 2.5, 3.5],
                "Low": [0.9, 1.9, 2.9],
                "Close": [1.2, 2.2, 3.2],
                "Volume": [10.0, 20.0, 30.0],
            }
        ).set_index("Date")

        import market_ml.download as dl

        monkeypatch.setattr(dl, "fetch_ohlcv", lambda *a, **k: yf_frame.copy())
        norm = download_nse_stock(
            "TEST",
            "2024-01-01",
            "2024-01-10",
            retries=1,
            retry_delay=0,
            fallback_provider="yfinance",
            raw_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
        )
        assert (norm["source"] == "yfinance").all()
        assert len(norm) == 3
