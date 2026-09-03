import numpy as np
import pandas as pd
import pytest

from market_ml.data import make_synthetic_ohlcv
from market_ml.features import build_features, compute_atr, compute_macd, compute_rsi, make_dataset

EXPECTED_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "log_ret_1d",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_20_ratio",
    "sma_20_50_cross",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "bb_width",
    "atr_14_ratio",
    "volatility_10",
    "volatility_20",
    "volume_ratio",
    "volume_change",
]


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv(n_days=300, seed=7)


def test_feature_columns_present(ohlcv):
    feats = build_features(ohlcv)
    for col in EXPECTED_COLUMNS:
        assert col in feats.columns


def test_features_are_deterministic(ohlcv):
    f1 = build_features(ohlcv)
    f2 = build_features(ohlcv)
    pd.testing.assert_frame_equal(f1, f2)


def test_returns_match_manual_computation(ohlcv):
    feats = build_features(ohlcv)
    expected_ret = ohlcv["close"].pct_change(1)
    np.testing.assert_allclose(feats["ret_1d"].dropna(), expected_ret.dropna(), rtol=1e-12)

    expected_log = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    np.testing.assert_allclose(feats["log_ret_1d"].dropna(), expected_log.dropna(), rtol=1e-12)


def test_rsi_bounds(ohlcv):
    rsi = compute_rsi(ohlcv["close"])
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_rsi_extremes():
    up = pd.Series(np.linspace(100, 200, 60))
    assert compute_rsi(up).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.linspace(200, 100, 60))
    assert compute_rsi(down).iloc[-1] == pytest.approx(0.0)


def test_macd_relationship(ohlcv):
    macd_line, signal_line, hist = compute_macd(ohlcv["close"])
    np.testing.assert_allclose(hist, macd_line - signal_line, rtol=1e-12)


def test_atr_nonnegative(ohlcv):
    atr = compute_atr(ohlcv)
    assert (atr.dropna() >= 0).all()


def test_bollinger_position_within_band_mostly(ohlcv):
    feats = build_features(ohlcv)
    pos = feats["bb_position"].dropna()
    # Position should be finite and predominantly within [0, 1]
    assert pos.notna().all()
    assert (pos.between(-0.5, 1.5)).mean() > 0.95


def test_make_dataset_drops_nans_and_aligns(ohlcv):
    X, y = make_dataset(ohlcv)
    assert len(X) == len(y)
    assert X.notna().all().all()
    assert set(y.unique()).issubset({0, 1})
    # Labels must align with the same dates as features
    assert (X.index == y.index).all()


def test_volume_ratio_baseline(ohlcv):
    feats = build_features(ohlcv)
    ratio = feats["volume_ratio"].dropna()
    assert (ratio > 0).all()
