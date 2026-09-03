import numpy as np
import pandas as pd
import pytest

from market_ml.labels import make_labels


def test_up_move_is_one_down_is_zero():
    close = pd.Series([100.0, 102.0, 101.0, 105.0])
    labels = make_labels(close)
    # next-day direction: up, down, up; last is NaN
    assert labels.iloc[0] == 1.0
    assert labels.iloc[1] == 0.0
    assert labels.iloc[2] == 1.0
    assert np.isnan(labels.iloc[3])


def test_flat_close_is_zero():
    close = pd.Series([100.0, 100.0])
    assert make_labels(close).iloc[0] == 0.0


def test_horizon_two():
    close = pd.Series([100.0, 99.0, 98.0, 103.0])
    labels = make_labels(close, horizon=2)
    # t0 -> t2: 98 < 100 -> 0 ; t1 -> t3: 103 > 99 -> 1 ; rest NaN
    assert labels.iloc[0] == 0.0
    assert labels.iloc[1] == 1.0
    assert np.isnan(labels.iloc[2])
    assert np.isnan(labels.iloc[3])


def test_invalid_horizon_raises():
    with pytest.raises(ValueError):
        make_labels(pd.Series([1.0, 2.0]), horizon=0)


def test_no_lookahead_tail_is_nan():
    close = pd.Series([1.0, 2.0, 3.0])
    labels = make_labels(close)
    assert np.isnan(labels.iloc[-1])
