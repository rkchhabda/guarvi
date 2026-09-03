import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from market_ml.data import load_ohlcv, make_synthetic_ohlcv
from market_ml.features import make_dataset
from market_ml.model import AVAILABLE_MODELS, build_model, evaluate_walk_forward
from market_ml.splits import walk_forward_splits


@pytest.fixture(scope="module")
def dataset():
    df = make_synthetic_ohlcv(n_days=700, seed=11)
    return make_dataset(df)


def test_load_ohlcv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ohlcv(tmp_path / "nope.csv")


def test_load_ohlcv_normalises_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,1,2,0.9,1.5,100\n")
    df = load_ohlcv(p)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_walk_forward_accuracy_reasonable(dataset, name):
    X, y = dataset
    res = evaluate_walk_forward(X, y, name, initial_train_size=500, test_size=25)
    assert len(res.folds) > 0
    accs = [f.accuracy for f in res.folds]
    assert all(0.0 <= a <= 1.0 for a in accs)
    assert 0.0 <= res.mean_accuracy <= 1.0


def test_walk_forward_no_future_leakage(dataset):
    """Training indices of every fold must be strictly before test indices."""
    X, y = dataset
    n = len(X)
    splits = walk_forward_splits(n, initial_train_size=n - 60, test_size=20)
    for s in splits:
        assert s.train_idx.max() < s.test_idx.min()


def test_build_model_unknown_name():
    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_model_predict_proba_shape(dataset):
    X, y = dataset
    model = build_model("logistic_regression")
    model.fit(X.iloc[:500], y.iloc[:500])
    proba = model.predict_proba(X.iloc[500:503])
    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
