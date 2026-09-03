import numpy as np
import pytest

from market_ml.splits import assert_chronological, walk_forward_splits


def test_basic_split_sizes():
    splits = walk_forward_splits(n_samples=100, initial_train_size=50, test_size=10)
    assert len(splits) == 5
    for s in splits:
        assert len(s.test_idx) == 10
    # Expanding train window
    sizes = [len(s.train_idx) for s in splits]
    assert sizes == [50, 60, 70, 80, 90]


def test_train_always_precedes_test():
    splits = walk_forward_splits(n_samples=120, initial_train_size=60, test_size=5)
    for s in splits:
        assert s.train_idx.max() < s.test_idx.min()
        assert s.train_idx.min() == 0
    assert_chronological(splits)


def test_last_partial_fold():
    splits = walk_forward_splits(n_samples=103, initial_train_size=100, test_size=10)
    assert len(splits) == 1
    assert len(splits[0].test_idx) == 3


def test_step_size_larger_than_test_size():
    splits = walk_forward_splits(n_samples=100, initial_train_size=40, test_size=5, step_size=15)
    starts = [s.test_idx[0] for s in splits]
    assert starts == [40, 55, 70, 85]


def test_no_overlap_between_folds_when_step_equals_test():
    splits = walk_forward_splits(n_samples=100, initial_train_size=50, test_size=10)
    seen = []
    for s in splits:
        seen.extend(s.test_idx.tolist())
    assert len(seen) == len(set(seen))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        walk_forward_splits(n_samples=10, initial_train_size=1)
    with pytest.raises(ValueError):
        walk_forward_splits(n_samples=10, initial_train_size=5, test_size=0)
    with pytest.raises(ValueError):
        walk_forward_splits(n_samples=10, initial_train_size=5, step_size=0)


def test_never_shuffles_data():
    splits = walk_forward_splits(n_samples=30, initial_train_size=10, test_size=5)
    all_idx = np.concatenate([np.concatenate([s.train_idx, s.test_idx]) for s in splits])
    # Indices must be strictly increasing within each fold pair
    for s in splits:
        assert np.all(np.diff(s.train_idx) == 1)
        assert np.all(np.diff(s.test_idx) == 1)
