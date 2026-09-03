"""Walk-forward (expanding window) time-series splitting.

Only chronological splits are used — never random shuffling of market data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_samples: int,
    initial_train_size: int,
    test_size: int = 1,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Expanding-window walk-forward splits.

    Each split trains on everything up to a cut point and tests on the
    following ``test_size`` samples; the window then advances by ``step_size``
    (default: ``test_size``).
    """
    if initial_train_size < 2:
        raise ValueError("initial_train_size must be >= 2")
    if test_size < 1:
        raise ValueError("test_size must be >= 1")
    if step_size is None:
        step_size = test_size
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    splits: list[WalkForwardSplit] = []
    start = initial_train_size
    while start < n_samples:
        end = min(start + test_size, n_samples)
        splits.append(
            WalkForwardSplit(
                train_idx=np.arange(0, start),
                test_idx=np.arange(start, end),
            )
        )
        start += step_size
    return splits


def assert_chronological(splits: list[WalkForwardSplit]) -> None:
    """Guard: every training index must precede every test index."""
    for split in splits:
        if split.train_idx.max() >= split.test_idx.min():
            raise AssertionError("Walk-forward split violated chronology")
