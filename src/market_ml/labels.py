"""Label generation: next-day direction."""
from __future__ import annotations

import pandas as pd


def make_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Return 1 if close rises over the next ``horizon`` days, else 0.

    The final ``horizon`` rows have no future value and are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    future_close = close.shift(-horizon)
    labels = (future_close > close).astype(float)
    labels[future_close.isna()] = float("nan")
    return labels.rename("label")
