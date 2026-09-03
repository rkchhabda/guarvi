"""Model training and evaluation with walk-forward validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .splits import WalkForwardSplit, walk_forward_splits

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


@dataclass
class FoldResult:
    fold: int
    train_end: int
    accuracy: float
    n_test: int


@dataclass
class WalkForwardResult:
    model_name: str
    folds: list[FoldResult]

    @property
    def mean_accuracy(self) -> float:
        weights = np.array([f.n_test for f in self.folds], dtype=float)
        accs = np.array([f.accuracy for f in self.folds], dtype=float)
        return float(np.average(accs, weights=weights))


def build_model(name: str, seed: int = 42):
    name = name.lower()
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


AVAILABLE_MODELS = ["logistic_regression", "random_forest", "xgboost"]


def evaluate_walk_forward(
    X, y, model_name: str, initial_train_size: int, test_size: int = 21, seed: int = 42
) -> WalkForwardResult:
    """Train and score a model across expanding-window walk-forward folds."""
    splits: list[WalkForwardSplit] = walk_forward_splits(
        len(X), initial_train_size=initial_train_size, test_size=test_size
    )
    folds: list[FoldResult] = []
    y_arr = np.asarray(y)
    for i, split in enumerate(splits):
        model = build_model(model_name, seed=seed)
        model.fit(X.iloc[split.train_idx], y_arr[split.train_idx])
        pred = model.predict(X.iloc[split.test_idx])
        acc = accuracy_score(y_arr[split.test_idx], pred)
        folds.append(FoldResult(i, int(split.train_idx[-1]), float(acc), len(split.test_idx)))
    return WalkForwardResult(model_name=model_name, folds=folds)
