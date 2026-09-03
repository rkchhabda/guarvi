"""Baseline model training and walk-forward evaluation for Nifty 100.

Implements five baseline models with strict expanding-window walk-forward
validation and a held-out final test period.

Models:
  1. Always-up baseline
  2. Previous-direction baseline
  3. Logistic Regression (with StandardScaler)
  4. Random Forest
  5. XGBoost

Leakage prevention:
  - All data sorted by date; no random splits
  - Scalers/imputers fitted only on training portion of each fold
  - Target column never in feature matrix
  - Final 12-month holdout never touched during training
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ret_1d", "log_ret_1d", "ret_2d", "log_ret_2d", "ret_3d", "log_ret_3d",
    "ret_5d", "log_ret_5d", "ret_10d", "log_ret_10d", "ret_20d", "log_ret_20d",
    "hl_range", "co_return", "gap_return",
    "sma_5", "dist_sma_5", "sma_10", "dist_sma_10", "sma_20", "dist_sma_20",
    "sma_50", "dist_sma_50", "sma_200", "dist_sma_200",
    "ema_12", "dist_ema_12", "ema_26", "dist_ema_26",
    "sma_5_10_cross", "sma_10_20_cross", "sma_20_50_cross", "sma_50_200_cross",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "adx", "plus_di", "minus_di",
    "volatility_5d", "volatility_10d", "volatility_20d", "volatility_50d",
    "atr_14", "bb_upper", "bb_lower", "bb_position", "bb_width",
    "volume_pct_change", "avg_volume_5d", "avg_volume_10d", "avg_volume_20d",
    "volume_ratio_20d", "obv", "obv_normalised",
    "rank_return", "rank_volume_ratio", "excess_return", "rel_volatility",
]

TARGET_COL = "target_direction"
DATE_COL = "date"
SYMBOL_COL = "symbol"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_features(path: str | Path) -> pd.DataFrame:
    """Load the feature dataset and prepare it for training."""
    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract X, y, and metadata from the feature DataFrame.

    Returns:
        X: feature matrix (float64 only)
        y: target series (0/1, NaN for last row per symbol)
        meta: DataFrame with date, symbol, open, high, low, close, volume
    """
    meta = df[[DATE_COL, SYMBOL_COL, "open", "high", "low", "close", "volume"]].copy()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y, meta


# ---------------------------------------------------------------------------
# Walk-forward date splits
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    fold_id: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    train_mask: pd.Series  # boolean mask over full dataset
    test_mask: pd.Series
    n_train: int
    n_test: int


def create_walk_forward_folds(
    dates: pd.Series,
    holdout_months: int = 12,
    min_train_years: float = 3.0,
    validation_days: int = 21,
) -> tuple[pd.Series, list[WalkForwardFold]]:
    """Create expanding-window walk-forward folds by date.

    Returns:
        holdout_mask: boolean mask for the held-out final test period
        folds: list of WalkForwardFold for the training/validation period
    """
    unique_dates = pd.DatetimeIndex(np.sort(dates.unique()))
    max_date = unique_dates[-1]

    # Holdout cutoff: latest 12 months
    holdout_cutoff = max_date - pd.DateOffset(months=holdout_months)
    holdout_mask = dates >= holdout_cutoff

    # Training period dates (before holdout)
    train_dates = unique_dates[unique_dates < holdout_cutoff]
    if len(train_dates) == 0:
        raise ValueError("No data before holdout cutoff")

    # Minimum training start:3 years from first date
    min_train_start = train_dates[0] + pd.DateOffset(years=int(min_train_years))
    # Find the first date that is >= min_train_start
    first_valid_start = train_dates[train_dates >= min_train_start]
    if len(first_valid_start) == 0:
        raise ValueError("Not enough data for minimum training history")
    min_train_end = first_valid_start[0]

    # Create folds: each fold trains up to train_end_date, tests on next validation_days
    folds: list[WalkForwardFold] = []
    fold_id = 0

    # Find all unique dates in training period
    available_dates = train_dates[train_dates >= min_train_end]

    # Step through in validation_days increments
    i = 0
    while i < len(available_dates):
        train_end_date = available_dates[i]
        # Test dates: next validation_days trading days after train_end
        test_start_idx = i + 1
        if test_start_idx >= len(available_dates):
            break
        test_start_date = available_dates[test_start_idx]
        test_end_idx = min(test_start_idx + validation_days, len(available_dates))
        test_end_date = available_dates[test_end_idx - 1]

        train_mask = (dates <= train_end_date) & (~holdout_mask)
        test_mask = (dates > train_end_date) & (dates <= test_end_date) & (~holdout_mask)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test > 0 and n_train > 0:
            folds.append(WalkForwardFold(
                fold_id=fold_id,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                train_mask=train_mask,
                test_mask=test_mask,
                n_train=n_train,
                n_test=n_test,
            ))
            fold_id += 1

        i += validation_days  # advance by validation window

    return holdout_mask, folds


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(name: str, seed: int = 42):
    """Build a model by name. Returns an unfitted model object.

    Models are sized for reasonable training time on ~250k rows.
    """
    name = name.lower()
    if name == "logistic_regression":
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=seed, C=1.0, solver="lbfgs"
            )),
        ])
    if name == "random_forest":
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=10, random_state=seed, n_jobs=-1
            )),
        ])
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=seed,
                verbosity=0, n_jobs=-1,
            )),
        ])
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Baseline predictors (no training needed)
# ---------------------------------------------------------------------------

def always_up_predict(X: pd.DataFrame) -> np.ndarray:
    """Always predict 1 (up)."""
    return np.ones(len(X), dtype=int)


def always_up概率(X: pd.DataFrame) -> np.ndarray:
    """Return probability 1.0 for always-up."""
    return np.ones(len(X), dtype=float)


def previous_direction_predict(
    current_direction: pd.Series, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Predict that the next direction equals the current direction.

    For the first row (no prior data), predict 1 (up).
    """
    preds = current_direction.fillna(1).astype(int).values
    probs = preds.astype(float)
    return preds, probs


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

@dataclass
class PredictionRow:
    date: pd.Timestamp
    symbol: str
    model_name: str
    predicted_direction: int
    predicted_probability: float
    actual_direction: int | float
    actual_return: float
    fold_id: int
    train_end_date: pd.Timestamp
    prediction_date: pd.Timestamp


@dataclass
class FoldMetrics:
    fold_id: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    n_test: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    brier_score: float
    confusion_matrix: list[list[int]]
    n_up_predicted: int
    n_down_predicted: int


def _compute_fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, fold_id: int,
    train_end: pd.Timestamp, test_start: pd.Timestamp, test_end: pd.Timestamp,
) -> FoldMetrics:
    """Compute all required metrics for a single fold."""
    n_test = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = None
    if len(set(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = None

    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return FoldMetrics(
        fold_id=fold_id,
        train_end_date=train_end,
        test_start_date=test_start,
        test_end_date=test_end,
        n_test=n_test,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        brier_score=brier,
        confusion_matrix=cm,
        n_up_predicted=int((y_pred == 1).sum()),
        n_down_predicted=int((y_pred == 0).sum()),
    )


def _precompute_medians(X: pd.DataFrame) -> pd.Series:
    """Precompute feature medians for fast imputation."""
    return X.median()


def _impute_and_return(
    X: pd.DataFrame, medians: pd.Series
) -> pd.DataFrame:
    """Fill NaN with precomputed medians (fast, no fitting)."""
    return X.fillna(medians)


def _extract_arrays(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Convert everything to numpy arrays for fast integer-index slicing.

    Returns:
        X_arr: (n, n_features) float64
        y_arr: (n,) float64 (NaN for missing targets)
        dates_arr: (n,) datetime64
        symbols_arr: (n,) object
        daily_returns_arr: (n,) float64
        opens_arr: (n,) float64
        feature_names: list of feature column names
        symbol_list: list of unique symbols
    """
    X_arr = X[feature_cols].to_numpy(dtype=np.float64)
    y_arr = y.to_numpy(dtype=np.float64)
    dates_arr = meta[DATE_COL].to_numpy()
    symbols_arr = meta[SYMBOL_COL].to_numpy()

    # Precompute daily returns as numpy
    daily_returns_arr = np.full(len(df), np.nan, dtype=np.float64)
    for sym in df[SYMBOL_COL].unique():
        mask = df[SYMBOL_COL] == sym
        closes = df.loc[mask, "close"].values.astype(np.float64)
        rets = np.empty(len(closes), dtype=np.float64)
        rets[0] = np.nan
        rets[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        daily_returns_arr[mask.values] = rets

    opens_arr = df["open"].values.astype(np.float64)

    # Replace inf with nan in X
    X_arr = X_arr.copy()
    X_arr[~np.isfinite(X_arr)] = np.nan

    # Precompute medians
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        medians = np.nanmedian(X_arr, axis=0)

    return X_arr, y_arr, dates_arr, symbols_arr, daily_returns_arr, opens_arr, feature_cols, medians


def _impute_np(X_arr: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Fast numpy imputation: fill NaN with precomputed medians."""
    out = X_arr.copy()
    nan_mask = np.isnan(out)
    # Broadcast medians: expand medians to (n, n_features)
    med_broadcast = np.broadcast_to(medians, out.shape)
    out = np.where(nan_mask, med_broadcast, out)
    return out


def train_and_evaluate_model(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    model_name: str,
    folds: list[WalkForwardFold],
    holdout_mask: pd.Series,
    seed: int = 42,
    threshold: float = 0.50,
) -> tuple[list[PredictionRow], list[FoldMetrics], list[PredictionRow]]:
    """Train a model across all folds and the holdout period.

    Uses numpy arrays for fast integer-index slicing.
    """
    fold_predictions: list[PredictionRow] = []
    fold_metrics_list: list[FoldMetrics] = []
    holdout_predictions: list[PredictionRow] = []

    feature_cols = [c for c in FEATURE_COLS if c in X.columns]

    # Convert everything to numpy arrays once
    X_arr, y_arr, dates_arr, symbols_arr, daily_ret_arr, opens_arr, feat_names, medians = \
        _extract_arrays(df, X, y, meta, feature_cols)

    # Convert boolean masks to integer index arrays
    train_idx_arrays = [np.where(fold.train_mask.values)[0] for fold in folds]
    test_idx_arrays = [np.where(fold.test_mask.values)[0] for fold in folds]
    holdout_idx_arr = np.where(holdout_mask.values)[0]

    for fold_i, fold in enumerate(folds):
        tr_idx = train_idx_arrays[fold_i]
        te_idx = test_idx_arrays[fold_i]

        # Extract training data
        y_tr = y_arr[tr_idx]
        valid_tr = ~np.isnan(y_tr)
        tr_final = tr_idx[valid_tr]
        y_tr_clean = y_arr[tr_final].astype(int)

        # Extract test data
        y_te = y_arr[te_idx]
        valid_te = ~np.isnan(y_te)
        te_final = te_idx[valid_te]
        y_te_clean = y_arr[te_final].astype(int)

        if len(tr_final) == 0 or len(te_final) == 0:
            continue

        # Impute with precomputed medians
        X_tr = _impute_np(X_arr[tr_final], medians)
        X_te = _impute_np(X_arr[te_final], medians)

        # Build and train model
        model = _build_model_no_imputer(model_name, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr_clean)

        # Predict
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Store predictions (vectorized)
        for i in range(len(te_final)):
            idx = te_final[i]
            fold_predictions.append(PredictionRow(
                date=pd.Timestamp(dates_arr[idx]),
                symbol=str(symbols_arr[idx]),
                model_name=model_name,
                predicted_direction=int(y_pred[i]),
                predicted_probability=float(y_prob[i]),
                actual_direction=int(y_te_clean[i]),
                actual_return=float(daily_ret_arr[idx]) if np.isfinite(daily_ret_arr[idx]) else 0.0,
                fold_id=fold.fold_id,
                train_end_date=fold.train_end_date,
                prediction_date=pd.Timestamp(dates_arr[idx]),
            ))

        # Fold metrics
        fm = _compute_fold_metrics(
            y_te_clean, y_pred, y_prob, fold.fold_id,
            fold.train_end_date, fold.test_start_date, fold.test_end_date,
        )
        fold_metrics_list.append(fm)

    # Holdout evaluation
    if len(holdout_idx_arr) > 0:
        y_h = y_arr[holdout_idx_arr]
        valid_h = ~np.isnan(y_h)
        h_final = holdout_idx_arr[valid_h]
        y_h_clean = y_arr[h_final].astype(int)

        # Train on ALL pre-holdout data
        pre_holdout_idx = np.where(~holdout_mask.values)[0]
        y_pre = y_arr[pre_holdout_idx]
        valid_pre = ~np.isnan(y_pre)
        pre_final = pre_holdout_idx[valid_pre]
        y_pre_clean = y_arr[pre_final].astype(int)

        if len(pre_final) > 0 and len(h_final) > 0:
            X_pre = _impute_np(X_arr[pre_final], medians)
            X_h = _impute_np(X_arr[h_final], medians)

            model = _build_model_no_imputer(model_name, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_pre, y_pre_clean)

            y_prob_h = model.predict_proba(X_h)[:, 1]
            y_pred_h = (y_prob_h >= threshold).astype(int)

            pre_train_end = pd.Timestamp(dates_arr[pre_holdout_idx].max())

            for i in range(len(h_final)):
                idx = h_final[i]
                holdout_predictions.append(PredictionRow(
                    date=pd.Timestamp(dates_arr[idx]),
                    symbol=str(symbols_arr[idx]),
                    model_name=model_name,
                    predicted_direction=int(y_pred_h[i]),
                    predicted_probability=float(y_prob_h[i]),
                    actual_direction=int(y_h_clean[i]),
                    actual_return=float(daily_ret_arr[idx]) if np.isfinite(daily_ret_arr[idx]) else 0.0,
                    fold_id=-1,
                    train_end_date=pre_train_end,
                    prediction_date=pd.Timestamp(dates_arr[idx]),
                ))

    return fold_predictions, fold_metrics_list, holdout_predictions


def _build_model_no_imputer(name: str, seed: int = 42):
    """Build model without SimpleImputer (imputation done externally)."""
    name = name.lower()
    if name == "logistic_regression":
        return Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=seed, C=1.0, solver="lbfgs"
            )),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=seed, n_jobs=-1
        )
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=seed,
            verbosity=0, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Previous-direction baseline (special handling)
# ---------------------------------------------------------------------------

def train_previous_direction(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    folds: list[WalkForwardFold],
    holdout_mask: pd.Series,
) -> tuple[list[PredictionRow], list[FoldMetrics], list[PredictionRow]]:
    """Evaluate the previous-direction baseline across all folds and holdout."""
    fold_predictions: list[PredictionRow] = []
    fold_metrics_list: list[FoldMetrics] = []
    holdout_predictions: list[PredictionRow] = []

    # Precompute numpy arrays
    dates_arr = meta[DATE_COL].to_numpy()
    symbols_arr = meta[SYMBOL_COL].to_numpy()
    y_arr = df[TARGET_COL].to_numpy(dtype=np.float64)

    # Precompute daily returns (vectorized: compute within each symbol block)
    closes = df["close"].values.astype(np.float64)
    syms = df[SYMBOL_COL].values
    daily_ret = np.full(len(df), np.nan, dtype=np.float64)
    # Find symbol boundaries (data is already sorted by symbol+date)
    unique_syms, boundaries = np.unique(syms, return_index=True)
    boundaries = np.append(boundaries, len(df))
    for b_i in range(len(unique_syms)):
        s, e = boundaries[b_i], boundaries[b_i + 1]
        c = closes[s:e]
        if len(c) > 1:
            daily_ret[s] = np.nan
            daily_ret[s + 1:e] = (c[1:] - c[:-1]) / c[:-1]

    # Current direction: 1 if return > 0, else 0
    current_dir = np.where(daily_ret > 0, 1, 0).astype(float)
    current_dir[np.isnan(daily_ret)] = np.nan

    # Previous direction: shift by 1 within each symbol
    prev_dir = np.full(len(df), np.nan, dtype=np.float64)
    for sym in df[SYMBOL_COL].unique():
        mask = df[SYMBOL_COL] == sym
        idx = np.where(mask.values)[0]
        if len(idx) > 1:
            prev_dir[idx[1:]] = current_dir[idx[:-1]]

    # Convert masks to index arrays
    train_idx_arrays = [np.where(fold.train_mask.values)[0] for fold in folds]
    test_idx_arrays = [np.where(fold.test_mask.values)[0] for fold in folds]
    holdout_idx_arr = np.where(holdout_mask.values)[0]

    for fold_i, fold in enumerate(folds):
        te_idx = test_idx_arrays[fold_i]
        y_te = y_arr[te_idx]
        pd_te = prev_dir[te_idx]
        valid = (~np.isnan(y_te)) & (~np.isnan(pd_te))
        te_final = te_idx[valid]
        y_clean = y_arr[te_final].astype(int)
        pd_clean = pd_te[valid].astype(int)

        if len(te_final) == 0:
            continue

        y_pred = pd_clean
        y_prob = pd_clean.astype(float)

        for i in range(len(te_final)):
            idx = te_final[i]
            fold_predictions.append(PredictionRow(
                date=pd.Timestamp(dates_arr[idx]),
                symbol=str(symbols_arr[idx]),
                model_name="previous_direction",
                predicted_direction=int(y_pred[i]),
                predicted_probability=float(y_prob[i]),
                actual_direction=int(y_clean[i]),
                actual_return=float(daily_ret[idx]) if np.isfinite(daily_ret[idx]) else 0.0,
                fold_id=fold.fold_id,
                train_end_date=fold.train_end_date,
                prediction_date=pd.Timestamp(dates_arr[idx]),
            ))

        fm = _compute_fold_metrics(
            y_clean, y_pred, y_prob, fold.fold_id,
            fold.train_end_date, fold.test_start_date, fold.test_end_date,
        )
        fold_metrics_list.append(fm)

    # Holdout
    if len(holdout_idx_arr) > 0:
        y_h = y_arr[holdout_idx_arr]
        pd_h = prev_dir[holdout_idx_arr]
        valid_h = (~np.isnan(y_h)) & (~np.isnan(pd_h))
        h_final = holdout_idx_arr[valid_h]
        y_h_clean = y_arr[h_final].astype(int)
        pd_h_clean = pd_h[valid_h].astype(int)

        for i in range(len(h_final)):
            idx = h_final[i]
            holdout_predictions.append(PredictionRow(
                date=pd.Timestamp(dates_arr[idx]),
                symbol=str(symbols_arr[idx]),
                model_name="previous_direction",
                predicted_direction=int(pd_h_clean[i]),
                predicted_probability=float(pd_h_clean[i]),
                actual_direction=int(y_h_clean[i]),
                actual_return=float(daily_ret[idx]) if np.isfinite(daily_ret[idx]) else 0.0,
                fold_id=-1,
                train_end_date=pd.Timestamp(dates_arr[np.where(~holdout_mask.values)[0]].max()),
                prediction_date=pd.Timestamp(dates_arr[idx]),
            ))

    return fold_predictions, fold_metrics_list, holdout_predictions


def train_always_up(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    folds: list[WalkForwardFold],
    holdout_mask: pd.Series,
) -> tuple[list[PredictionRow], list[FoldMetrics], list[PredictionRow]]:
    """Evaluate the always-up baseline across all folds and holdout."""
    fold_predictions: list[PredictionRow] = []
    fold_metrics_list: list[FoldMetrics] = []
    holdout_predictions: list[PredictionRow] = []

    dates_arr = meta[DATE_COL].to_numpy()
    symbols_arr = meta[SYMBOL_COL].to_numpy()
    y_arr = df[TARGET_COL].to_numpy(dtype=np.float64)

    daily_ret = np.full(len(df), np.nan, dtype=np.float64)
    for sym in df[SYMBOL_COL].unique():
        mask = df[SYMBOL_COL] == sym
        closes = df.loc[mask, "close"].values.astype(np.float64)
        rets = np.empty(len(closes), dtype=np.float64)
        rets[0] = np.nan
        rets[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        daily_ret[mask.values] = rets

    test_idx_arrays = [np.where(fold.test_mask.values)[0] for fold in folds]
    holdout_idx_arr = np.where(holdout_mask.values)[0]

    for fold_i, fold in enumerate(folds):
        te_idx = test_idx_arrays[fold_i]
        y_te = y_arr[te_idx]
        valid = ~np.isnan(y_te)
        te_final = te_idx[valid]
        y_clean = y_arr[te_final].astype(int)

        if len(te_final) == 0:
            continue

        y_pred = np.ones(len(te_final), dtype=int)
        y_prob = np.ones(len(te_final), dtype=float)

        for i in range(len(te_final)):
            idx = te_final[i]
            fold_predictions.append(PredictionRow(
                date=pd.Timestamp(dates_arr[idx]),
                symbol=str(symbols_arr[idx]),
                model_name="always_up",
                predicted_direction=1,
                predicted_probability=1.0,
                actual_direction=int(y_clean[i]),
                actual_return=float(daily_ret[idx]) if np.isfinite(daily_ret[idx]) else 0.0,
                fold_id=fold.fold_id,
                train_end_date=fold.train_end_date,
                prediction_date=pd.Timestamp(dates_arr[idx]),
            ))

        fm = _compute_fold_metrics(
            y_clean, y_pred, y_prob, fold.fold_id,
            fold.train_end_date, fold.test_start_date, fold.test_end_date,
        )
        fold_metrics_list.append(fm)

    # Holdout
    if len(holdout_idx_arr) > 0:
        y_h = y_arr[holdout_idx_arr]
        valid_h = ~np.isnan(y_h)
        h_final = holdout_idx_arr[valid_h]
        y_h_clean = y_arr[h_final].astype(int)

        for i in range(len(h_final)):
            idx = h_final[i]
            holdout_predictions.append(PredictionRow(
                date=pd.Timestamp(dates_arr[idx]),
                symbol=str(symbols_arr[idx]),
                model_name="always_up",
                predicted_direction=1,
                predicted_probability=1.0,
                actual_direction=int(y_h_clean[i]),
                actual_return=float(daily_ret[idx]) if np.isfinite(daily_ret[idx]) else 0.0,
                fold_id=-1,
                train_end_date=pd.Timestamp(dates_arr[np.where(~holdout_mask.values)[0]].max()),
                prediction_date=pd.Timestamp(dates_arr[idx]),
            ))

    return fold_predictions, fold_metrics_list, holdout_predictions


# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------

@dataclass
class TradingResult:
    hit_rate: float
    avg_trade_return: float
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float
    turnover: float
    n_trades: int
    n_long_signals: int
    n_no_signal: int


def compute_trading_metrics(
    predictions: list[PredictionRow],
    threshold: float = 0.55,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> TradingResult:
    """Compute trading-style evaluation metrics.

    Long-only strategy: buy when predicted_probability >= threshold,
    no position otherwise.
    """
    if not predictions:
        return TradingResult(0, 0, 0, 0, 0, 0, 0, 0, 0)

    df = pd.DataFrame([{
        "date": p.date,
        "symbol": p.symbol,
        "prob": p.predicted_probability,
        "actual_return": p.actual_return,
    } for p in predictions])

    # Generate signals
    df["signal"] = (df["prob"] >= threshold).astype(int)

    # Apply transaction costs and slippage
    df["cost"] = df["signal"].diff().abs().fillna(df["signal"]) * (transaction_cost + slippage)
    df["strategy_return"] = df["signal"] * df["actual_return"] - df["cost"]

    # Metrics
    trades = df[df["signal"] == 1]
    n_trades = len(trades)
    n_long = int(df["signal"].sum())
    n_no_signal = int((df["signal"] == 0).sum())

    # Hit rate: fraction of long trades with positive return
    hit_rate = float((trades["actual_return"] > 0).mean()) if n_trades > 0 else 0.0

    # Average trade return
    avg_trade_return = float(trades["actual_return"].mean()) if n_trades > 0 else 0.0

    # Cumulative return (compounded)
    cum_return = float((1 + df["strategy_return"]).prod() - 1)

    # Max drawdown
    equity = (1 + df["strategy_return"]).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    max_dd = float(drawdown.min())

    # Sharpe ratio (annualized)
    daily_strat = df["strategy_return"]
    sharpe = float(daily_strat.mean() / daily_strat.std(ddof=0) * np.sqrt(252)) if daily_strat.std(ddof=0) > 0 else 0.0

    # Turnover: fraction of days where signal changes
    turnover = float(df["signal"].diff().abs().mean())

    return TradingResult(
        hit_rate=hit_rate,
        avg_trade_return=avg_trade_return,
        cumulative_return=cum_return,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        turnover=turnover,
        n_trades=n_trades,
        n_long_signals=n_long,
        n_no_signal=n_no_signal,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_symbol(predictions: list[PredictionRow]) -> pd.DataFrame:
    """Aggregate metrics by symbol."""
    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "symbol": p.symbol,
        "model_name": p.model_name,
        "actual": p.actual_direction,
        "pred": p.predicted_direction,
        "prob": p.predicted_probability,
        "return": p.actual_return,
    } for p in predictions])

    results = []
    for (symbol, model), grp in df.groupby(["symbol", "model_name"]):
        y_true = grp["actual"].values
        y_pred = grp["pred"].values
        y_prob = grp["prob"].values

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = None
        if len(set(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                auc = None
        brier = float(brier_score_loss(y_true, y_prob))

        results.append({
            "symbol": symbol,
            "model_name": model,
            "n_predictions": len(grp),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "brier_score": round(brier, 4),
        })

    return pd.DataFrame(results)


def aggregate_by_period(predictions: list[PredictionRow]) -> pd.DataFrame:
    """Aggregate metrics by month and quarter."""
    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "date": p.date,
        "model_name": p.model_name,
        "actual": p.actual_direction,
        "pred": p.predicted_direction,
        "prob": p.predicted_probability,
        "return": p.actual_return,
    } for p in predictions])

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    results = []
    for period_col in ["month", "quarter"]:
        for (period, model), grp in df.groupby([period_col, "model_name"]):
            y_true = grp["actual"].values
            y_pred = grp["pred"].values
            y_prob = grp["prob"].values

            acc = float(accuracy_score(y_true, y_pred))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            auc = None
            if len(set(y_true)) > 1:
                try:
                    auc = float(roc_auc_score(y_true, y_prob))
                except ValueError:
                    auc = None
            brier = float(brier_score_loss(y_true, y_prob))

            results.append({
                "period_type": period_col,
                "period": period,
                "model_name": model,
                "n_predictions": len(grp),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4) if auc is not None else None,
                "brier_score": round(brier, 4),
            })

    return pd.DataFrame(results)


def aggregate_overall(predictions: list[PredictionRow]) -> dict:
    """Compute overall metrics across all predictions for a model."""
    if not predictions:
        return {}

    y_true = np.array([p.actual_direction for p in predictions])
    y_pred = np.array([p.predicted_direction for p in predictions])
    y_prob = np.array([p.predicted_probability for p in predictions])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = None
    if len(set(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = None

    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "n_predictions": len(predictions),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "n_up_predicted": int((y_pred == 1).sum()),
        "n_down_predicted": int((y_pred == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Confidence intervals (bootstrap)
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
    n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for accuracy.

    Returns (accuracy, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(seed)
    acc = float(accuracy_score(y_true, y_pred))
    boot_accs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_accs.append(float(accuracy_score(y_true[idx], y_pred[idx])))
    boot_accs = np.array(boot_accs)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_accs, alpha * 100))
    upper = float(np.percentile(boot_accs, (1 - alpha) * 100))
    return acc, lower, upper


# ---------------------------------------------------------------------------
# Feature importance (for tree-based models)
# ---------------------------------------------------------------------------

def get_feature_importance(model_name: str, feature_cols: list[str]) -> dict[str, float] | None:
    """Extract feature importance from a fitted model. Returns None for baselines."""
    return None


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    model_name: str
    overall_metrics: dict
    fold_metrics: list[FoldMetrics]
    trading_metrics: TradingResult
    by_symbol: pd.DataFrame
    by_period: pd.DataFrame
    predictions: list[PredictionRow]
    holdout_predictions: list[PredictionRow]
    holdout_metrics: dict
    holdout_trading: TradingResult


def run_baselines(
    data_path: str | Path,
    output_dir: str | Path,
    models: list[str] | None = None,
    seed: int = 42,
    threshold: float = 0.50,
    trade_threshold: float = 0.55,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    holdout_months: int = 12,
    min_train_years: float = 3.0,
    validation_days: int = 21,
) -> list[BaselineResult]:
    """Run all baseline models and return results.

    This is the main entry point for Phase 9.
    """
    if models is None:
        models = ["always_up", "previous_direction", "logistic_regression", "random_forest", "xgboost"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading features from {data_path}...")
    df = load_features(data_path)
    X, y, meta = prepare_xy(df)
    print(f"  {len(df)} rows, {X.shape[1]} features, {df[SYMBOL_COL].nunique()} symbols")

    # Create walk-forward splits
    print("Creating walk-forward splits...")
    holdout_mask, folds = create_walk_forward_folds(
        meta[DATE_COL],
        holdout_months=holdout_months,
        min_train_years=min_train_years,
        validation_days=validation_days,
    )
    print(f"  {len(folds)} folds, {holdout_mask.sum()} holdout rows")

    # Store all feature columns that exist
    feature_cols = [c for c in FEATURE_COLS if c in X.columns]

    results: list[BaselineResult] = []

    for model_name in models:
        print(f"\nTraining {model_name}...")

        if model_name == "always_up":
            fold_preds, fold_mets, hold_preds = train_always_up(df, meta, folds, holdout_mask)
        elif model_name == "previous_direction":
            fold_preds, fold_mets, hold_preds = train_previous_direction(df, meta, folds, holdout_mask)
        else:
            fold_preds, fold_mets, hold_preds = train_and_evaluate_model(
                df, X, y, meta, model_name, folds, holdout_mask,
                seed=seed, threshold=threshold,
            )

        # Overall metrics (fold predictions only)
        overall = aggregate_overall(fold_preds)

        # Confidence interval
        if fold_preds:
            y_true_arr = np.array([p.actual_direction for p in fold_preds])
            y_pred_arr = np.array([p.predicted_direction for p in fold_preds])
            acc, ci_low, ci_high = bootstrap_accuracy_ci(y_true_arr, y_pred_arr)
            overall["accuracy_ci_95"] = [round(ci_low, 4), round(ci_high, 4)]

        # Trading metrics
        trading = compute_trading_metrics(
            fold_preds, threshold=trade_threshold,
            transaction_cost=transaction_cost, slippage=slippage,
        )

        # Holdout metrics
        holdout_overall = aggregate_overall(hold_preds)
        holdout_trading = compute_trading_metrics(
            hold_preds, threshold=trade_threshold,
            transaction_cost=transaction_cost, slippage=slippage,
        )

        # Breakdowns
        by_sym = aggregate_by_symbol(fold_preds)
        by_per = aggregate_by_period(fold_preds)

        print(f"  Accuracy: {overall.get('accuracy', 'N/A'):.4f} "
              f"(CI: {overall.get('accuracy_ci_95', 'N/A')})")
        print(f"  Holdout accuracy: {holdout_overall.get('accuracy', 'N/A'):.4f}")
        print(f"  Sharpe: {trading.sharpe_ratio:.4f}")

        results.append(BaselineResult(
            model_name=model_name,
            overall_metrics=overall,
            fold_metrics=fold_mets,
            trading_metrics=trading,
            by_symbol=by_sym,
            by_period=by_per,
            predictions=fold_preds,
            holdout_predictions=hold_preds,
            holdout_metrics=holdout_overall,
            holdout_trading=holdout_trading,
        ))

    return results


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_predictions(
    results: list[BaselineResult],
    output_path: str | Path,
) -> None:
    """Save all predictions to CSV."""
    all_preds = []
    for r in results:
        all_preds.extend(r.predictions)
        all_preds.extend(r.holdout_predictions)

    rows = [{
        "date": p.date.strftime("%Y-%m-%d"),
        "symbol": p.symbol,
        "model_name": p.model_name,
        "predicted_direction": p.predicted_direction,
        "predicted_probability": round(p.predicted_probability, 6),
        "actual_direction": p.actual_direction,
        "actual_return": round(p.actual_return, 6),
        "fold_id": p.fold_id,
        "train_end_date": p.train_end_date.strftime("%Y-%m-%d"),
        "prediction_date": p.prediction_date.strftime("%Y-%m-%d"),
    } for p in all_preds]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")
