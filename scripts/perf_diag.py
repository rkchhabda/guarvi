"""Quick performance diagnostic."""
import sys, time
sys.path.insert(0, "src")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
df = pd.read_parquet("data/processed/nifty100_features.parquet")
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded: {len(df)} rows in {time.time()-t0:.1f}s")

# Time the fold creation
from market_ml.baselines import create_walk_forward_folds, prepare_xy, FEATURE_COLS
X, y, meta = prepare_xy(df)
t1 = time.time()
holdout_mask, folds = create_walk_forward_folds(meta["date"])
print(f"Folds created: {len(folds)} in {time.time()-t1:.1f}s")

# Time one fold's data extraction
f = folds[0]
t2 = time.time()
train_idx = f.train_mask.values
test_idx = f.test_mask.values
print(f"Mask creation: {time.time()-t2:.3f}s")

t3 = time.time()
X_train = X.loc[train_idx, [c for c in FEATURE_COLS if c in X.columns]].copy()
print(f"X_train extraction ({X_train.shape}): {time.time()-t3:.3f}s")

t4 = time.time()
y_train = y.loc[train_idx].copy()
print(f"y_train extraction: {time.time()-t4:.3f}s")

# Time actual_return computation
t5 = time.time()
daily_returns = df.groupby("symbol")["close"].pct_change()
print(f"Daily returns: {time.time()-t5:.3f}s")

# Time model training
from market_ml.baselines import _build_model_no_imputer, _precompute_medians, _impute_and_return
t6 = time.time()
medians = _precompute_medians(X[[c for c in FEATURE_COLS if c in X.columns]])
print(f"Medians: {time.time()-t6:.3f}s")

t7 = time.time()
X_tr_imp = _impute_and_return(X_train, medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
y_tr_clean = y.loc[train_idx].dropna().astype(int)
valid = y_tr_clean.index
X_tr_clean = X_tr_imp.loc[valid]
y_tr_final = y_tr_clean
print(f"Impute+clean: {time.time()-t7:.3f}s, rows={len(y_tr_final)}")

t8 = time.time()
model = _build_model_no_imputer("logistic_regression")
model.fit(X_tr_final, y_tr_final)
print(f"LR fit: {time.time()-t8:.3f}s")

t9 = time.time()
X_test_raw = X.loc[test_idx, [c for c in FEATURE_COLS if c in X.columns]].copy()
X_test_imp = _impute_and_return(X_test_raw, medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
y_test_clean = y.loc[test_idx].dropna().astype(int)
valid_t = y_test_clean.index
X_test_final = X_test_imp.loc[valid_t]
y_test_final = y_test_clean
preds = model.predict_proba(X_test_final)[:, 1]
print(f"LR predict: {time.time()-t9:.3f}s, rows={len(y_test_final)}")
print(f"\nTotal per fold: ~{(time.time()-t8)+(time.time()-t7):.1f}s")
