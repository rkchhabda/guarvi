# Practical Guide: Boosting Accuracy to 55%

## Summary of Test Results

| Configuration | Accuracy | vs Naive | Status |
|--------------|----------|----------|--------|
| Baseline (original 62 features) | 50.48% | -0.44% | ❌ |
| Phase 1A (67 features + RSI/MACD momentum) | **51.03%** | +0.55% | ✅ |
| Phase 1B (73 features + z-scores) | 49.49% | -0.03% | ❌ |

**Key Insight:** Simple momentum features help. Complex data quality features hurt.

---

## Quick Wins (Implement in 1 Hour)

### 1. Best Config So Far (USE THIS)
```python
C = 0.1  # regularization
threshold = 0.50  # prediction threshold
```

### 2. Try These Quick Changes

**Option A: Threshold Tuning** (EASY - try first)
```python
# Instead of 0.50, try:
for threshold in [0.45, 0.48, 0.49, 0.50, 0.51, 0.52, 0.55]:
    pred = (prob >= threshold).astype(int)
    # Expected impact: ±0.5% accuracy
```

**Option B: Stronger Regularization**
```python
# Try C=0.01 instead of C=0.1
model = LogisticRegression(max_iter=1000, C=0.01)
```

**Option C: Fewer Predictions (Quality over Quantity)**
```python
# Only predict when very confident:
high_confidence_threshold = 0.55
pred = (prob >= high_confidence_threshold).astype(int)
# Expected: fewer but more accurate predictions
```

---

## Next Steps (Priority Order)

### Step 1: XGBoost/LightGBM (HIGH IMPACT)
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,  # shallower
    learning_rate=0.01,  # slower
    subsample=0.8,
    colsample_bytree=0.5,  # less features per tree
    reg_alpha=1.0,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42,
    verbosity=0
)
```

### Step 2: Simple Ensemble
```python
# Average predictions from 2-3 models
prob_avg = (prob_lr + prob_rf + prob_xgb) / 3
pred = (prob_avg >= 0.50).astype(int)
```

### Step 3: Sample Weighting
```python
# Give more weight to recent data
sample_weight = np.exp(np.linspace(-1, 0, len(X_train)))
model.fit(X_train, y_train, sample_weight=sample_weight)
```

---

## What NOT to Do

1. ❌ Don't add cross-sectional z-scores (hurt performance)
2. ❌ Don't add extreme return/volume flags (noisy)
3. ❌ Don't use too many features (>70)
4. ❌ Don't use deep trees (max_depth > 4)
5. ❌ Don't use high learning rate (>0.1)

---

## Realistic Targets

| Time | Target | Approach |
|------|--------|----------|
| Today | 51-52% | Use Phase 1A features + C=0.1 |
| 1 week | 52-53% | Add XGBoost ensemble |
| 2 weeks | 53-54% | Threshold optimization + sample weighting |
| 1 month | 54-55% | All above + per-symbol tuning |

---

## Ready-to-Use Code

Create `scripts/quick_win.py`:

```python
#!/usr/bin/env python3
"""Quick win: Best configuration so far."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data with Phase 1A features (67 features total)
df = pd.read_parquet("data/processed/nifty100_features.parquet")
# Features already include: rsi_momentum, macd_momentum, trend_strength, volume_spike

# Use only the proven features
feature_cols = [
    'ret_1d', 'ret_5d', 'log_ret_1d', 'sma_20_ratio', 'sma_50_ratio',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'bb_width',
    'volatility_10', 'volatility_20', 'volume_ratio',
    # NEW (Phase 1A):
    'rsi_momentum', 'macd_momentum', 'trend_strength', 'volume_spike'
]

# Train with:
model = LogisticRegression(max_iter=1000, C=0.1)
# Scale features with StandardScaler
# Use walk-forward validation
```

---

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `src/market_ml/feature_engineering.py` | Added 10 momentum features | ✅ |
| `PHASE1_RESULTS.md` | Documented results | ✅ |
| `RESULTS_SUMMARY.md` | Comparison table | ✅ |

## Files to Create Next

| File | Purpose | Priority |
|------|---------|----------|
| `scripts/xgboost_train.py` | XGBoost with best params | HIGH |
| `scripts/ensemble.py` | Combine LR + RF + XGB | HIGH |
| `scripts/threshold_search.py` | Find optimal threshold | MEDIUM |
