#!/usr/bin/env python3
"""Phase 1 Improvement Guide - Quick Reference"""
print("""
================================================================================
PHASE 1: BOOST ACCURACY FROM 50.48% TO 55%+
================================================================================

CURRENT STATUS:
- Model: 50.48% | Baseline: 50.92% | Edge: -0.44%
- This run Logistic Regression: 51.03% (+0.55% improvement)

================================================================================
4-PHASE STRATEGY
================================================================================

PHASE 1A: FEATURES (Week 1-2) → Target: 51-52%
- Add volatility regime features (6)
- Add volume-price features (5)
- Add momentum quality features (4)
- Add market microstructure (3)

PHASE 1B: DATA QUALITY (Week 1-2) → Target: 51-52%
- Winsorize returns at 1st/99th percentiles
- Filter extreme volume days
- Daily z-score normalization

PHASE 1C: MODEL TUNING (Week 2-3) → Target: 52-53%
- Try LightGBM/CatBoost
- Hyperparameter grid search
- Simple ensemble of top 3 models

PHASE 1D: TRAINING (Week 3-4) → Target: 53-55%
- Extend training window (3→5 years)
- Sample weighting (exponential decay)
- Regime detection

================================================================================
QUICK WINS (Implement Today)
================================================================================

Add to src/market_ml/feature_engineering.py:

# In _compute_volume or build_symbol_features:
df['rsi_momentum'] = df.groupby('symbol')['rsi_14'].diff()
df['macd_momentum'] = df.groupby('symbol')['macd_hist'].diff()
df['trend_strength'] = df['sma_20_ratio'] - df['sma_50_ratio']
df['volume_spike'] = (df['volume_ratio_20d'] > 2.0).astype(float)

================================================================================
REALISTIC TARGETS
================================================================================

Week 2:  51-52% (feature additions)
Week 4:  52-53% (model tuning)  
Week 8:  53-55% (full implementation)

================================================================================
""")