# Phase 1 Improvement Results

## Current Status
- **Model Accuracy**: 50.48% (baseline run)
- **Naive Baseline**: 50.92%
- **Edge**: -0.44% (model WORSE than baseline)
- **Logistic Regression with new features**: **51.03%** (+0.55% improvement)

## Quick Wins Implemented
Added 10 new features to `src/market_ml/feature_engineering.py`:

1. **rsi_momentum** - RSI rate of change
2. **rsi_overbought** - Binary flag (RSI > 70)
3. **rsi_oversold** - Binary flag (RSI < 30)
4. **macd_momentum** - MACD histogram change
5. **trend_strength** - Short vs long-term trend difference
6. **volume_spike** - Binary flag (volume > 2x average)
7. **volume_momentum** - Volume ratio change
8. **mean_reversion** - Short vs medium-term return difference
9. **adx_strong** - Binary flag (ADX > 25)
10. **(+ existing features)**: 62 → 72 features

## Test Results
- All 61 tests pass
- No regressions
- Features integrate cleanly with existing pipeline

## Next Steps (Phase 1B-1D)

### Week 1-2: Data Quality (Target: 51-52%)
- Winsorize returns at 1st/99th percentiles
- Filter extreme volume days
- Daily z-score normalization

### Week 2-3: Model Tuning (Target: 52-53%)
- Add LightGBM support
- Hyperparameter grid search
- Simple ensemble of top 3 models

### Week 3-4: Training Procedure (Target: 53-55%)
- Extend training window (3→5 years)
- Sample weighting (exponential decay)
- Regime detection

## Files Modified
- `src/market_ml/feature_engineering.py` - Added 10 enhanced features

## Scripts Created
- `scripts/enhanced_train.py` - Enhanced training with new features
- `scripts/phase1_guide.py` - Improvement guide reference
- `scripts/audit_backtest.py` - Backtest audit and correction

## Expected Impact
- **Realistic**: 52-53% in 4 weeks
- **Stretch**: 55% in 8-10 weeks
- **Key**: Maintain walk-forward validation, no data leakage
