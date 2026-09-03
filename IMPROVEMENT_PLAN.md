# Improvement Plan: Boost Accuracy from 50.5% to 55%+

## Current Status
- Directional Accuracy: 50.48% 
- Always-up Baseline: 50.92%
- Model Edge: -0.44% (worse than naive baseline)
- Sharpe Ratio: 0.544 (before audit showed it's actually negative)

## Root Causes for Poor Performance
1. **Inherent noise**: Daily stock direction is difficult to predict from technicals alone
2. **Limited features**: Only 62 technical indicators, no alternative data
3. **Suboptimal model tuning**: Default parameters may not be ideal
4. **Feature quality**: Many features have high missing rates (SMA200: 20K+ missing)

## Phase 1 Improvements (Target: 50.5% → 52-53%)
### Immediate Actions (Week 1):

#### A. Feature Engineering Enhancements
1. **Volatility Regime Features** (Add 4-6 features)
   - Current volatility / 60-day average volatility 
   - Volatility percentile rank (last 252 days)
   - Volatility change (today vs yesterday)
   
2. **Volume-Price Relationship** (Add 3-4 features)
   - Volume-weighted price change
   - High-volume days indicator (volume > 2x average)
   - Price change on high vs low volume days
   
3. **Momentum Quality** (Add 3-4 features)
   - RSI divergence (price making new high/low, RSI not confirming)
   - MACD histogram momentum (rate of change)
   - Stochastic crossover strength
   
4. **Market Microstructure** (Add 2-3 features)
   - Intraday range (high-low)/close
   - Gap magnitude (open vs previous close)
   - Intraday reversal (close vs (high+low)/2)

#### B. Data Quality Improvements
1. **Handle Missing Features Better**
   - For SMA200/dist_sma_200: use available shorter MAs when long MA missing
   - Create "data availability" features (how many days of history we have)
   
2. **Outlier Treatment**
   - Winsorize returns at 1st/99th percentile
   - Filter extreme volume days (>5x median)
   
3. **Cross-sectional Normalization** 
   - Daily z-score normalization of features within universe
   - Helps compare stocks fairly regardless of price level

#### C. Model Improvements
1. **Algorithm Exploration**
   - Test LightGBM (often beats XGBoost on structured data)
   - Test CatBoot (handles categorical/time dependencies well)
   - Compare with tuned XGBoost/RF
   
2. **Hyperparameter Tuning**
   - Focus on: learning_rate, max_depth, subsample, colsample_bytree
   - Use time-series cross-validation for tuning
   
3. **Simple Ensemble**
   - Average predictions from top 3 models
   - Use validation performance to weight ensemble

#### D. Training Procedure Improvements
1. **Extended Training Window**
   - Increase minimum training history from 3 years to 4-5 years
   - More stable parameter estimates
   
2. **Sample Weighting**
   - Give more weight to recent data (exponential decay)
   - Market regimes change, recent data more relevant

## Implementation Order:

### Week 1-2: Feature Engineering + Data Quality
1. Implement volatility regime features
2. Add volume-price relationship features  
3. Improve missing data handling
4. Add cross-sectional normalization
5. Test impact on validation accuracy

### Week 3: Model Improvements
1. Implement LightGBM/CatBoost alternatives
2. Tune hyperparameters for top 2 models
3. Create simple ensemble
4. Test combined impact

### Week 4: Training Procedure + Final Push
1. Implement extended training window
2. Add sample weighting
3. Final validation and tuning
4. Document results

## Success Metrics:
- Target: 52-53% accuracy after Phase 1 (Week 1-2)
- Ultimate target: 55%+ after all phases
- Must beat naive baseline consistently
- Maintain or improve Sharpe ratio
- Keep drawdown reasonable (< 50% max)

## Risk Mitigation:
- All changes maintain strict walk-forward validation
- No future data leakage in new features
- Features tested on synthetic data first
- Baseline performance tracked throughout

Let me start implementing the feature engineering improvements first.