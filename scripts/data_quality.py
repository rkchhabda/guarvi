#!/usr/bin/env python3
"""Phase 1B: Data Quality Improvements.
- Winsorize returns at 1st/99th percentiles
- Filter extreme volume days
- Add z-score normalization
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def add_data_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add data quality features and handle outliers."""
    df = df.copy()
    
    # 1. Returns winsorization (already in actual_return, but add flags)
    if 'ret_1d' in df.columns:
        # Flag extreme returns
        p01 = df.groupby('symbol')['ret_1d'].quantile(0.01)
        p99 = df.groupby('symbol')['ret_1d'].quantile(0.99)
        df['extreme_return_up'] = df.apply(
            lambda r: 1 if r['ret_1d'] > p99.get(r['symbol'], np.inf) else 0, axis=1)
        df['extreme_return_down'] = df.apply(
            lambda r: 1 if r['ret_1d'] < p01.get(r['symbol'], -np.inf) else 0, axis=1)
        
        # Z-score normalization within symbol
        sym_mean = df.groupby('symbol')['ret_1d'].transform('mean')
        sym_std = df.groupby('symbol')['ret_1d'].transform('std')
        df['ret_zscore'] = (df['ret_1d'] - sym_mean) / sym_std.replace(0, 1)
    
    # 2. Volume filtering
    if 'volume_ratio_20d' in df.columns:
        # Flag extreme volume
        df['extreme_volume'] = (df['volume_ratio_20d'] > 5.0).astype(float)
        df['low_volume'] = (df['volume_ratio_20d'] < 0.2).astype(float)
        
        # Volume z-score
        vol_mean = df.groupby('symbol')['volume_ratio_20d'].transform('mean')
        vol_std = df.groupby('symbol')['volume_ratio_20d'].transform('std')
        df['vol_zscore'] = (df['volume_ratio_20d'] - vol_mean) / vol_std.replace(0, 1)
    
    # 3. Cross-sectional z-score (normalize by universe on each date)
    numeric_cols = ['ret_1d', 'rsi_14', 'macd', 'bb_position', 'volatility_20d']
    for col in numeric_cols:
        if col in df.columns:
            date_mean = df.groupby('date')[col].transform('mean')
            date_std = df.groupby('date')[col].transform('std')
            df[f'{col}_cs_zscore'] = (df[col] - date_mean) / date_std.replace(0, 1)
    
    return df

def get_features(df):
    """Get all numeric features."""
    exc = ['date','symbol','open','high','low','close','volume','source','target_direction']
    return [c for c in df.columns if c not in exc and 
            df[c].dtype in ['float64','float32','int64','int32']]

def train_and_evaluate(df, features, n_init=500, test_sz=21):
    """Train with walk-forward validation."""
    df = df.sort_values(['date','symbol']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction'])
    dates = np.sort(df['date'].unique())
    
    preds, accs = [], []
    for i in range(n_init, len(dates)-test_sz, test_sz):
        trn = df[df['date'] <= dates[i]]
        tst = df[(df['date'] > dates[i]) & (df['date'] <= dates[i+1])]
        if len(trn) < n_init or len(tst) == 0: continue
        
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(trn[features].values)
        X_ts = imp.transform(tst[features].values)
        y_tr, y_ts = trn['target_direction'].values, tst['target_direction'].values
        
        mdl = LogisticRegression(max_iter=1000, C=0.1).fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_ts)[:,1]
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_ts, pred)
        accs.append(acc)
        
        for j,(_,r) in enumerate(tst.iterrows()):
            preds.append({'date':r['date'],'symbol':r['symbol'],
                         'pred':int(pred[j]),'actual':int(y_ts[j]),'prob':float(prob[j])})
    
    return {'preds':pd.DataFrame(preds),'acc':np.mean(accs),'std':np.std(accs),'n':len(accs)}

def main():
    print("="*60)
    print("PHASE 1B: DATA QUALITY IMPROVEMENTS")
    print("="*60)
    
    # Load data
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    print(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    
    # Add data quality features
    print("\nAdding data quality features...")
    df = add_data_quality_features(df)
    feats = get_features(df)
    print(f"Total features: {len(feats)}")
    
    # Train
    print("\nTraining Logistic Regression...")
    result = train_and_evaluate(df, feats)
    print(f"Accuracy: {result['acc']:.4f} (±{result['std']:.4f})")
    
    # Compare with baseline
    naive = float((result['preds']['actual'] == 1).mean())
    print(f"Naive baseline: {naive:.4f}")
    print(f"Edge vs naive: {result['acc'] - naive:+.4f}")
    
    # Save
    result['preds'].to_csv(ROOT / "reports/phase1b_predictions.csv", index=False)
    
    # Feature importance (top 20)
    print("\n" + "="*60)
    print("NEW DATA QUALITY FEATURES ADDED:")
    print("="*60)
    new_feats = ['extreme_return_up','extreme_return_down','ret_zscore',
                'extreme_volume','low_volume','vol_zscore']
    for f in new_feats:
        if f in feats:
            print(f"  - {f}")
    
    cs_feats = [c for c in feats if '_cs_zscore' in c]
    for f in cs_feats:
        print(f"  - {f}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())