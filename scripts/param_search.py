#!/usr/bin/env python3
"""Phase 1C: Hyperparameter and Threshold Optimization.
Quick wins: Test different C values and prediction thresholds.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def add_features(df):
    """Add Phase 1A features only (proven to help)."""
    df = df.copy()
    if 'rsi_14' in df.columns:
        df['rsi_mom'] = df.groupby('symbol')['rsi_14'].diff()
    if 'macd_hist' in df.columns:
        df['macd_mom'] = df.groupby('symbol')['macd_hist'].diff()
    if 'sma_20_ratio' in df.columns and 'sma_50_ratio' in df.columns:
        df['trend'] = df['sma_20_ratio'] - df['sma_50_ratio']
    if 'volume_ratio_20d' in df.columns:
        df['vol_spike'] = (df['volume_ratio_20d'] > 2.0).astype(float)
    return df

def get_features(df):
    exc = ['date','symbol','open','high','low','close','volume','source','target_direction']
    return [c for c in df.columns if c not in exc and 
            df[c].dtype in ['float64','float32','int64','int32']]

def train_fold(df, features, C, threshold, n_init=500, test_sz=21):
    """Single walk-forward fold."""
    df = df.sort_values(['date','symbol']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction'])
    dates = np.sort(df['date'].unique())
    
    accs = []
    for i in range(n_init, len(dates)-test_sz, test_sz):
        trn = df[df['date'] <= dates[i]]
        tst = df[(df['date'] > dates[i]) & (df['date'] <= dates[i+1])]
        if len(trn) < n_init or len(tst) == 0: continue
        
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(trn[features].values)
        X_ts = imp.transform(tst[features].values)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_ts = sc.transform(X_ts)
        
        y_tr, y_ts = trn['target_direction'].values, tst['target_direction'].values
        mdl = LogisticRegression(max_iter=1000, C=C).fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_ts)[:,1]
        pred = (prob >= threshold).astype(int)
        accs.append(accuracy_score(y_ts, pred))
    
    return np.mean(accs) if accs else 0.0

def main():
    print("="*60)
    print("PHASE 1C: HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Load data
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    print(f"Loaded {len(df):,} rows")
    
    df = add_features(df)
    feats = get_features(df)
    print(f"Features: {len(feats)}")
    
    # Grid search
    C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    thresholds = [0.45, 0.48, 0.50, 0.52, 0.55]
    
    print("\nGrid search: C values x Thresholds...")
    print("-"*60)
    
    results = []
    for C in C_values:
        for thresh in thresholds:
            acc = train_fold(df, feats, C, thresh)
            results.append({'C': C, 'threshold': thresh, 'accuracy': acc})
    
    # Sort by accuracy
    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    print("\nTop 10 configurations:")
    print("-"*60)
    for i, row in results_df.head(10).iterrows():
        edge = row['accuracy'] - 0.5092  # naive baseline
        print(f"C={row['C']:6.3f}, thresh={row['threshold']:.2f} -> {row['accuracy']:.4f} (edge: {edge:+.4f})")
    
    # Best config
    best = results_df.iloc[0]
    print(f"\n*** BEST: C={best['C']}, threshold={best['threshold']} -> {best['accuracy']:.4f} ***")
    print(f"    Edge vs naive: {best['accuracy']-0.5092:+.4f}")
    
    # Save results
    results_df.to_csv(ROOT / "reports/hyperparam_search.csv", index=False)
    
    # Quick summary: effect of threshold alone
    print("\n" + "="*60)
    print("THRESHOLD EFFECT (averaged across C values):")
    print("-"*60)
    for thresh in thresholds:
        avg = results_df[results_df['threshold'] == thresh]['accuracy'].mean()
        edge = avg - 0.5092
        print(f"Threshold {thresh:.2f}: avg accuracy {avg:.4f} (edge: {edge:+.4f})")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())