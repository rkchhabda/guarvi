#!/usr/bin/env python3
"""XGBoost hyperparameter tuning - find best config."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed")
    exit(1)

def add_features(df):
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

def quick_eval(df, feats, params, n_init=500, test_sz=21):
    """Quick evaluation with fewer folds."""
    df = df.sort_values(['date','symbol']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction'])
    dates = np.sort(df['date'].unique())
    
    # Only use every 3rd fold for speed
    accs = []
    for i in range(n_init, len(dates)-test_sz, test_sz*3):
        trn = df[df['date'] <= dates[i]]
        tst = df[(df['date'] > dates[i]) & (df['date'] <= dates[i+1])]
        if len(trn) < n_init or len(tst) == 0: continue
        
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(trn[feats].values)
        X_ts = imp.transform(tst[feats].values)
        y_tr, y_ts = trn['target_direction'].values, tst['target_direction'].values
        
        mdl = XGBClassifier(**params, random_state=42, verbosity=0, n_jobs=-1)
        mdl.fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_ts)[:,1]
        pred = (prob >= 0.50).astype(int)
        accs.append(accuracy_score(y_ts, pred))
    
    return np.mean(accs) if accs else 0.0

def main():
    print("="*50)
    print("XGBoost Hyperparameter Search")
    print("="*50)
    
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    df = add_features(df)
    feats = get_features(df)
    print(f"Features: {len(feats)}")
    
    # Grid
    configs = [
        # Vary max_depth
        {'name': 'depth=2', 'n_estimators':100, 'max_depth':2, 'learning_rate':0.01, 'colsample_bytree':0.5, 'reg_alpha':1.0, 'reg_lambda':1.0},
        {'name': 'depth=3', 'n_estimators':100, 'max_depth':3, 'learning_rate':0.01, 'colsample_bytree':0.5, 'reg_alpha':1.0, 'reg_lambda':1.0},
        {'name': 'depth=4', 'n_estimators':100, 'max_depth':4, 'learning_rate':0.01, 'colsample_bytree':0.5, 'reg_alpha':1.0, 'reg_lambda':1.0},
        # Vary learning_rate
        {'name': 'lr=0.005', 'n_estimators':200, 'max_depth':3, 'learning_rate':0.005, 'colsample_bytree':0.5, 'reg_alpha':1.0, 'reg_lambda':1.0},
        {'name': 'lr=0.02', 'n_estimators':100, 'max_depth':3, 'learning_rate':0.02, 'colsample_bytree':0.5, 'reg_alpha':1.0, 'reg_lambda':1.0},
        # Vary colsample
        {'name': 'col=0.3', 'n_estimators':100, 'max_depth':3, 'learning_rate':0.01, 'colsample_bytree':0.3, 'reg_alpha':1.0, 'reg_lambda':1.0},
        {'name': 'col=0.7', 'n_estimators':100, 'max_depth':3, 'learning_rate':0.01, 'colsample_bytree':0.7, 'reg_alpha':1.0, 'reg_lambda':1.0},
        # More regularization
        {'name': 'reg=2.0', 'n_estimators':100, 'max_depth':3, 'learning_rate':0.01, 'colsample_bytree':0.5, 'reg_alpha':2.0, 'reg_lambda':2.0},
    ]
    
    print(f"\nTesting {len(configs)} configurations...")
    results = []
    for cfg in configs:
        acc = quick_eval(df, feats, {k: v for k, v in cfg.items() if k != 'name'})
        results.append({'name': cfg['name'], 'accuracy': acc})
        print(f"  {cfg['name']:15s}: {acc:.4f}")
    
    # Sort
    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 3 CONFIGURATIONS")
    print("="*50)
    for _, row in results_df.head(3).iterrows():
        print(f"{row['name']:15s}: {row['accuracy']:.4f}")
    
    best = results_df.iloc[0]
    print(f"\n*** BEST: {best['name']} at {best['accuracy']:.4f} ***")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())