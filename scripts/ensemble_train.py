#!/usr/bin/env python3
"""Ensemble: Combine LR + XGBoost for better accuracy."""
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

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def add_features(df):
    """Add proven Phase 1A features."""
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

def train_ensemble(df, features, n_init=500, test_sz=21):
    """Train ensemble with walk-forward."""
    df = df.sort_values(['date','symbol']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction'])
    dates = np.sort(df['date'].unique())
    
    preds_lr, preds_xgb, preds_ens, accs_lr, accs_xgb, accs_ens = [], [], [], [], [], []
    
    for i in range(n_init, len(dates)-test_sz, test_sz):
        trn = df[df['date'] <= dates[i]]
        tst = df[(df['date'] > dates[i]) & (df['date'] <= dates[i+1])]
        if len(trn) < n_init or len(tst) == 0: continue
        
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(trn[features].values)
        X_ts = imp.transform(tst[features].values)
        y_tr, y_ts = trn['target_direction'].values, tst['target_direction'].values
        
        # LR
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_ts_sc = sc.transform(X_ts)
        lr = LogisticRegression(max_iter=1000, C=0.1).fit(X_tr_sc, y_tr)
        prob_lr = lr.predict_proba(X_ts_sc)[:,1]
        
        # XGB
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01,
                            subsample=0.8, colsample_bytree=0.5, reg_alpha=1.0,
                            reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=-1)
        xgb.fit(X_tr, y_tr)
        prob_xgb = xgb.predict_proba(X_ts)[:,1]
        
        # Ensemble: simple average
        prob_ens = (prob_lr + prob_xgb) / 2
        
        # Try different thresholds
        for thresh in [0.48, 0.50, 0.52]:
            acc_lr = accuracy_score(y_ts, (prob_lr >= thresh).astype(int))
            acc_xgb = accuracy_score(y_ts, (prob_xgb >= thresh).astype(int))
            acc_ens = accuracy_score(y_ts, (prob_ens >= thresh).astype(int))
            
            if thresh == 0.50:
                accs_lr.append(acc_lr)
                accs_xgb.append(acc_xgb)
                accs_ens.append(acc_ens)
            
            for j,(_,r) in enumerate(tst.iterrows()):
                if thresh == 0.50:
                    preds_lr.append({'thresh':thresh,'pred':int((prob_lr[j]>=thresh)),'actual':int(y_ts[j])})
                    preds_xgb.append({'thresh':thresh,'pred':int((prob_xgb[j]>=thresh)),'actual':int(y_ts[j])})
                    preds_ens.append({'thresh':thresh,'pred':int((prob_ens[j]>=thresh)),'actual':int(y_ts[j])})
    
    return {
        'lr': {'acc': np.mean(accs_lr), 'std': np.std(accs_lr)},
        'xgb': {'acc': np.mean(accs_xgb), 'std': np.std(accs_xgb)},
        'ens': {'acc': np.mean(accs_ens), 'std': np.std(accs_ens)},
    }

def main():
    print("="*50)
    print("ENSEMBLE: LR + XGBoost")
    print("="*50)
    
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    print(f"Loaded {len(df):,} rows")
    
    df = add_features(df)
    feats = get_features(df)
    print(f"Features: {len(feats)}")
    
    print("\nTraining ensemble...")
    result = train_ensemble(df, feats)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Logistic Regression: {result['lr']['acc']:.4f} (±{result['lr']['std']:.4f})")
    print(f"XGBoost:            {result['xgb']['acc']:.4f} (±{result['xgb']['std']:.4f})")
    print(f"ENSEMBLE:           {result['ens']['acc']:.4f} (±{result['ens']['std']:.4f})")
    
    naive = 0.5092
    print(f"\nNaive baseline:     {naive:.4f}")
    print(f"Ensemble edge:      {result['ens']['acc'] - naive:+.4f}")
    
    # Best so far
    print("\n" + "="*50)
    print("PROGRESS")
    print("="*50)
    print(f"Baseline:           50.48%")
    print(f"Phase 1A (LR):     51.03%")
    print(f"XGBoost alone:      51.23%")
    print(f">>> ENSEMBLE:        {result['ens']['acc']*100:.2f}% <<<")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())