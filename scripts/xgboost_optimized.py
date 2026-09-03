#!/usr/bin/env python3
"""Quick win: XGBoost with optimized hyperparameters."""
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
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")
    exit(1)

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

def train_xgb(df, features, n_init=500, test_sz=21):
    """Train XGBoost with walk-forward."""
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
        
        # OPTIMIZED XGBoost params
        mdl = XGBClassifier(
            n_estimators=100,
            max_depth=3,  # shallower
            learning_rate=0.01,  # slower
            subsample=0.8,
            colsample_bytree=0.5,  # less features per tree
            reg_alpha=1.0,  # L1
            reg_lambda=1.0,  # L2
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        mdl.fit(X_tr, y_tr)
        
        prob = mdl.predict_proba(X_ts)[:,1]
        pred = (prob >= 0.50).astype(int)
        acc = accuracy_score(y_ts, pred)
        accs.append(acc)
        
        for j,(_,r) in enumerate(tst.iterrows()):
            preds.append({'date':r['date'],'symbol':r['symbol'],
                         'pred':int(pred[j]),'actual':int(y_ts[j]),'prob':float(prob[j])})
    
    return {'preds':pd.DataFrame(preds),'acc':np.mean(accs),'std':np.std(accs),'n':len(accs)}

def main():
    print("="*50)
    print("XGBoost OPTIMIZED")
    print("="*50)
    
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    print(f"Loaded {len(df):,} rows")
    
    df = add_features(df)
    feats = get_features(df)
    print(f"Features: {len(feats)}")
    
    print("\nTraining XGBoost...")
    result = train_xgb(df, feats)
    
    print(f"\n*** XGBoost Accuracy: {result['acc']:.4f} (±{result['std']:.4f}) ***")
    
    naive = float((result['preds']['actual'] == 1).mean())
    print(f"Naive baseline: {naive:.4f}")
    print(f"Edge: {result['acc'] - naive:+.4f}")
    
    result['preds'].to_csv(ROOT / "reports/xgboost_predictions.csv", index=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())