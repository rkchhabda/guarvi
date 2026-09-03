#!/usr/bin/env python3
"""Phase 1C: Fast training test with 64 advanced features."""
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
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def get_feats(df):
    exc = ['date','symbol','open','high','low','close','volume','source','target_direction']
    return [c for c in df.columns if c not in exc and df[c].dtype in ['float64','float32','int64']]

def train_quick(df, model_type, n_windows=20):
    """Faster training: use fewer windows, smaller model."""
    feats = get_feats(df)
    df = df.sort_values(['date','symbol']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction'])
    dates = np.sort(df['date'].unique())
    preds, accs = [], []
    # Use fewer, larger windows for speed
    step = max(1, (len(dates) - 500 - 21) // n_windows)
    for i in range(500, len(dates)-21, step):
        trn = df[df['date'] <= dates[i]]
        tst = df[(df['date'] > dates[i]) & (df['date'] <= dates[i+1])]
        if len(trn) < 500 or len(tst) == 0: continue
        imp = SimpleImputer(strategy='median')
        X1 = imp.fit_transform(trn[feats].values)
        X2 = imp.transform(tst[feats].values)
        y1, y2 = trn['target_direction'].values, tst['target_direction'].values
        if model_type == 'logistic':
            sc = StandardScaler()
            X1 = sc.fit_transform(X1); X2 = sc.transform(X2)
            m = LogisticRegression(max_iter=500, C=0.1).fit(X1, y1)
        else:  # xgboost
            m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  n_jobs=-1, random_state=42, verbosity=0).fit(X1, y1)
        p = m.predict_proba(X2)[:,1]
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y2, pred))
    return np.mean(accs), np.std(accs), len(accs)

def main():
    print("="*50)
    print("PHASE 1C: FAST TRAINING (64 features)")
    print("="*50)
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    print(f"Loaded {len(df):,} rows")
    feats = get_feats(df)
    print(f"Features: {len(feats)}")
    print(f"New features: bb_squeeze, stoch_cross, vol_regime, trend_alignment, dist_52w_high/low, etc.")
    
    # Sample to speed up
    df_sample = df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampled to: {len(df_sample):,} rows")
    
    for mt in ['logistic', 'xgboost']:
        print(f"\nTraining {mt}...")
        acc, std, n = train_quick(df_sample, mt, n_windows=15)
        print(f"  Accuracy: {acc:.4f} (±{std:.4f}) n={n}")
    
    print("\nNote: Random Forest was too slow; XGBoost is the better choice for non-linear patterns.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
