#!/usr/bin/env python3
"""Phase 1C: Full data validation with 62 advanced features."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def get_feats(df):
    exc = ['date','symbol','open','high','low','close','volume','source','target_direction']
    return [c for c in df.columns if c not in exc and df[c].dtype in ['float64','float32','int64']]

def train_quick(df, model_type, n_windows=30):
    """Train with rolling windows."""
    feats = get_feats(df)
    df = df.sort_values(['date','symbol']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction'])
    dates = np.sort(df['date'].unique())
    preds, accs, aucs = [], [], []
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
        else:
            m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                  n_jobs=-1, random_state=42, verbosity=0).fit(X1, y1)
        p = m.predict_proba(X2)[:,1]
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y2, pred))
        aucs.append(roc_auc_score(y2, p))
        for j,(_,r) in enumerate(tst.iterrows()):
            preds.append({'date':r['date'],'symbol':r['symbol'],'pred':int(pred[j]),'actual':int(y2[j]),'prob':float(p[j])})
    return {'preds':pd.DataFrame(preds),'acc':np.mean(accs),'std':np.std(accs),'auc':np.mean(aucs),'n':len(accs)}

def main():
    print("="*60)
    print("PHASE 1C: FULL DATA VALIDATION (62 features)")
    print("="*60)
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    print(f"Loaded {len(df):,} rows")
    feats = get_feats(df)
    print(f"Features: {len(feats)}")
    
    results = {}
    for mt in ['logistic', 'xgboost']:
        print(f"\nTraining {mt} on FULL data...")
        r = train_quick(df, mt, n_windows=30)
        results[mt] = r
        print(f"  Accuracy: {r['acc']:.4f} (±{r['std']:.4f}) AUC: {r['auc']:.4f} n={r['n']}")
    
    all_p = pd.concat([r['preds'] for r in results.values()])
    naive = float((all_p['actual'] == 1).mean())
    print(f"\nNaive (always up): {naive:.4f}")
    
    best = max(results.items(), key=lambda x: x[1]['acc'])
    print(f"Best: {best[0]} at {best[1]['acc']:.4f}")
    print(f"Edge: {best[1]['acc']-naive:+.4f}")
    
    # Save
    best[1]['preds'].to_csv(ROOT / "reports/phase1c_predictions.csv", index=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
