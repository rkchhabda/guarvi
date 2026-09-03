#!/usr/bin/env python3
"""Phase 2: Threshold optimization + feature selection + ensemble.

Walk-forward validation (same protocol as full_phase1c.py) with:
  1. XGBoost + Logistic Regression out-of-sample probabilities collected per window
  2. XGBoost gain importances aggregated across windows
  3. Honest threshold selection: sweep on first 60% of windows, evaluate on last 40%
  4. Ensemble = mean of XGB and LR probabilities
  5. Pass 2: re-run XGBoost with top-30 features only
No new advanced models are introduced.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

LOG = ROOT / "reports" / "phase2_run.log"
N_WINDOWS = 24
MAX_TRAIN_ROWS = 200_000


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def get_feats(df):
    exc = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source', 'target_direction']
    return [c for c in df.columns if c not in exc and df[c].dtype in ('float64', 'float32', 'int64')]


def load_data():
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_direction']).sort_values(['date', 'symbol']).reset_index(drop=True)
    return df


def make_windows(d, n_windows=N_WINDOWS):
    dates = np.sort(d['date'].unique())
    start = 500
    step = max(1, (len(dates) - start - 2) // n_windows)
    out = []
    for i in range(start, len(dates) - 1, step):
        trn = d.index[d['date'] <= dates[i]]
        tst = d.index[d['date'] == dates[i + 1]]
        if len(trn) < 500 or len(tst) == 0:
            continue
        out.append((trn, tst))
    return out


def fit_predict(trn, tst, feats, model_type):
    """Train on trn, return test probabilities, labels and (xgb) gain importances."""
    imp = SimpleImputer(strategy='median')
    X1 = imp.fit_transform(trn[feats].values)
    X2 = imp.transform(tst[feats].values)
    y1 = trn['target_direction'].values
    y2 = tst['target_direction'].values
    importance = None
    if model_type == 'logistic':
        sc = StandardScaler()
        X1 = sc.fit_transform(X1)
        X2 = sc.transform(X2)
        m = LogisticRegression(max_iter=500, C=0.1).fit(X1, y1)
    else:
        m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              tree_method='hist', n_jobs=-1, random_state=42,
                              verbosity=0, importance_type='gain').fit(X1, y1)
        importance = m.feature_importances_
    p = m.predict_proba(X2)[:, 1]
    return p, y2, importance


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')
def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    log("=" * 60)
    log("PHASE 2: threshold optimization + feature selection + ensemble")
    log("=" * 60)

    df = load_data()
    feats = get_feats(df)
    log(f"rows={len(df):,}  features={len(feats)}")
    splits = make_windows(df)
    log(f"windows={len(splits)} (train cap {MAX_TRAIN_ROWS:,} rows)")

    cols = ['date', 'symbol', 'actual', 'prob_xgb', 'prob_lr', 'win']
    rows, imp_sum = [], pd.Series(0.0, index=feats)
    imp_n = 0
    t0 = time.time()
    for k, (ti, si) in enumerate(splits):
        trn, tst = df.loc[ti], df.loc[si]
        if len(trn) > MAX_TRAIN_ROWS:
            trn = trn.iloc[-MAX_TRAIN_ROWS:]
        px, y2, impv = fit_predict(trn, tst, feats, 'xgboost')
        pl, _, _ = fit_predict(trn, tst, feats, 'logistic')
        if impv is not None:
            imp_sum += pd.Series(impv, index=feats)
            imp_n += 1
        wdate = tst['date'].iloc[0]
        for sym, ax, bx, bl in zip(tst['symbol'].values, y2, px, pl):
            rows.append((wdate, sym, int(ax), float(bx), float(bl), k))
        log(f"window {k + 1}/{len(splits)}: date={wdate.date()} n_test={len(tst)} elapsed={time.time() - t0:.0f}s")
        pd.DataFrame(rows, columns=cols).to_csv(ROOT / "reports/phase2_predictions.csv", index=False)

    P = pd.DataFrame(rows, columns=cols)
    P['prob_ens'] = (P['prob_xgb'] + P['prob_lr']) / 2.0
    P['pred_xgb'] = (P['prob_xgb'] >= 0.5).astype(int)
    P['pred_lr'] = (P['prob_lr'] >= 0.5).astype(int)
    P['pred_ens'] = (P['prob_ens'] >= 0.5).astype(int)
    P.to_csv(ROOT / "reports/phase2_predictions.csv", index=False)

    def full_auc(d, col):
        dd = d[['actual', col]].dropna()
        return float(roc_auc_score(dd['actual'], dd[col]))

    naive = float((P['actual'] == 1).mean())
    res = {
        'n_windows': len(splits),
        'n_predictions': int(len(P)),
        'naive_accuracy': naive,
        'xgb_accuracy_050': float((P['pred_xgb'] == P['actual']).mean()),
        'lr_accuracy_050': float((P['pred_lr'] == P['actual']).mean()),
        'ensemble_accuracy_050': float((P['pred_ens'] == P['actual']).mean()),
        'xgb_auc': full_auc(P, 'prob_xgb'),
        'lr_auc': full_auc(P, 'prob_lr'),
        'ensemble_auc': full_auc(P, 'prob_ens'),
    }
# ---- honest threshold optimization: select on first 60% of windows ----
    n_sel = int(len(splits) * 0.6)
    sel_df, ev_df = P[P['win'] < n_sel], P[P['win'] >= n_sel]
    log(f"threshold selection windows 0..{n_sel - 1} ({len(sel_df):,} preds); "
        f"evaluation windows {n_sel}..{len(splits) - 1} ({len(ev_df):,} preds)")

    def sweep(d, col):
        best_thr, best_acc = 0.5, -1.0
        for thr in np.arange(0.40, 0.601, 0.005):
            a = float(((d[col] >= thr).astype(int) == d['actual']).mean())
            if a > best_acc:
                best_thr, best_acc = float(thr), a
        return best_thr, best_acc

    thr_res = {}
    for col in ['prob_xgb', 'prob_ens']:
        bt, ba = sweep(sel_df, col)
        ev_bt = float(((ev_df[col] >= bt).astype(int) == ev_df['actual']).mean())
        ev_50 = float(((ev_df[col] >= 0.5).astype(int) == ev_df['actual']).mean())
        thr_res[col] = {'selected_threshold': bt, 'selection_accuracy': ba,
                        'eval_accuracy_at_selected': ev_bt, 'eval_accuracy_at_050': ev_50}
        log(f"{col}: thr*={bt:.3f} (sel acc {ba:.4f}) -> eval acc {ev_bt:.4f} (0.50 -> {ev_50:.4f})")
    res['threshold_optimization'] = thr_res

    # ---- confidence buckets (descriptive, all OOS) ----
    buckets = {}
    for lo, hi_b in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        m = (P['prob_ens'] >= lo) & (P['prob_ens'] < hi_b)
        if m.sum() > 0:
            buckets[f"ens_{lo:.2f}_{hi_b:.2f}"] = {
                'n': int(m.sum()),
                'accuracy': float((P.loc[m, 'pred_ens'] == P.loc[m, 'actual']).mean())}
    hi_conf = P[(P['prob_ens'] >= 0.6) | (P['prob_ens'] <= 0.4)]
    if len(hi_conf) > 0:
        buckets['ens_high_conf_ge_0.60_or_le_0.40'] = {
            'n': int(len(hi_conf)),
            'accuracy': float((hi_conf['pred_ens'] == hi_conf['actual']).mean())}
    res['confidence_buckets'] = buckets

    # ---- feature importances ----
    imp_mean = (imp_sum / max(1, imp_n)).sort_values(ascending=False)
    imp_mean.to_csv(ROOT / "reports/phase2_feature_importance.csv", header=['gain_importance'])
    top30 = list(imp_mean.head(30).index)
    res['top30_features'] = top30
    log(f"top-10 features: {top30[:10]}")
# ---- pass 2: XGBoost with top-30 features only ----
    log("pass 2: XGBoost with top-30 features...")
    rows2, accs2 = [], []
    t2 = time.time()
    for k, (ti, si) in enumerate(splits):
        trn, tst = df.loc[ti], df.loc[si]
        if len(trn) > MAX_TRAIN_ROWS:
            trn = trn.iloc[-MAX_TRAIN_ROWS:]
        p, y2, _ = fit_predict(trn, tst, top30, 'xgboost')
        pred = (p >= 0.5).astype(int)
        accs2.append(float((pred == y2).mean()))
        wdate = tst['date'].iloc[0]
        for sym, ax, bx in zip(tst['symbol'].values, y2, p):
            rows2.append((wdate, sym, int(ax), float(bx)))
        if (k + 1) % 6 == 0:
            log(f"  pass2 window {k + 1}/{len(splits)} elapsed={time.time() - t2:.0f}s")
    P2 = pd.DataFrame(rows2, columns=['date', 'symbol', 'actual', 'prob_xgb_top30'])
    P2.to_csv(ROOT / "reports/phase2_top30_predictions.csv", index=False)
    P2['pred'] = (P2['prob_xgb_top30'] >= 0.5).astype(int)
    res['xgb_top30_accuracy_050'] = float((P2['pred'] == P2['actual']).mean())
    res['xgb_top30_mean_window_acc'] = float(np.mean(accs2))
    res['xgb_top30_auc'] = full_auc(P2.rename(columns={'prob_xgb_top30': 'prob'}), 'prob')

    log(f"naive={naive:.4f}  xgb@0.5={res['xgb_accuracy_050']:.4f}  lr@0.5={res['lr_accuracy_050']:.4f}  "
        f"ens@0.5={res['ensemble_accuracy_050']:.4f}  xgb_top30@0.5={res['xgb_top30_accuracy_050']:.4f}")

    with open(ROOT / "reports/phase2_results.json", "w") as f:
        json.dump(res, f, indent=2)

    lines = [
        "# Phase 2 Summary (threshold + feature selection + ensemble)", "",
        f"- Walk-forward windows: {len(splits)} | OOS predictions: {len(P):,} | "
        f"Naive (always-up) accuracy: {naive:.4f}", "",
        "| Model | Accuracy | AUC |", "|---|---|---|",
        f"| XGBoost (62 feats) @0.50 | {res['xgb_accuracy_050']:.4f} | {res['xgb_auc']:.4f} |",
        f"| Logistic (62 feats) @0.50 | {res['lr_accuracy_050']:.4f} | {res['lr_auc']:.4f} |",
        f"| Ensemble (mean prob) @0.50 | {res['ensemble_accuracy_050']:.4f} | {res['ensemble_auc']:.4f} |",
        f"| XGBoost (top-30 feats) @0.50 | {res['xgb_top30_accuracy_050']:.4f} | {res['xgb_top30_auc']:.4f} |", "",
        "## Threshold optimization (select on first 60% windows, evaluate on last 40%)", "",
    ]
    for col, v in thr_res.items():
        lines.append(f"- **{col}**: selected thr={v['selected_threshold']:.3f} "
                     f"(selection acc {v['selection_accuracy']:.4f}) -> eval acc "
                     f"{v['eval_accuracy_at_selected']:.4f} (0.50 baseline {v['eval_accuracy_at_050']:.4f})")
    lines += ["", "## Confidence buckets (ensemble prob, all OOS)", ""]
    for name, v in buckets.items():
        lines.append(f"- {name}: n={v['n']:,} acc={v['accuracy']:.4f}")
    (ROOT / "reports/phase2_summary.md").write_text("\n".join(lines) + "\n")
    log("DONE - wrote reports/phase2_results.json, phase2_summary.md, phase2_predictions.csv, "
        "phase2_top30_predictions.csv, phase2_feature_importance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())