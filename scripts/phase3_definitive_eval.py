#!/usr/bin/env python3
"""Phase 3: Definitive low-noise walk-forward evaluation.

Continuous walk-forward: retrain every RETRAIN_EVERY trading days, predict EVERY
day in between -> ~150K OOS predictions (SE ~0.13%) instead of scattered windows.
Includes audited-methodology portfolio backtest (equal-weight, exposure<=1,
10bps costs, Sharpe sqrt(252), MDD in [-1,0]) vs the naive always-up portfolio.
No new advanced models.
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

LOG = ROOT / "reports" / "phase3_run.log"
RETRAIN_EVERY = 42
START_DAYS = 500
MAX_TRAIN_ROWS = 200_000
COST_RATE = 0.001  # 10 bps per unit one-way turnover


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def get_feats(df):
    exc = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source',
           'target_direction', 'next_ret']
    return [c for c in df.columns if c not in exc and df[c].dtype in ('float64', 'float32', 'int64')]


def load_data():
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    # realized next-day return (evaluation only, never a feature)
    df['next_ret'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1.0
    # same bad-data filter as the original pipeline: |ret| > 50% are almost
    # always corporate actions (splits/demergers) mispriced in adjusted data
    n_bad = int((df['next_ret'].abs() > 0.5).sum())
    df = df[df['next_ret'].abs() <= 0.5]
    df = df.dropna(subset=['target_direction', 'next_ret'])
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    print(f"excluded {n_bad} rows with |next_ret|>0.5 (bad price data)", flush=True)
    return df


def fit_models(trn, feats):
    """Fit XGB + LR on trn. Returns (xgb_model, lr_model, imputer, scaler)."""
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(trn[feats].values)
    y = trn['target_direction'].values
    m_x = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                            tree_method='hist', n_jobs=-1, random_state=42,
                            verbosity=0, importance_type='gain').fit(X, y)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m_l = LogisticRegression(max_iter=500, C=0.1).fit(Xs, y)
    return m_x, m_l, imp, sc


def predict_block(m_x, m_l, imp, sc, tst, feats):
    X = imp.transform(tst[feats].values)
    px = m_x.predict_proba(X)[:, 1]
    pl = m_l.predict_proba(sc.transform(X))[:, 1]
    return px, pl
def walk_forward(df, feats):
    """Continuous walk-forward: retrain every RETRAIN_EVERY days, predict all days in gap."""
    dates = pd.DatetimeIndex(np.sort(df['date'].unique()))
    dmap = {d: i for i, d in enumerate(dates)}
    date_idx = df['date'].map(dmap).values
    cols = ['date', 'symbol', 'actual', 'next_ret', 'prob_xgb', 'prob_lr', 'block']
    rows = []
    n_blocks = 0
    t0 = time.time()
    for cut in range(START_DAYS, len(dates) - 1, RETRAIN_EVERY):
        trn_mask = date_idx <= cut
        tst_mask = (date_idx > cut) & (date_idx <= min(cut + RETRAIN_EVERY, len(dates) - 1))
        trn = df[trn_mask]
        tst = df[tst_mask]
        if len(trn) < 500 or len(tst) == 0:
            continue
        if len(trn) > MAX_TRAIN_ROWS:
            trn = trn.iloc[-MAX_TRAIN_ROWS:]
        m_x, m_l, imp, sc = fit_models(trn, feats)
        px, pl = predict_block(m_x, m_l, imp, sc, tst, feats)
        for d, s, a, r, bx, bl in zip(tst['date'].values, tst['symbol'].values,
                                      tst['target_direction'].values, tst['next_ret'].values,
                                      px, pl):
            rows.append((d, s, int(a), float(r), float(bx), float(bl), n_blocks))
        n_blocks += 1
        log(f"block {n_blocks}: cutoff={dates[cut].date()} train={len(trn):,} "
            f"test_days={tst['date'].nunique()} preds={len(tst):,} elapsed={time.time() - t0:.0f}s")
        pd.DataFrame(rows, columns=cols).to_csv(ROOT / "reports/phase3_predictions.csv", index=False)
    return pd.DataFrame(rows, columns=cols), n_blocks
def portfolio_backtest(P, prob_col, pred_col, cost_rate=COST_RATE):
    """Audited methodology: equal-weight daily portfolio, exposure<=1, costs on turnover."""
    P = P.sort_values(['date', 'symbol']).copy()
    if prob_col is not None:
        P[pred_col] = (P[prob_col] >= 0.5).astype(int)
    act = P[P[pred_col] == 1]
    # daily equal-weight strategy return across active signals
    g = act.groupby('date')['next_ret']
    daily = pd.DataFrame({'n_active': g.size(), 'gross': g.mean()})
    all_dates = np.sort(P['date'].unique())
    daily = daily.reindex(all_dates)
    daily['n_active'] = daily['n_active'].fillna(0).astype(int)
    daily['gross'] = daily['gross'].fillna(0.0)
    # turnover: 0.5 * sum |w_t - w_{t-1}| using weight vectors over full universe
    piv = P.pivot_table(index='date', columns='symbol', values=pred_col, aggfunc='max').reindex(all_dates).fillna(0.0)
    n_act = piv.sum(axis=1).replace(0, 1)
    W = piv.div(n_act, axis=0)  # weights sum to 1 when active, 0 when in cash
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    daily['turnover'] = turnover.values
    daily['cost'] = cost_rate * daily['turnover']
    daily['net'] = daily['gross'] - daily['cost']
    # safety assertions (from backtest audit)
    assert (P['next_ret'].abs() <= 0.5 + 1e-12).all(), "per-symbol |return| > 0.5 in portfolio"
    assert np.isfinite(daily['net']).all(), "NaN/inf in daily net returns"
    assert (daily['net'].abs() <= 0.5 + 1e-12).all(), "absurd daily portfolio return (bad data?)"
    assert (daily['net'] > -1).all(), "daily return <= -1"
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all(), "daily exposure exceeds 1.0"
    eq = (1.0 + daily['net']).cumprod()
    assert np.isfinite(eq).all() and (eq > 0).all(), "equity not finite/positive"
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min())
    assert -1.0 <= mdd <= 0.0, "MDD outside [-1, 0]"
    sd = daily['net'].std(ddof=0)
    sharpe = float(daily['net'].mean() / sd * np.sqrt(252)) if sd > 0 else float('nan')
    return {
        'cumulative_return_net': float(eq.iloc[-1] - 1.0),
        'cumulative_return_gross': float((1.0 + daily['gross']).prod() - 1.0),
        'max_drawdown': mdd,
        'sharpe_net': sharpe,
        'final_equity': float(eq.iloc[-1]),
        'n_days': int(len(daily)),
        'n_trades': int(len(act)),
        'n_active_days': int((daily['n_active'] > 0).sum()),
        'total_turnover': float(daily['turnover'].sum()),
        'avg_trade_return': float(act['next_ret'].mean()),
        'hit_rate': float((act[pred_col] == act['actual']).mean()),
        'daily_ret_min': float(daily['net'].min()),
        'daily_ret_max': float(daily['net'].max()),
    }, daily
def main():
    if LOG.exists():
        LOG.unlink()
    log("=" * 60)
    log("PHASE 3: DEFINITIVE CONTINUOUS WALK-FORWARD EVALUATION")
    log("=" * 60)
    df = load_data()
    feats = get_feats(df)
    dates = pd.DatetimeIndex(np.sort(df['date'].unique()))
    log(f"rows={len(df):,} features={len(feats)} dates={len(dates)} "
        f"span={dates[0].date()}..{dates[-1].date()} retrain_every={RETRAIN_EVERY}")

    # resume: reuse completed predictions if available
    pred_path = ROOT / "reports/phase3_predictions.csv"
    if pred_path.exists():
        P = pd.read_csv(pred_path)
        P['date'] = pd.to_datetime(P['date'])
        n_bad = int((P['next_ret'].abs() > 0.5).sum())
        P = P[P['next_ret'].abs() <= 0.5]
        log(f"resume: loaded {len(P):,} predictions (excluded {n_bad} rows with |next_ret|>0.5)")
        if len(P) > 100_000:
            log(f"resume: loaded {len(P):,} existing predictions, skipping walk-forward")
            n_blocks = int(P['block'].max()) + 1
        else:
            P, n_blocks = walk_forward(df, feats)
    else:
        P, n_blocks = walk_forward(df, feats)
    P['prob_ens'] = (P['prob_xgb'] + P['prob_lr']) / 2.0
    P.to_csv(ROOT / "reports/phase3_predictions.csv", index=False)

    n = len(P)
    naive_acc = float((P['actual'] == 1).mean())
    se = float(np.sqrt(0.25 / n))
    res = {'n_blocks': n_blocks, 'n_predictions': int(n), 'n_dates': int(P['date'].nunique()),
           'naive_accuracy': naive_acc, 'binomial_se': se, 'span': [str(dates[0].date()), str(dates[-1].date())]}
    for name, col in [('xgb', 'prob_xgb'), ('lr', 'prob_lr'), ('ens', 'prob_ens')]:
        pred = (P[col] >= 0.5).astype(int)
        acc = float((pred == P['actual']).mean())
        z = (acc - 0.5) / se
        res[f'{name}_accuracy_050'] = acc
        res[f'{name}_z_vs_coin'] = float(z)
        res[f'{name}_auc'] = float(roc_auc_score(P['actual'], P[col]))
        res[f'{name}_edge_vs_naive'] = acc - naive_acc
        log(f"{name}: acc={acc:.4f} (z={z:+.1f} vs coin) auc={res[f'{name}_auc']:.4f} "
            f"edge_vs_naive={acc - naive_acc:+.4f}")

    # ---- yearly breakdown ----
    P['year'] = pd.to_datetime(P['date']).dt.year
    yearly = {}
    for y, g in P.groupby('year'):
        yp = (g['prob_xgb'] >= 0.5).astype(int)
        yearly[str(y)] = {'n': int(len(g)),
                          'xgb_acc': float((yp == g['actual']).mean()),
                          'naive_acc': float((g['actual'] == 1).mean())}
    res['yearly'] = yearly

    # ---- high-confidence buckets (big sample) ----
    buckets = {}
    for lo, hi_b in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01)]:
        m = (P['prob_xgb'] >= lo) & (P['prob_xgb'] < hi_b)
        if m.sum() > 0:
            buckets[f"xgb_{lo:.2f}_{hi_b:.2f}"] = {'n': int(m.sum()),
                                                   'acc': float(((P.loc[m, 'prob_xgb'] >= 0.5).astype(int) == P.loc[m, 'actual']).mean())}
    res['confidence_buckets_xgb'] = buckets

    # ---- portfolio backtests (audited methodology) ----
    bt_x, daily_x = portfolio_backtest(P, 'prob_xgb', 'pred_xgb')
    daily_x.to_csv(ROOT / "reports/phase3_daily_equity_xgb.csv")
    # naive always-up portfolio: hold every symbol equally every day
    Pn = P.copy()
    Pn['pred_naive'] = 1
    bt_n, daily_n = portfolio_backtest(Pn, None, 'pred_naive')
    daily_n.to_csv(ROOT / "reports/phase3_daily_equity_naive.csv")
    res['backtest_xgb'] = bt_x
    res['backtest_naive'] = bt_n
    log(f"backtest xgb: cum={bt_x['cumulative_return_net']:+.4f} mdd={bt_x['max_drawdown']:.4f} "
        f"sharpe={bt_x['sharpe_net']:.2f} trades={bt_x['n_trades']:,}")
    log(f"backtest naive: cum={bt_n['cumulative_return_net']:+.4f} mdd={bt_n['max_drawdown']:.4f} "
        f"sharpe={bt_n['sharpe_net']:.2f}")

    with open(ROOT / "reports/phase3_results.json", "w") as f:
        json.dump(res, f, indent=2)

    lines = ["# Phase 3: Definitive Walk-Forward Evaluation", "",
             f"- Span: {res['span'][0]} .. {res['span'][1]} | blocks: {n_blocks} | "
             f"OOS predictions: {n:,} | binomial SE: {se:.4f}", "",
             "| Model | Accuracy | AUC | Edge vs Naive | z vs Coin |", "|---|---|---|---|---|"]
    for name in ['xgb', 'lr', 'ens']:
        lines.append(f"| {name.upper()} @0.50 | {res[f'{name}_accuracy_050']:.4f} | "
                     f"{res[f'{name}_auc']:.4f} | {res[f'{name}_edge_vs_naive']:+.4f} | "
                     f"{res[f'{name}_z_vs_coin']:+.1f} |")
    lines.append(f"| Naive (always-up) | {naive_acc:.4f} | - | - | - |")
    lines += ["", "## Yearly accuracy", "", "| Year | N | XGB | Naive |", "|---|---|---|---|"]
    for y, v in yearly.items():
        lines.append(f"| {y} | {v['n']:,} | {v['xgb_acc']:.4f} | {v['naive_acc']:.4f} |")
    lines += ["", "## Portfolio backtest (net of 10bps, audited methodology)", "",
              f"- XGB strategy: cum {bt_x['cumulative_return_net']:+.4f}, MDD {bt_x['max_drawdown']:.4f}, "
              f"Sharpe {bt_x['sharpe_net']:.2f}, trades {bt_x['n_trades']:,}",
              f"- Naive buy-all: cum {bt_n['cumulative_return_net']:+.4f}, MDD {bt_n['max_drawdown']:.4f}, "
              f"Sharpe {bt_n['sharpe_net']:.2f}"]
    (ROOT / "reports/phase3_summary.md").write_text("\n".join(lines) + "\n")
    log("DONE - wrote reports/phase3_results.json, phase3_summary.md, predictions, equity curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())