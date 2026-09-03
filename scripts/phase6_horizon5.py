#!/usr/bin/env python3
"""Phase 6: 5-day horizon labels — pre-registered fresh walk-forward.

PRE-REGISTRATION (declared before evaluation):
  Label: target_5d = close[t+5]/close[t] - 1 > 0 (per symbol); |5d ret| <= 0.5 filter.
  Model: XGB only, IDENTICAL hyperparameters to Phase 3 (150 trees, depth 4, lr 0.05,
  hist) — no new advanced models.
  Walk-forward: retrain every 42 trading days, predict every day, 500-day warmup,
  200K train-row cap (identical protocol to Phase 3).
  Designs (fixed a priori; judged ONLY on 2025-01-01..end holdout):
    F0 naive5_holdall_w5 : hold all tradable, weights frozen every 5 trading days
    F1 xgb5_binary_w5    : prob >= 0.5, frozen every 5 days, equal weight
    F2 xgb5_tercile_w5   : tercile tilt 1.5/1.0/0.5 by prob, frozen every 5 days
  PASS rule: F1 or F2 beats F0 on holdout net return AND Sharpe at BOTH 10 and 25 bps.
  Secondary (descriptive): accuracy vs 5d base rate with /sqrt(5) overlap-adjusted z,
  top-vs-bottom tercile mean 5d return spread, yearly breakdown.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

LOG = ROOT / "reports" / "phase6_run.log"
RETRAIN_EVERY = 42
START_DAYS = 500
MAX_TRAIN_ROWS = 200_000
H = 5  # horizon in trading days


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def get_feats(df):
    exc = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source',
           'target_direction', 'next_ret', 'next_ret_5d', 'target_5d']
    return [c for c in df.columns if c not in exc and df[c].dtype in ('float64', 'float32', 'int64')]


def load_data():
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol')['close']
    df['next_ret_5d'] = g.shift(-H) / df['close'] - 1.0
    df['next_ret_1d'] = g.shift(-1) / df['close'] - 1.0
    bad = (df['next_ret_5d'].abs() > 0.5) | (df['next_ret_1d'].abs() > 0.5)
    n_bad = int(bad.sum())
    df = df[~bad]
    df = df.dropna(subset=['next_ret_5d', 'next_ret_1d'])
    df['target_5d'] = (df['next_ret_5d'] > 0).astype(int)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    log(f"excluded {n_bad} rows with |ret|>0.5 (1d or 5d); rows={len(df):,} "
        f"base_rate={df['target_5d'].mean():.4f}")
    return df


def walk_forward(df, feats):
    dates = pd.DatetimeIndex(np.sort(df['date'].unique()))
    dmap = {d: i for i, d in enumerate(dates)}
    date_idx = df['date'].map(dmap).values
    cols = ['date', 'symbol', 'actual', 'ret_5d', 'ret_1d', 'prob']
    out_csv = ROOT / "reports/phase6_predictions.csv"
    rows, n_blocks = [], 0
    t0 = time.time()
    for cut in range(START_DAYS, len(dates) - 1, RETRAIN_EVERY):
        trn_mask = date_idx <= cut
        tst_mask = (date_idx > cut) & (date_idx <= min(cut + RETRAIN_EVERY, len(dates) - 1))
        trn, tst = df[trn_mask], df[tst_mask]
        if len(trn) < 500 or len(tst) == 0:
            continue
        if len(trn) > MAX_TRAIN_ROWS:
            trn = trn.iloc[-MAX_TRAIN_ROWS:]
        imp = SimpleImputer(strategy='median')
        X = imp.fit_transform(trn[feats].values)
        m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              tree_method='hist', n_jobs=-1, random_state=42,
                              verbosity=0).fit(X, trn['target_5d'].values)
        p = m.predict_proba(imp.transform(tst[feats].values))[:, 1]
        for d, s, a, r5, r1, pp in zip(tst['date'].values, tst['symbol'].values,
                                       tst['target_5d'].values, tst['next_ret_5d'].values,
                                       tst['next_ret_1d'].values, p):
            rows.append((d, s, int(a), float(r5), float(r1), float(pp)))
        n_blocks += 1
        pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
        log(f"block {n_blocks}: cutoff={dates[cut].date()} train={len(trn):,} "
            f"preds={len(tst):,} elapsed={time.time()-t0:.0f}s")
    return pd.DataFrame(rows, columns=cols), n_blocks


def backtest(P, W, cost_rate=0.001):
    """Audited methodology, overlap-safe: a 5-day position is held 5 days and
    earns its symbol's NEXT-DAY return on each held day (compounding of 5 daily
    returns ~= the 5-day return). Every price move is counted exactly once.
    Same assertions as Phases 3-5."""
    all_dates = np.sort(P['date'].unique())
    W = W.reindex(all_dates).fillna(0.0)
    R1 = P.pivot_table(index='date', columns='symbol', values='ret_1d',
                       aggfunc='mean').reindex(all_dates)
    gross = (W * R1.fillna(0.0)).sum(axis=1)
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - cost_rate * turnover
    assert (P['ret_1d'].abs() <= 0.5 + 1e-12).all(), "per-symbol |1d ret| > 0.5"
    assert np.isfinite(net).all() and (net > -1).all(), "bad daily net returns"
    assert (net.abs() <= 0.15 + 1e-12).all(), "absurd daily portfolio return"
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all(), "daily exposure exceeds 1.0"
    eq = (1.0 + net).cumprod()
    assert np.isfinite(eq).all() and (eq > 0).all(), "equity not finite/positive"
    mdd = float((eq / eq.cummax() - 1.0).min())
    assert -1.0 <= mdd <= 0.0, "MDD outside [-1, 0]"
    assert turnover.sum() < 5000, "implausible total turnover"
    sd = net.std(ddof=0)
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else float('nan')
    return {'cumulative_return_net': float(eq.iloc[-1] - 1.0),
            'cumulative_return_gross': float((1.0 + gross).prod() - 1.0),
            'max_drawdown': mdd, 'sharpe_net': sharpe,
            'total_turnover': float(turnover.sum()),
            'n_active_mean': float((W > 0).sum(axis=1).mean()),
            'n_days': int(len(all_dates))}, eq, net


def freeze(pv, step=H):
    return pv.iloc[::step].reindex(pv.index).ffill()


def norm(raw):
    W = raw.div(raw.sum(axis=1), axis=0)
    assert (W.sum(axis=1).fillna(0.0) <= 1.0 + 1e-9).all(), "exposure > 1"
    return W.fillna(0.0)


def main():
    t0 = time.time()
    log("PHASE 6: 5-DAY HORIZON — PRE-REGISTERED WALK-FORWARD")
    df = load_data()
    feats = get_feats(df)
    log(f"rows={len(df):,} features={len(feats)} dates={df['date'].nunique():,} "
        f"span={df['date'].min().date()}..{df['date'].max().date()}")
    cache = ROOT / "reports/phase6_predictions.csv"
    if cache.exists():
        P = pd.read_csv(cache, parse_dates=['date'])
        if 'ret_1d' not in P.columns:
            m1 = df.set_index(['date', 'symbol'])['next_ret_1d']
            key = pd.MultiIndex.from_arrays([P['date'], P['symbol']])
            P['ret_1d'] = m1.reindex(key).values
        n_na = int(P['ret_1d'].isna().sum())
        n_bad1 = int((P['ret_1d'].abs() > 0.5 + 1e-12).sum())
        P = P[P['ret_1d'].notna() & (P['ret_1d'].abs() <= 0.5 + 1e-12)]
        n_bad5 = int((P['ret_5d'].abs() > 0.5 + 1e-12).sum())
        P = P[P['ret_5d'].abs() <= 0.5 + 1e-12]
        log(f"resume: dropped {n_na} rows w/o 1d ret, {n_bad1} with |1d ret|>0.5, "
            f"{n_bad5} with |5d ret|>0.5; kept {len(P):,}")
        P.to_csv(cache, index=False)
        assert not P.duplicated(['date', 'symbol']).any(), "duplicate date-symbol predictions"
        n_blocks = 51
        log(f"resumed cached predictions: {len(P):,} rows (no retraining)")
    else:
        P, n_blocks = walk_forward(df, feats)
    log(f"walk-forward ready: blocks={n_blocks} oos_predictions={len(P):,}")

    # ---- secondary: accuracy vs 5d base rate (overlap-adjusted) ----
    base = float(P['actual'].mean())
    pred = (P['prob'] >= 0.5).astype(int)
    acc = float((pred == P['actual']).mean())
    z_nom = (acc - base) / np.sqrt(base * (1 - base) / len(P))
    n1, n0 = int((P['actual'] == 1).sum()), int((P['actual'] == 0).sum())
    r_all = P['prob'].rank()
    auc = float((r_all[P['actual'] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    log(f"acc5d@0.5={acc:.4f} base_rate={base:.4f} z_nom={z_nom:+.1f} "
        f"z_overlap_adj={z_nom/np.sqrt(H):+.1f} auc={auc:.4f}")

    P['tercile'] = pd.qcut(P['prob'], 3, labels=['low', 'mid', 'high'])
    terr = P.groupby('tercile', observed=True).agg(
        mean_ret=('ret_5d', 'mean'), mean_dir=('actual', 'mean'), n=('actual', 'size'))
    log("terciles: " + " | ".join(
        f"{i}: ret={r['mean_ret']:+.4f} dir={r['mean_dir']:.4f} n={int(r['n'])}"
        for i, r in terr.iterrows()))
    spread = float(terr.loc['high', 'mean_ret'] - terr.loc['low', 'mean_ret'])
    log(f"top-bottom tercile 5d-return spread: {spread:+.4f} ({spread*1e4:.1f} bps/5d)")

    # ---- pre-registered designs (weights frozen every 5 trading days) ----
    T = P.assign(t=1.0).pivot_table(index='date', columns='symbol', values='t', aggfunc='max')
    probs = P.pivot_table(index='date', columns='symbol', values='prob', aggfunc='mean')
    Tf, probs_f = freeze(T), freeze(probs)
    rk = probs_f.rank(axis=1, pct=True)
    tw = pd.DataFrame(1.0, index=probs_f.index, columns=probs_f.columns)
    tw[rk >= 2.0 / 3.0] = 1.5
    tw[rk <= 1.0 / 3.0] = 0.5
    designs = {
        'F0_naive5_holdall_w5': norm(Tf),
        'F1_xgb5_binary_w5': norm((probs_f >= 0.5).astype(float).where(T.notna())),
        'F2_xgb5_tercile_w5': norm(tw.where(T.notna())),
    }
    h_dates = pd.DatetimeIndex(P.loc[P['date'] > '2024-12-31', 'date'].unique())
    results, curves = {}, {}
    for name, W in designs.items():
        full, eq, net = backtest(P, W)
        is_r, _, _ = backtest(P[P['date'] <= '2024-12-31'], W.loc[W.index <= '2024-12-31'])
        h_r, _, _ = backtest(P[P['date'].isin(h_dates)], W.loc[W.index.isin(h_dates)])
        hc = {}
        for bps in (0.001, 0.0025):
            r, _, _ = backtest(P[P['date'].isin(h_dates)],
                               W.loc[W.index.isin(h_dates)], cost_rate=bps)
            hc[f'{bps}'] = {'net': r['cumulative_return_net'], 'sharpe': r['sharpe_net']}
        results[name] = {'full': full, 'in_sample_2018_2024': is_r,
                         'holdout_2025_2026': h_r, 'holdout_costs': hc}
        curves[name] = net
        log(f"{name:24s} net={full['cumulative_return_net']:+8.3f} sh={full['sharpe_net']:5.2f} "
            f"mdd={full['max_drawdown']:6.3f} to={full['total_turnover']:6.1f} "
            f"nact={full['n_active_mean']:5.1f} | H net={h_r['cumulative_return_net']:+7.3f} "
            f"sh={h_r['sharpe_net']:5.2f} | H@25 net={hc['0.0025']['net']:+7.3f} "
            f"sh={hc['0.0025']['sharpe']:5.2f} ({time.time()-t0:.0f}s)")

    ref = results['F0_naive5_holdall_w5']['holdout_costs']
    verdicts = {}
    for name in ('F1_xgb5_binary_w5', 'F2_xgb5_tercile_w5'):
        c = results[name]['holdout_costs']
        ok = all(c[b]['net'] > ref[b]['net'] and c[b]['sharpe'] > ref[b]['sharpe'] for b in c)
        verdicts[name] = {'pass': bool(ok), 'design': c, 'reference': ref}
        log(f"VERDICT {name}: {'PASS' if ok else 'FAIL'} vs F0 "
            f"(H@10 {c['0.001']['net']:+.3f}/{c['0.001']['sharpe']:+.2f} vs "
            f"{ref['0.001']['net']:+.3f}/{ref['0.001']['sharpe']:+.2f}; "
            f"H@25 {c['0.0025']['net']:+.3f}/{c['0.0025']['sharpe']:+.2f} vs "
            f"{ref['0.0025']['net']:+.3f}/{ref['0.0025']['sharpe']:+.2f})")

    pd.DataFrame(curves).to_csv(ROOT / "reports/phase6_daily_equity.csv")
    out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'horizon_days': H,
           'n_predictions': int(len(P)), 'base_rate_5d': base,
           'acc_at_050': acc, 'auc': auc, 'z_nominal': float(z_nom),
           'z_overlap_adjusted': float(z_nom / np.sqrt(H)),
           'terciles': terr.reset_index().to_dict(orient='records'),
           'tercile_spread_5d': spread, 'results': results, 'verdicts': verdicts,
           'pre_registration': 'designs fixed a priori; PASS = beat F0 on 2025-26 '
                               'holdout net AND Sharpe at 10 and 25 bps'}
    (ROOT / "reports/phase6_results.json").write_text(json.dumps(out, indent=2))
    log(f"DONE in {time.time()-t0:.0f}s -> reports/phase6_results.json")


if __name__ == '__main__':
    main()



