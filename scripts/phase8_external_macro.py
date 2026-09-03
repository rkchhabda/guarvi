#!/usr/bin/env python3
"""Phase 8: external macro features — pre-registered two-gate test.

PRE-REGISTRATION (declared before evaluation):
  External series (yfinance, public): ^INDIAVIX (India VIX), ^GSPC (S&P 500),
  INR=X (USD/INR), CL=F (WTI crude).
  Lookahead rule: India VIX value at NSE date t is computed at NSE close t -> allowed same-day.
  Foreign series (S&P, FX, crude) finalize AFTER NSE close t -> at NSE date t we use the most
  recent foreign session with date STRICTLY BEFORE t (merge_asof backward, no exact match).
  Features (9): vix_level, vix_chg1d, vix_z20, vix_high_regime, spx_ret1d_lag, spx_ret5d_lag,
                usdinr_chg1d_lag, crude_ret1d_lag, crude_ret5d_lag  => 62 base + 9 = 71 total.
  Protocol: identical to Phase 3 (1-day direction label, |ret|<=0.5 filter, retrain every 42
  trading days, 500-day warmup, 200K train cap, XGB 150/depth4/lr0.05 hist, median imputer).
  Base arm = cached Phase-3 predictions (reports/phase3_predictions.csv) -> exact pairing.
  Gate 1: extended AUC > base AUC on paired OOS predictions.
  Sanity alarm: OOS accuracy > 0.60 or AUC > 0.65 => ABORT (leakage), as in Phase 7.
  Gate 2 (only if Gate 1 passes): G0 naive daily equal-weight reference; G1 xgb>=0.5 daily;
  G2 tercile tilt 1.5/1.0/0.5 daily. PASS = beat G0 on 2025-26 holdout net AND Sharpe at
  BOTH 10 and 25 bps. No new advanced models.
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

LOG = ROOT / "reports" / "phase8_run.log"
RETRAIN_EVERY = 42
START_DAYS = 500
MAX_TRAIN_ROWS = 200_000
MACRO_FEATS = ['vix_level', 'vix_chg1d', 'vix_z20', 'vix_high_regime',
               'spx_ret1d_lag', 'spx_ret5d_lag', 'usdinr_ret1d_lag',
               'crude_ret1d_lag', 'crude_ret5d_lag']
BASE_EXCL = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source',
             'target_direction', 'next_ret',
             'next_ret_1d', 'target_1d', 'next_ret_5d', 'target_5d', 'sector'}
LEAKY = {'target_1d', 'next_ret_1d', 'next_ret', 'target_direction',
         'target_5d', 'next_ret_5d'}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch_external(nse_dates):
    """Download external series and align to NSE dates with the no-lookahead rule."""
    import yfinance as yf
    log("downloading external series: ^INDIAVIX ^GSPC INR=X CL=F (2015-01-01 ..)")
    raw = {}
    for t in ('^INDIAVIX', '^GSPC', 'INR=X', 'CL=F'):
        h = yf.download(t, start='2015-01-01', progress=False, auto_adjust=True, threads=False)
        if h is None or len(h) == 0:
            raise RuntimeError(f"empty download for {t}")
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        s = pd.to_numeric(h['Close'], errors='coerce').dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        raw[t] = s.sort_index()
        log(f"  {t}: {len(s):,} rows {s.index[0].date()}..{s.index[-1].date()}")
    f = pd.DataFrame({'date': pd.DatetimeIndex(nse_dates)})
    f['date'] = f['date'].astype('datetime64[ns]')
    v = raw['^INDIAVIX'].reindex(raw['^INDIAVIX'].index.union(f['date'])).ffill()
    v = v.reindex(f['date'])
    f['vix_level'] = v.values
    f['vix_chg1d'] = v.pct_change().values
    m20, s20 = v.rolling(20, min_periods=10).mean(), v.rolling(20, min_periods=10).std()
    f['vix_z20'] = ((v - m20) / s20).values
    f['vix_high_regime'] = (v > v.rolling(60, min_periods=20).median()).astype(float).values
    for key, tk, lags in (('spx', '^GSPC', (1, 5)), ('usdinr', 'INR=X', (1,)),
                          ('crude', 'CL=F', (1, 5))):
        s = raw[tk]
        for L in lags:
            rL = s.pct_change(L) if L > 1 else s.pct_change()
            rL = rL.replace([np.inf, -np.inf], np.nan).dropna()
            src = pd.DataFrame({'date': rL.index, 'val': rL.values}).sort_values('date')
            src['date'] = src['date'].astype('datetime64[ns]')
            q = pd.DataFrame({'date': f['date']}).sort_values('date')
            m = pd.merge_asof(q, src, on='date', direction='backward',
                              allow_exact_matches=False)
            f[f'{key}_ret{L}d_lag'] = m['val'].values
    cov = f[MACRO_FEATS].notna().mean()
    log("macro coverage: " + ", ".join(f"{c}={cov[c]:.4f}" for c in MACRO_FEATS))
    assert (cov >= 0.99).all(), f"macro coverage < 99%: {cov.to_dict()}"
    return f


def load_data():
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol')['close']
    df['next_ret_1d'] = g.shift(-1) / df['close'] - 1.0
    bad = df['next_ret_1d'].abs() > 0.5
    n_bad = int(bad.sum())
    df = df[~bad].dropna(subset=['next_ret_1d'])
    df['target_1d'] = (df['next_ret_1d'] > 0).astype(int)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    log(f"excluded {n_bad} rows with |1d ret|>0.5; rows={len(df):,} "
        f"base_rate={df['target_1d'].mean():.4f}")
    return df


def get_feats(df):
    return [c for c in df.columns
            if c not in BASE_EXCL and df[c].dtype in ('float64', 'float32', 'int64')]


def walk_forward(df, feats):
    dates = pd.DatetimeIndex(np.sort(df['date'].unique()))
    dmap = {d: i for i, d in enumerate(dates)}
    date_idx = df['date'].map(dmap).values
    cols = ['date', 'symbol', 'actual', 'ret_1d', 'prob']
    out_csv = ROOT / "reports/phase8_predictions.csv"
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
                              verbosity=0).fit(X, trn['target_1d'].values)
        p = m.predict_proba(imp.transform(tst[feats].values))[:, 1]
        for d, s, a, r, pp in zip(tst['date'].values, tst['symbol'].values,
                                  tst['target_1d'].values, tst['next_ret_1d'].values, p):
            rows.append((d, s, int(a), float(r), float(pp)))
        n_blocks += 1
        pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
        log(f"block {n_blocks}: cutoff={dates[cut].date()} train={len(trn):,} "
            f"preds={len(tst):,} elapsed={time.time()-t0:.0f}s")
    return pd.DataFrame(rows, columns=cols), n_blocks


def auc_mw(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def backtest(P, W, cost_rate=0.001):
    """Audited methodology (Phases 3-5): exposure<=1, turnover=0.5*sum|dw|,
    costs on turnover, Sharpe sqrt(252), MDD in [-1,0], full assertions."""
    assert not P.duplicated(['date', 'symbol']).any(), "duplicate date-symbol predictions"
    all_dates = np.sort(P['date'].unique())
    W = W.reindex(all_dates).fillna(0.0)
    R = P.pivot_table(index='date', columns='symbol', values='ret_1d',
                      aggfunc='mean').reindex(all_dates)
    gross = (W * R.fillna(0.0)).sum(axis=1)
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
    sd = net.std(ddof=0)
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else float('nan')
    return {'cumulative_return_net': float(eq.iloc[-1] - 1.0),
            'cumulative_return_gross': float((1.0 + gross).prod() - 1.0),
            'max_drawdown': mdd, 'sharpe_net': sharpe,
            'total_turnover': float(turnover.sum()),
            'n_active_mean': float((W > 0).sum(axis=1).mean()),
            'n_days': int(len(all_dates))}, eq, net


def norm(raw):
    W = raw.div(raw.sum(axis=1), axis=0)
    assert (W.sum(axis=1).fillna(0.0) <= 1.0 + 1e-9).all(), "exposure > 1"
    return W.fillna(0.0)


def main():
    t0 = time.time()
    log("PHASE 8: EXTERNAL MACRO FEATURES — PRE-REGISTERED TWO-GATE TEST")
    df = load_data()
    df['date'] = df['date'].astype('datetime64[ns]')
    macro = fetch_external(np.sort(df['date'].unique()))
    df = df.merge(macro, on='date', how='left')
    base_feats = [c for c in get_feats(df) if c not in MACRO_FEATS]
    assert len(base_feats) == 62, f"base feature count {len(base_feats)} != 62 (Phase-3 protocol)"
    ext_feats = base_feats + MACRO_FEATS
    assert len(ext_feats) == 71, f"ext feature count {len(ext_feats)} != 71"
    assert not (set(base_feats) & LEAKY), f"LEAKAGE in base: {set(base_feats) & LEAKY}"
    assert not (set(ext_feats) & LEAKY), f"LEAKAGE in ext: {set(ext_feats) & LEAKY}"
    log(f"rows={len(df):,} base_feats={len(base_feats)} ext_feats={len(ext_feats)} "
        f"(anti-leakage assertions OK)")

    cache = ROOT / "reports/phase8_predictions.csv"
    if cache.exists():
        ext = pd.read_csv(cache, parse_dates=['date'])
        n_blocks = -1
        log(f"resumed cached extended predictions: {len(ext):,} rows (no retraining)")
    else:
        ext, n_blocks = walk_forward(df, ext_feats)
        log(f"walk-forward done: blocks={n_blocks} oos_predictions={len(ext):,}")

    base = pd.read_csv(ROOT / "reports/phase3_predictions.csv", parse_dates=['date'])
    pcol = ('prob_base' if 'prob_base' in base.columns else
            'prob_xgb' if 'prob_xgb' in base.columns else 'prob')
    log(f"base arm source: phase3_predictions.csv column '{pcol}' ({len(base):,} rows)")
    base = base[['date', 'symbol', pcol]].rename(columns={pcol: 'prob_base'})
    m = ext.merge(base, on=['date', 'symbol'], how='inner')
    assert not m[['prob_base', 'prob']].isna().any().any(), "NaN probabilities after merge"
    log(f"paired n={len(m):,} (ext={len(ext):,}, base={len(base):,})")
    y = m['actual'].values.astype(int)

    auc_b, auc_e = auc_mw(y, m['prob_base']), auc_mw(y, m['prob'])
    pb = (m['prob_base'] >= 0.5).astype(int).values
    pe = (m['prob'] >= 0.5).astype(int).values
    if float(np.mean(pe == y)) > 0.60 or auc_e > 0.65:
        log("SANITY ALARM: extended OOS acc>0.60 or AUC>0.65 — leakage suspected; ABORT")
        raise SystemExit(1)
    acc_b, acc_e = float(np.mean(pb == y)), float(np.mean(pe == y))
    br = float(y.mean())
    z_b = (acc_b - br) / np.sqrt(br * (1 - br) / len(m))
    z_e = (acc_e - br) / np.sqrt(br * (1 - br) / len(m))
    b = int(((pb == y) & (pe != y)).sum())
    c = int(((pe == y) & (pb != y)).sum())
    z_mc = (b - c) / np.sqrt(b + c) if (b + c) > 0 else float('nan')
    log(f"base: auc={auc_b:.4f} acc={acc_b:.4f} z={z_b:+.1f} | "
        f"ext: auc={auc_e:.4f} acc={acc_e:.4f} z={z_e:+.1f} | dAUC={auc_e-auc_b:+.4f} | "
        f"McNemar b={b} c={c} z={z_mc:+.2f}")
    gate1 = bool(auc_e > auc_b)
    log(f"GATE 1: {'PASS (proceed to portfolios)' if gate1 else 'FAIL (stop per pre-registration)'}")

    results, verdicts, gate2_pass = {}, {}, False
    if gate1:
        # ---- Gate 2: pre-registered daily portfolios vs naive reference ----
        P = m[['date', 'symbol', 'actual', 'ret_1d', 'prob']]
        T = P.assign(t=1.0).pivot_table(index='date', columns='symbol', values='t', aggfunc='max')
        probs = P.pivot_table(index='date', columns='symbol', values='prob', aggfunc='mean')
        rk = probs.rank(axis=1, pct=True)
        tw = pd.DataFrame(1.0, index=probs.index, columns=probs.columns)
        tw[rk >= 2.0 / 3.0] = 1.5
        tw[rk <= 1.0 / 3.0] = 0.5
        designs = {
            'G0_naive_daily': norm(T),
            'G1_xgb_binary_daily': norm((probs >= 0.5).astype(float).where(T.notna())),
            'G2_xgb_tercile_daily': norm(tw.where(T.notna())),
        }
        h_dates = pd.DatetimeIndex(P.loc[P['date'] > '2024-12-31', 'date'].unique())
        curves = {}
        for name, W in designs.items():
            full, _, net = backtest(P, W)
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
            log(f"{name:22s} net={full['cumulative_return_net']:+8.3f} sh={full['sharpe_net']:5.2f} "
                f"to={full['total_turnover']:7.1f} | H net={h_r['cumulative_return_net']:+7.3f} "
                f"sh={h_r['sharpe_net']:5.2f} | H@25 net={hc['0.0025']['net']:+7.3f} "
                f"sh={hc['0.0025']['sharpe']:5.2f}")
        ref = results['G0_naive_daily']['holdout_costs']
        for name in ('G1_xgb_binary_daily', 'G2_xgb_tercile_daily'):
            cde = results[name]['holdout_costs']
            ok = all(cde[k]['net'] > ref[k]['net'] and cde[k]['sharpe'] > ref[k]['sharpe']
                     for k in cde)
            verdicts[name] = {'pass': bool(ok), 'design': cde, 'reference': ref}
            log(f"VERDICT {name}: {'PASS' if ok else 'FAIL'} vs G0 "
                f"(H@10 {cde['0.001']['net']:+.3f}/{cde['0.001']['sharpe']:+.2f} vs "
                f"{ref['0.001']['net']:+.3f}/{ref['0.001']['sharpe']:+.2f}; "
                f"H@25 {cde['0.0025']['net']:+.3f}/{cde['0.0025']['sharpe']:+.2f} vs "
                f"{ref['0.0025']['net']:+.3f}/{ref['0.0025']['sharpe']:+.2f})")
        gate2_pass = any(v['pass'] for v in verdicts.values())
        pd.DataFrame(curves).to_csv(ROOT / "reports/phase8_daily_equity.csv")

    out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'n_predictions_ext': int(len(ext)), 'n_paired': int(len(m)),
           'base_feats': len(base_feats), 'ext_feats': len(ext_feats),
           'macro_features': MACRO_FEATS,
           'auc_base': auc_b, 'auc_ext': auc_e, 'acc_base': acc_b, 'acc_ext': acc_e,
           'z_base': float(z_b), 'z_ext': float(z_e),
           'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_z': float(z_mc),
           'gate1_pass': gate1, 'gate2_pass': bool(gate2_pass),
           'results': results, 'verdicts': verdicts,
           'pre_registration': 'Gate1: ext AUC > base AUC (paired). Sanity alarm acc>0.60/'
                               'AUC>0.65 aborts. Gate2 (only if Gate1 PASS): G1/G2 daily '
                               'designs must beat G0 on 2025-26 holdout net AND Sharpe at '
                               '10 and 25 bps. Foreign series lagged 1 session (no lookahead).'}
    (ROOT / "reports/phase8_results.json").write_text(json.dumps(out, indent=2))
    log(f"DONE in {time.time()-t0:.0f}s -> reports/phase8_results.json | "
        f"gate1={'PASS' if gate1 else 'FAIL'} gate2={'PASS' if gate2_pass else 'FAIL/NA'}")


if __name__ == '__main__':
    main()



