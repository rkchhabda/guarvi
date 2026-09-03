#!/usr/bin/env python3
"""Phase 7: Market-state & sector-relative features — pre-registered two-gate test.

PRE-REGISTRATION (declared before any holdout evaluation):
  New information (computed ONLY from t-dated cross-sections of existing OHLCV
  features — no lookahead, no external data):
    market: breadth_1d, breadth_20d, breadth_50d, cs_disp_1d, mom_5d, mom_20d,
            vol_regime (market vol_20d vs its 250d mean)
    sector: sect_mom_5d/20d, rel_mom_5d/20d (symbol minus sector mean),
            sect_breadth_1d, sect_disp_20d; static NSE-style sector map.
  Model: XGB, IDENTICAL Phase-3 params/protocol (retrain 42d, 500-day warmup,
  200K train cap, median imputation, 1-day labels, |ret_1d|<=0.5 filter).
  GATE 1 (paired, same OOS keys as Phase-3 base): proceed to portfolios ONLY IF
  extended OOS AUC > base OOS AUC. Paired accuracy via McNemar z.
  GATE 2 designs (only if Gate 1 passes; weights frozen every 21 trading days):
    G0 naive daily equal-weight (reference) · G1 xgb>=0.5 frozen w21 ·
    G2 tercile tilt 1.5/1/0.5 frozen w21.
  PASS rule: G1 or G2 beats G0 on 2025-26 holdout net AND Sharpe at BOTH 10 and
  25 bps. Overlap-safe backtest with all Phase-6 assertions.
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

LOG = ROOT / "reports" / "phase7_run.log"
RETRAIN_EVERY, START_DAYS, MAX_TRAIN_ROWS = 42, 500, 200_000

SECTORS = {
    'AXISBANK': 'Banks', 'BANKBARODA': 'Banks', 'CANBK': 'Banks', 'HDFCBANK': 'Banks',
    'ICICIBANK': 'Banks', 'INDUSINDBK': 'Banks', 'KOTAKBANK': 'Banks', 'PNB': 'Banks',
    'BAJFINANCE': 'Financials', 'BAJAJFINSV': 'Financials', 'BAJAJHLDNG': 'Financials',
    'BAJAJHFL': 'Financials', 'CHOLAFIN': 'Financials', 'HDFCLIFE': 'Financials',
    'SBILIFE': 'Financials', 'LICI': 'Financials', 'SHRIRAMFIN': 'Financials',
    'PFC': 'Financials', 'RECLTD': 'Financials', 'IRFC': 'Financials', 'HUDCO': 'Financials',
    'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT',
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'IOC': 'Energy', 'BPCL': 'Energy',
    'GAIL': 'Energy', 'COALINDIA': 'Energy', 'ATGL': 'Energy', 'ADANIENT': 'Energy',
    'NTPC': 'Power', 'POWERGRID': 'Power', 'TATAPOWER': 'Power', 'JSWENERGY': 'Power',
    'ADANIENSOL': 'Power', 'ADANIGREEN': 'Power', 'ADANIPOWER': 'Power',
    'MARUTI': 'Auto', 'M&M': 'Auto', 'TATAMOTORS': 'Auto', 'TMPV': 'Auto',
    'BAJAJ-AUTO': 'Auto', 'EICHERMOT': 'Auto', 'HEROMOTOCO': 'Auto', 'TVSMOTOR': 'Auto',
    'ESCORTS': 'Auto', 'MOTHERSON': 'Auto', 'BOSCHLTD': 'Auto',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
    'TATACONSUM': 'FMCG', 'DABUR': 'FMCG', 'GODREJCP': 'FMCG', 'UNITDSPR': 'FMCG', 'VBL': 'FMCG',
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals', 'VEDL': 'Metals',
    'JINDALSTEL': 'Metals',
    'ULTRACEMCO': 'Cement', 'GRASIM': 'Cement', 'SHREECEM': 'Cement', 'AMBUJACEM': 'Cement',
    'SUNPHARMA': 'Healthcare', 'CIPLA': 'Healthcare', 'DRREDDY': 'Healthcare',
    'DIVISLAB': 'Healthcare', 'TORNTPHARM': 'Healthcare', 'ZYDUSLIFE': 'Healthcare',
    'APOLLOHOSP': 'Healthcare', 'MAXHEALTH': 'Healthcare',
    'LT': 'CapitalGoods', 'SIEMENS': 'CapitalGoods', 'BEL': 'CapitalGoods',
    'HAL': 'CapitalGoods', 'BHEL': 'CapitalGoods', 'CGPOWER': 'CapitalGoods', 'ABB': 'CapitalGoods',
    'TITAN': 'ConsumerDurables', 'HAVELLS': 'ConsumerDurables',
    'DLF': 'Realty', 'LODHA': 'Realty',
    'BHARTIARTL': 'Telecom', 'IDEA': 'Telecom',
    'PIDILITIND': 'Chemicals', 'TATACHEM': 'Chemicals', 'ASIANPAINT': 'Chemicals',
    'DMART': 'RetailHospitality', 'TRENT': 'RetailHospitality', 'INDHOTEL': 'RetailHospitality',
    'INDIGO': 'Transport', 'ADANIPORTS': 'Transport',
    'SWIGGY': 'Internet', 'ETERNAL': 'Internet', 'NAUKRI': 'Internet',
}
BASE_EXCL = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source',
             'target_direction', 'next_ret',
             'next_ret_1d', 'target_1d', 'next_ret_5d', 'target_5d', 'sector'}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def add_market_sector_features(df):
    """All features are within-date cross-sections of t-dated columns -> lag-safe."""
    g = df.groupby('date', sort=False)
    df['mkt_breadth_1d'] = g['ret_1d'].transform(lambda s: (s > 0).mean())
    df['mkt_breadth_20d'] = g['dist_sma_20'].transform(lambda s: (s > 0).mean())
    df['mkt_breadth_50d'] = g['dist_sma_50'].transform(lambda s: (s > 0).mean())
    df['mkt_cs_disp_1d'] = g['ret_1d'].transform('std')
    df['mkt_mom_5d'] = g['ret_5d'].transform('mean')
    df['mkt_mom_20d'] = g['ret_20d'].transform('mean')
    mvol = g['volatility_20d'].mean().sort_index()
    df['mkt_vol_regime'] = df['date'].map(mvol / mvol.rolling(250, min_periods=100).mean())
    df['sector'] = df['symbol'].map(SECTORS).fillna('Other')
    gs = df.groupby(['date', 'sector'], sort=False)
    df['sect_mom_5d'] = gs['ret_5d'].transform('mean')
    df['sect_mom_20d'] = gs['ret_20d'].transform('mean')
    df['sect_breadth_1d'] = gs['ret_1d'].transform(lambda s: (s > 0).mean())
    df['sect_disp_20d'] = gs['ret_20d'].transform('std')
    df['rel_mom_5d'] = df['ret_5d'] - df['sect_mom_5d']
    df['rel_mom_20d'] = df['ret_20d'] - df['sect_mom_20d']
    new_feats = ['mkt_breadth_1d', 'mkt_breadth_20d', 'mkt_breadth_50d', 'mkt_cs_disp_1d',
                 'mkt_mom_5d', 'mkt_mom_20d', 'mkt_vol_regime', 'sect_mom_5d',
                 'sect_mom_20d', 'sect_breadth_1d', 'sect_disp_20d', 'rel_mom_5d', 'rel_mom_20d']
    n_unmapped = int((df['sector'] == 'Other').sum())
    log(f"sector features added: {len(new_feats)} new; unmapped rows (Other)={n_unmapped:,} "
        f"({n_unmapped/len(df)*100:.1f}%); sectors={df['sector'].nunique()}")
    return df, new_feats


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/nifty100_features.parquet")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol')['close']
    df['next_ret_1d'] = g.shift(-1) / df['close'] - 1.0
    bad = df['next_ret_1d'].abs() > 0.5
    n_bad = int(bad.sum())
    df = df[~bad]
    df = df.dropna(subset=['next_ret_1d'])
    df['target_1d'] = (df['next_ret_1d'] > 0).astype(int)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    log(f"excluded {n_bad} rows |1d ret|>0.5; rows={len(df):,} base_rate={df['target_1d'].mean():.4f}")
    return df


def walk_forward(df, feats):
    dates = pd.DatetimeIndex(np.sort(df['date'].unique()))
    dmap = {d: i for i, d in enumerate(dates)}
    date_idx = df['date'].map(dmap).values
    rows, n_blocks = [], 0
    t0 = time.time()
    for cut in range(START_DAYS, len(dates) - 1, RETRAIN_EVERY):
        trn, tst = df[date_idx <= cut], df[(date_idx > cut) & (date_idx <= min(cut + RETRAIN_EVERY, len(dates) - 1))]
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
        if n_blocks % 10 == 0 or n_blocks == 1:
            pd.DataFrame(rows, columns=['date', 'symbol', 'actual', 'ret_1d', 'prob']).to_csv(
                ROOT / "reports/phase7_predictions.csv", index=False)
        log(f"block {n_blocks}: cutoff={dates[cut].date()} train={len(trn):,} "
            f"preds={len(tst):,} elapsed={time.time()-t0:.0f}s")
    pd.DataFrame(rows, columns=['date', 'symbol', 'actual', 'ret_1d', 'prob']).to_csv(
        ROOT / "reports/phase7_predictions.csv", index=False)
    return pd.DataFrame(rows, columns=['date', 'symbol', 'actual', 'ret_1d', 'prob']), n_blocks


def backtest(P, W, cost_rate=0.001):
    """Overlap-safe audited backtest (Phase-6 methodology + assertions)."""
    all_dates = np.sort(P['date'].unique())
    W = W.reindex(all_dates).fillna(0.0)
    R1 = P.pivot_table(index='date', columns='symbol', values='ret_1d',
                       aggfunc='mean').reindex(all_dates)
    gross = (W * R1.fillna(0.0)).sum(axis=1)
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - cost_rate * turnover
    assert (P['ret_1d'].abs() <= 0.5 + 1e-12).all()
    assert np.isfinite(net).all() and (net > -1).all() and (net.abs() <= 0.15 + 1e-12).all()
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all()
    eq = (1.0 + net).cumprod()
    assert np.isfinite(eq).all() and (eq > 0).all()
    mdd = float((eq / eq.cummax() - 1.0).min())
    assert -1.0 <= mdd <= 0.0 and turnover.sum() < 5000
    sd = net.std(ddof=0)
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else float('nan')
    return {'cumulative_return_net': float(eq.iloc[-1] - 1.0),
            'cumulative_return_gross': float((1.0 + gross).prod() - 1.0),
            'max_drawdown': mdd, 'sharpe_net': sharpe,
            'total_turnover': float(turnover.sum()),
            'n_active_mean': float((W > 0).sum(axis=1).mean()),
            'n_days': int(len(all_dates))}, eq, net


def freeze(pv, step=21):
    return pv.iloc[::step].reindex(pv.index).ffill()


def norm(raw):
    W = raw.div(raw.sum(axis=1), axis=0)
    assert (W.sum(axis=1).fillna(0.0) <= 1.0 + 1e-9).all()
    return W.fillna(0.0)


def auc_mw(y, p):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(p).rank()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    log("PHASE 7: MARKET-STATE & SECTOR FEATURES — PRE-REGISTERED TWO-GATE TEST")
    df = load_data()
    base_feats = [c for c in df.columns if c not in BASE_EXCL and df[c].dtype in ('float64', 'float32', 'int64')]
    df, new_feats = add_market_sector_features(df)
    ext_feats = base_feats + new_feats
    LEAKY = {'target_1d', 'next_ret_1d', 'next_ret', 'target_direction', 'target_5d', 'next_ret_5d'}
    assert not (set(base_feats) & LEAKY), f"LEAKAGE: label cols in base features: {set(base_feats) & LEAKY}"
    assert not (set(ext_feats) & LEAKY), f"LEAKAGE: label cols in ext features: {set(ext_feats) & LEAKY}"
    assert len(base_feats) == 62, f"base feature count {len(base_feats)} != 62 (Phase-3 protocol)"
    log(f"rows={len(df):,} base_feats={len(base_feats)} ext_feats={len(ext_feats)} "
        f"(anti-leakage assertions OK)")

    # ---- GATE 1: paired comparison on identical OOS keys ----
    base = pd.read_csv(ROOT / "reports/phase3_predictions.csv", parse_dates=['date'])
    base = base.rename(columns={'prob_xgb': 'prob_base'})[['date', 'symbol', 'actual', 'next_ret', 'prob_base']]
    base['ret_1d'] = base['next_ret']
    cache = ROOT / "reports/phase7_predictions.csv"
    if cache.exists():
        ext = pd.read_csv(cache, parse_dates=['date'])
        n_blocks = 51
        log(f"resumed cached extended predictions: {len(ext):,} rows (no retraining)")
    else:
        ext, n_blocks = walk_forward(df, ext_feats)
        log(f"walk-forward done: blocks={n_blocks} oos_predictions={len(ext):,}")
    m = base.merge(ext, on=['date', 'symbol'], suffixes=('_b', '_e'))
    assert not m.duplicated(['date', 'symbol']).any()
    y = m['actual_b'].values
    assert (m['actual_b'] == m['actual_e']).all() and (m['ret_1d_b'] - m['ret_1d_e']).abs().max() < 1e-9
    auc_b, auc_e = auc_mw(y, m['prob_base']), auc_mw(y, m['prob'])
    pb = (m['prob_base'] >= 0.5).astype(int).values
    pe = (m['prob'] >= 0.5).astype(int).values
    if float(np.mean(pe == y)) > 0.60 or auc_e > 0.65:
        log("SANITY ALARM: extended OOS acc>0.60 or AUC>0.65 — leakage suspected; ABORT")
        raise SystemExit(1)
    acc_b, acc_e = float((pb == y).mean()), float((pe == y).mean())
    b_win = int(((pe == y) & (pb != y)).sum())
    c_win = int(((pb == y) & (pe != y)).sum())
    z_mc = (b_win - c_win) / np.sqrt(b_win + c_win) if (b_win + c_win) > 0 else float('nan')
    base_rate = float(y.mean())
    z_b = (acc_b - base_rate) / np.sqrt(base_rate * (1 - base_rate) / len(m))
    z_e = (acc_e - base_rate) / np.sqrt(base_rate * (1 - base_rate) / len(m))
    gate1 = bool(auc_e > auc_b)
    log(f"paired n={len(m):,} | base: auc={auc_b:.4f} acc={acc_b:.4f} z={z_b:+.1f} | "
        f"ext: auc={auc_e:.4f} acc={acc_e:.4f} z={z_e:+.1f} | dAUC={auc_e-auc_b:+.4f} | "
        f"McNemar b={b_win} c={c_win} z={z_mc:+.2f}")
    log(f"GATE 1: {'PASS' if gate1 else 'FAIL'} (proceed to portfolios only if ext AUC > base AUC)")

    out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_predictions': int(len(m)),
           'n_blocks': n_blocks, 'base_feats': len(base_feats), 'ext_feats': len(ext_feats),
           'auc_base': auc_b, 'auc_ext': auc_e, 'acc_base': acc_b, 'acc_ext': acc_e,
           'z_base': float(z_b), 'z_ext': float(z_e), 'mcnemar_b': b_win,
           'mcnemar_c': c_win, 'mcnemar_z': float(z_mc), 'gate1_pass': gate1,
           'pre_registration': 'Gate1: ext AUC > base AUC; Gate2 PASS = beat G0 on '
                               '2025-26 holdout net AND Sharpe at 10 and 25 bps'}
    if gate1:
        results, verdicts = gate2_portfolios(ext)
        out['portfolios'] = results
        out['verdicts'] = verdicts
        any_pass = any(v['pass'] for v in verdicts.values())
        out['gate2_any_pass'] = bool(any_pass)
        log(f"GATE 2 OVERALL: {'PASS' if any_pass else 'FAIL'}")
    (ROOT / "reports/phase7_results.json").write_text(json.dumps(out, indent=2))
    log(f"DONE in {time.time()-t0:.0f}s -> reports/phase7_results.json")


if __name__ == '__main__':
    main()



def gate2_portfolios(P):
    T = P.assign(t=1.0).pivot_table(index='date', columns='symbol', values='t', aggfunc='max')
    probs = P.pivot_table(index='date', columns='symbol', values='prob', aggfunc='mean')
    Tf, probs_f = freeze(T), freeze(probs)
    rk = probs_f.rank(axis=1, pct=True)
    tw = pd.DataFrame(1.0, index=probs_f.index, columns=probs_f.columns)
    tw[rk >= 2.0 / 3.0] = 1.5
    tw[rk <= 1.0 / 3.0] = 0.5
    designs = {'G0_naive_daily': norm(Tf),
               'G1_xgb_binary_w21': norm((probs_f >= 0.5).astype(float).where(T.notna())),
               'G2_xgb_tercile_w21': norm(tw.where(T.notna()))}
    h_dates = pd.DatetimeIndex(P.loc[P['date'] > '2024-12-31', 'date'].unique())
    results, curves = {}, {}
    for name, W in designs.items():
        full, _, net = backtest(P, W)
        h_r, _, _ = backtest(P[P['date'].isin(h_dates)], W.loc[W.index.isin(h_dates)])
        hc = {}
        for bps in (0.001, 0.0025):
            r, _, _ = backtest(P[P['date'].isin(h_dates)],
                               W.loc[W.index.isin(h_dates)], cost_rate=bps)
            hc[f'{bps}'] = {'net': r['cumulative_return_net'], 'sharpe': r['sharpe_net']}
        results[name] = {'full': full, 'holdout_2025_2026': h_r, 'holdout_costs': hc}
        curves[name] = net
        log(f"{name:22s} net={full['cumulative_return_net']:+8.3f} sh={full['sharpe_net']:5.2f} "
            f"to={full['total_turnover']:6.1f} | H@10 net={hc['0.001']['net']:+7.3f} "
            f"sh={hc['0.001']['sharpe']:5.2f} | H@25 net={hc['0.0025']['net']:+7.3f} "
            f"sh={hc['0.0025']['sharpe']:5.2f}")
    ref = results['G0_naive_daily']['holdout_costs']
    verdicts = {}
    for name in ('G1_xgb_binary_w21', 'G2_xgb_tercile_w21'):
        c = results[name]['holdout_costs']
        ok = all(c[b]['net'] > ref[b]['net'] and c[b]['sharpe'] > ref[b]['sharpe'] for b in c)
        verdicts[name] = {'pass': bool(ok), 'design': c, 'reference': ref}
        log(f"VERDICT {name}: {'PASS' if ok else 'FAIL'}")
    pd.DataFrame(curves).to_csv(ROOT / "reports/phase7_daily_equity.csv")
    return results, verdicts


