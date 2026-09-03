#!/usr/bin/env python3
"""Phase 4: Turnover reduction / cost-aware strategy design.

Reuses the 205,948 OOS predictions from phase3 (NO retraining). Tests whether
the XGB edge survives realistic costs by cutting turnover:

  A. confidence thresholds   : hold only when prob >= theta
  B. top-K concentration     : hold best K signals per day (prob >= 0.5)
  C. hysteresis bands        : enter at theta_hi, exit at theta_lo (anti-churn)
  D. minimum holding period  : state machine with min_hold + hard stop
  E. cost sensitivity        : 5 / 10 / 25 / 50 bps on the top variants

Methodology identical to the audited backtest (equal-weight, exposure<=1,
turnover = 0.5*sum|dw|, Sharpe sqrt(252), MDD in [-1,0], full assertions).
Anti-overfit guard: variants ranked on 2018-2024; 2025-2026 reported as holdout.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

COST_RATE = 0.001  # 10 bps per unit turnover (phase-3 convention: 0.5*sum|dw|)


def backtest(P, W, cost_rate=COST_RATE):
    """W: DataFrame index=date, columns=symbol, position weights (0/1/n)."""
    all_dates = np.sort(P['date'].unique())
    W = W.reindex(all_dates).fillna(0.0)
    held = W.gt(0)
    # per-symbol daily strategy return = weight * next_ret
    R = P.pivot_table(index='date', columns='symbol', values='next_ret', aggfunc='mean').reindex(all_dates)
    gross = (W * R.fillna(0.0)).sum(axis=1)
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    cost = cost_rate * turnover
    net = gross - cost
    # ---- audited safety assertions ----
    assert (P['next_ret'].abs() <= 0.5 + 1e-12).all(), "per-symbol |return| > 0.5"
    assert np.isfinite(net).all(), "NaN/inf in daily net returns"
    assert (net.abs() <= 0.5 + 1e-12).all(), "absurd daily portfolio return"
    assert (net > -1).all(), "daily return <= -1"
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all(), "daily exposure exceeds 1.0"
    eq = (1.0 + net).cumprod()
    assert np.isfinite(eq).all() and (eq > 0).all(), "equity not finite/positive"
    peak = eq.cummax()
    mdd = float((eq / peak - 1.0).min())
    assert -1.0 <= mdd <= 0.0, "MDD outside [-1, 0]"
    sd = net.std(ddof=0)
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else float('nan')
    entries = int((held & ~held.shift(1, fill_value=False)).sum().sum())
    return {
        'cumulative_return_gross': float((1.0 + gross).prod() - 1.0),
        'cumulative_return_net': float(eq.iloc[-1] - 1.0),
        'max_drawdown': mdd, 'sharpe_net': sharpe, 'final_equity': float(eq.iloc[-1]),
        'total_turnover': float(turnover.sum()), 'n_entries': entries,
        'n_position_days': int(held.sum().sum()),
        'pct_days_in_market': float((W.sum(axis=1) > 0).mean()),
        'n_days': int(len(all_dates)),
    }, eq, net


def weights_threshold(P, prob_col, theta):
    m = (P[prob_col] >= theta).astype(float)
    piv = P.assign(sig=m).pivot_table(index='date', columns='symbol', values='sig', aggfunc='max').fillna(0.0)
    n = piv.sum(axis=1).replace(0, 1)
    return piv.div(n, axis=0)


def weights_topk(P, prob_col, k):
    d = P[P[prob_col] >= 0.5].copy()
    d['rk'] = d.groupby('date')[prob_col].rank(ascending=False, method='first')
    d = d[d['rk'] <= k]
    sig = d.assign(w=1.0).set_index(['date', 'symbol'])['w']
    piv = sig.unstack('symbol').fillna(0.0)
    n = piv.sum(axis=1).replace(0, 1)
    return piv.div(n, axis=0)


def weights_hysteresis(P, prob_col, enter, exit_, min_hold=0, hard_stop=None):
    """State machine per symbol: enter at prob>=enter; exit when held>=min_hold
    and prob<exit_, or immediately on hard_stop breach. Anti-churn design."""
    d = P.sort_values(['symbol', 'date']).reset_index(drop=True)
    pos = np.zeros(len(d), dtype=float)
    for sym, idx in d.groupby('symbol', sort=False).indices.items():
        days_held = 0
        for i in idx:
            p = d[prob_col].iat[i]
            if days_held > 0:
                if hard_stop is not None and p < hard_stop:
                    days_held = 0
                elif days_held >= min_hold and p < exit_:
                    days_held = 0
                else:
                    days_held += 1
            if days_held == 0 and p >= enter:
                days_held = 1
            pos[i] = 1.0 if days_held > 0 else 0.0
    P2 = d.assign(w=pos)
    piv = P2.pivot_table(index='date', columns='symbol', values='w', aggfunc='max').fillna(0.0)
    n = piv.sum(axis=1).replace(0, 1)
    return piv.div(n, axis=0)


def weights_naive(P):
    piv = P.assign(w=1.0).pivot_table(index='date', columns='symbol', values='w', aggfunc='max').fillna(0.0)
    n = piv.sum(axis=1).replace(0, 1)
    return piv.div(n, axis=0)


def held_accuracy(P, W):
    """Accuracy on held position-days. np.where-based (stack() in this pandas
    version no longer drops NaN, which silently corrupted the first run)."""
    rows, cols = np.where(W.values > 0)
    idx = pd.DataFrame({'date': W.index[rows].values, 'symbol': W.columns[cols].values})
    assert len(idx) == int((W.values > 0).sum()), "held count mismatch"
    sub = P.merge(idx, on=['date', 'symbol'], how='inner')
    assert len(sub) == len(idx), \
        f"held-accuracy join mismatch: sub={len(sub)} idx={len(idx)}"
    return float(sub['actual'].mean()), int(len(sub))


def diagnostics(P, W, name):
    """Concentration & data-quality checks for one variant."""
    R = P.pivot_table(index='date', columns='symbol', values='next_ret',
                      aggfunc='mean').reindex(W.index).fillna(0.0)
    contrib = (W * R)
    daily_gross = contrib.sum(axis=1)
    sym_contrib = contrib.sum().sort_values(ascending=False)
    top = sym_contrib.head(12)
    # mean |next_ret| of top contributors -> flags persistently bad price series
    chk = []
    for sym in top.index:
        s = P.loc[P['symbol'] == sym, 'next_ret']
        chk.append({'symbol': sym, 'total_contribution': float(top[sym]),
                    'mean_abs_next_ret': float(s.abs().mean()),
                    'max_next_ret': float(s.max()), 'n_rows': int(len(s))})
    return {'daily_gross_mean': float(daily_gross.mean()),
            'daily_gross_std': float(daily_gross.std(ddof=0)),
            'daily_gross_min': float(daily_gross.min()),
            'daily_gross_max': float(daily_gross.max()),
            'n_active_mean': float((W > 0).sum(axis=1).mean()),
            'n_active_max': int((W > 0).sum(axis=1).max()),
            'top_contributors': chk}


def yearly_net(P, W):
    R = P.pivot_table(index='date', columns='symbol', values='next_ret',
                      aggfunc='mean').reindex(W.index).fillna(0.0)
    gross = (W * R.fillna(0.0)).sum(axis=1)
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - COST_RATE * turnover
    return {str(y): float((1.0 + g).prod() - 1.0) for y, g in net.groupby(net.index.year)}


def main():
    t0 = time.time()
    print("PHASE 4: TURNOVER REDUCTION / COST-AWARE STRATEGY DESIGN", flush=True)
    P = pd.read_csv(ROOT / "reports/phase3_predictions.csv")
    P['date'] = pd.to_datetime(P['date'])
    P = P.sort_values(['date', 'symbol']).reset_index(drop=True)
    print(f"predictions={len(P):,} dates={P['date'].nunique():,}", flush=True)

    prob = 'prob_xgb'
    variants = {
        'naive_all': weights_naive(P),
        'xgb@0.50': weights_threshold(P, prob, 0.50),
        'xgb@0.52': weights_threshold(P, prob, 0.52),
        'xgb@0.55': weights_threshold(P, prob, 0.55),
        'xgb@0.58': weights_threshold(P, prob, 0.58),
        'xgb@0.60': weights_threshold(P, prob, 0.60),
        'topK25': weights_topk(P, prob, 25),
        'topK50': weights_topk(P, prob, 50),
        'band55/50': weights_hysteresis(P, prob, 0.55, 0.50),
        'band55/50_hold5': weights_hysteresis(P, prob, 0.55, 0.50, min_hold=5, hard_stop=0.45),
        'band60/55': weights_hysteresis(P, prob, 0.60, 0.55),
    }

    is_dates = pd.DatetimeIndex(P.loc[P['date'] <= '2024-12-31', 'date'].unique())
    results, curves = {}, {}
    diag_variants = {'xgb@0.50', 'xgb@0.55', 'xgb@0.58', 'xgb@0.60', 'band60/55'}
    yearly_variants = ['naive_all', 'xgb@0.50', 'xgb@0.55', 'xgb@0.60', 'band60/55', 'topK50']
    for name, W in variants.items():
        full, eq, net = backtest(P, W)
        acc, n_held = held_accuracy(P, W)
        is_r, _, _ = backtest(P[P['date'].isin(is_dates)], W.loc[W.index.isin(is_dates)])
        h_r, _, _ = backtest(P[~P['date'].isin(is_dates)], W.loc[~W.index.isin(is_dates)])
        results[name] = {'full': full, 'held_accuracy': acc, 'n_held': n_held,
                         'in_sample_2018_2024': is_r, 'holdout_2025_2026': h_r}
        if name in diag_variants:
            results[name]['diagnostics'] = diagnostics(P, W, name)
        if name in yearly_variants:
            results[name]['yearly_net'] = yearly_net(P, W)
        curves[name] = net
        print(f"{name:18s} net={full['cumulative_return_net']:+8.3f} gross={full['cumulative_return_gross']:+8.3f} "
              f"sharpe={full['sharpe_net']:5.2f} mdd={full['max_drawdown']:6.3f} to={full['total_turnover']:7.1f} "
              f"held_acc={acc:.4f}(n={n_held:,}) IS_sharpe={is_r['sharpe_net']:5.2f} H_sharpe={h_r['sharpe_net']:5.2f}", flush=True)

    # cost sensitivity on key variants (incl. naive reference)
    sens = {}
    for name in ['naive_all', 'xgb@0.50', 'xgb@0.55', 'topK50', 'band55/50']:
        sens[name] = {}
        for bps in [5, 10, 25, 50]:
            r, _, _ = backtest(P, variants[name], cost_rate=bps / 10000.0)
            sens[name][f'{bps}bps'] = {'cum_net': r['cumulative_return_net'], 'sharpe': r['sharpe_net']}
        print(f"cost sensitivity {name}: " + " | ".join(
            f"{k}={v['cum_net']:+.2f}" for k, v in sens[name].items()), flush=True)

    pd.DataFrame(curves).to_csv(ROOT / "reports/phase4_daily_equity.csv")
    out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'n_predictions': int(len(P)), 'cost_convention': 'turnover=0.5*sum|dw|, cost_rate per unit',
           'results': results, 'cost_sensitivity': sens}
    (ROOT / "reports/phase4_results.json").write_text(json.dumps(out, indent=2))
    print(f"DONE in {time.time()-t0:.0f}s -> reports/phase4_results.json", flush=True)


if __name__ == '__main__':
    main()
