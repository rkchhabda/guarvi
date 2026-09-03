#!/usr/bin/env python3
"""Phase 5: Pre-registered low-turnover monetization designs.

PRE-REGISTRATION (declared before any holdout evaluation):
  Decision data: 2025-01-01 .. end holdout ONLY. 2018-2024 shown as context.
  Designs fixed a priori (no parameter search):
    D1 naive_daily      : equal-weight all tradable names, daily rebalance (reference)
    D2 naive_static     : true buy-and-hold, drifted weights, no rebalancing
    D3 tilt_cont_t0.5   : w_i ~ clip(1 + 0.5*(2*p-1), 0.1, inf), daily renorm
    D4 tilt_tercile     : top/mid/bottom prob tercile -> 1.5/1.0/0.5, daily renorm
    D5 tilt_tercile_w5  : same as D4, weights frozen every 5 trading days
    D6 xgb50_w5         : binary prob>=0.50 portfolio, frozen every 5 trading days
    D7 xgb50_w21        : binary prob>=0.50 portfolio, frozen every 21 trading days
    D0 xgb50_daily      : phase-4 binary reference (sanity reproduction check)
  PASS rule: design beats naive_daily on holdout net return AND Sharpe at BOTH
  10 bps and 25 bps. Only a PASS may be considered for further research.
No retraining. Reuses Phase-3 OOS predictions and the audited Phase-4 backtest().
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

COST_RATE = 0.001  # 10 bps per unit turnover (0.5*sum|dw| convention, as Phases 3-4)


def backtest(P, W, cost_rate=COST_RATE):
    """Audited methodology: equal-weight book, exposure<=1, costs on turnover."""
    all_dates = np.sort(P['date'].unique())
    W = W.reindex(all_dates).fillna(0.0)
    held = W.gt(0)
    R = P.pivot_table(index='date', columns='symbol', values='next_ret',
                      aggfunc='mean').reindex(all_dates)
    gross = (W * R.fillna(0.0)).sum(axis=1)
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - cost_rate * turnover
    assert (P['next_ret'].abs() <= 0.5 + 1e-12).all(), "per-symbol |return| > 0.5"
    assert np.isfinite(net).all(), "NaN/inf in daily net returns"
    assert (net.abs() <= 0.5 + 1e-12).all(), "absurd daily portfolio return"
    assert (net > -1).all(), "daily return <= -1"
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all(), "daily exposure exceeds 1.0"
    eq = (1.0 + net).cumprod()
    assert np.isfinite(eq).all() and (eq > 0).all(), "equity not finite/positive"
    mdd = float((eq / eq.cummax() - 1.0).min())
    assert -1.0 <= mdd <= 0.0, "MDD outside [-1, 0]"
    sd = net.std(ddof=0)
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else float('nan')
    return {
        'cumulative_return_net': float(eq.iloc[-1] - 1.0),
        'cumulative_return_gross': float((1.0 + gross).prod() - 1.0),
        'max_drawdown': mdd, 'sharpe_net': sharpe, 'final_equity': float(eq.iloc[-1]),
        'total_turnover': float(turnover.sum()),
        'n_position_days': int(held.sum().sum()),
        'n_active_mean': float((W > 0).sum(axis=1).mean()),
        'n_days': int(len(all_dates)),
    }, eq, net


def main():
    t0 = time.time()
    print("PHASE 5: PRE-REGISTERED LOW-TURNOVER DESIGNS", flush=True)
    P = pd.read_csv(ROOT / "reports/phase3_predictions.csv")
    P['date'] = pd.to_datetime(P['date'])
    P = P.sort_values(['date', 'symbol']).reset_index(drop=True)
    print(f"predictions={len(P):,} dates={P['date'].nunique():,}", flush=True)

    R = P.pivot_table(index='date', columns='symbol', values='next_ret', aggfunc='mean')
    T = P.assign(t=1.0).pivot_table(index='date', columns='symbol', values='t', aggfunc='max')
    probs = P.pivot_table(index='date', columns='symbol', values='prob_xgb', aggfunc='mean')

    def freeze(pv, step):
        return pv.iloc[::step].reindex(pv.index).ffill()

    def norm(raw):
        W = raw.div(raw.sum(axis=1), axis=0)
        assert (W.sum(axis=1).fillna(0.0) <= 1.0 + 1e-9).all(), "exposure > 1 in builder"
        return W.fillna(0.0)

    rk = probs.rank(axis=1, pct=True)
    tw = pd.DataFrame(1.0, index=probs.index, columns=probs.columns)
    tw[rk >= 2.0 / 3.0] = 1.5
    tw[rk <= 1.0 / 3.0] = 0.5
    idx = (1.0 + R.fillna(0.0)).cumprod()

    variants = {
        'D0_xgb50_daily': norm(((probs >= 0.5).astype(float)).where(T.notna())),
        'D1_naive_daily': norm(T.copy()),
        'D2_naive_static': norm(idx.shift(1).fillna(1.0).where(T.notna())),
        'D3_tilt_cont_t0.5': norm((1.0 + 0.5 * (2.0 * probs - 1.0)).clip(lower=0.1).where(T.notna())),
        'D4_tilt_tercile': norm(tw.where(T.notna())),
        'D5_tilt_tercile_w5': norm(freeze(tw, 5).where(T.notna())),
        'D6_xgb50_w5': norm(freeze((probs >= 0.5).astype(float), 5).where(T.notna())),
        'D7_xgb50_w21': norm(freeze((probs >= 0.5).astype(float), 21).where(T.notna())),
    }

    is_dates = pd.DatetimeIndex(P.loc[P['date'] <= '2024-12-31', 'date'].unique())
    h_dates = pd.DatetimeIndex(P.loc[P['date'] > '2024-12-31', 'date'].unique())
    results, curves = {}, {}
    for name, W in variants.items():
        full, eq, net = backtest(P, W)
        is_r, _, _ = backtest(P[P['date'].isin(is_dates)], W.loc[W.index.isin(is_dates)])
        h_r, _, _ = backtest(P[P['date'].isin(h_dates)], W.loc[W.index.isin(h_dates)])
        sens = {}
        for bps in [5, 10, 25, 50]:
            r, _, _ = backtest(P, W, cost_rate=bps / 10000.0)
            sens[f'{bps}bps'] = {'cum_net': r['cumulative_return_net'],
                                 'sharpe': r['sharpe_net']}
        results[name] = {'full': full, 'in_sample_2018_2024': is_r,
                         'holdout_2025_2026': h_r, 'cost_sensitivity': sens}
        curves[name] = net
        s10, s25 = sens['10bps'], sens['25bps']
        print(f"{name:20s} net={full['cumulative_return_net']:+8.3f} sh={full['sharpe_net']:5.2f} "
              f"mdd={full['max_drawdown']:6.3f} to={full['total_turnover']:7.1f} "
              f"nact={full['n_active_mean']:5.1f} | H sh={h_r['sharpe_net']:5.2f} | "
              f"@10 net={s10['cum_net']:+8.3f} sh={s10['sharpe']:5.2f} | "
              f"@25 net={s25['cum_net']:+8.3f} sh={s25['sharpe']:5.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return P, variants, results, h_dates


def verdicts(P, variants, results, h_dates):
    """Pre-registered PASS rule: beat naive_daily on HOLDOUT net AND Sharpe
    at BOTH 10 and 25 bps."""
    Ph = P[P['date'].isin(h_dates)]
    out = {}
    h1 = {}
    for bps in (0.001, 0.0025):
        r, _, _ = backtest(Ph, variants['D1_naive_daily'].loc[
            variants['D1_naive_daily'].index.isin(h_dates)], cost_rate=bps)
        h1[f'{bps}'] = r
    for name, W in variants.items():
        if name == 'D1_naive_daily':
            continue
        Wh = W.loc[W.index.isin(h_dates)]
        v = {}
        for bps in (0.001, 0.0025):
            r, _, _ = backtest(Ph, Wh, cost_rate=bps)
            v[f'{bps}'] = {'net': r['cumulative_return_net'], 'sharpe': r['sharpe_net'],
                           'naive_net': h1[f'{bps}']['cumulative_return_net'],
                           'naive_sharpe': h1[f'{bps}']['sharpe_net']}
        ok = all(v[b]['net'] > v[b]['naive_net'] and v[b]['sharpe'] > v[b]['naive_sharpe']
                 for b in v)
        out[name] = {'pass': bool(ok), **{f'holdout_{k}@{b}': v[b][k]
                                          for b in v for k in ('net', 'sharpe', 'naive_net', 'naive_sharpe')}}
        print(f"VERDICT {name:20s} {'PASS' if ok else 'FAIL'} | "
              f"H@10 net={v['0.001']['net']:+.3f} sh={v['0.001']['sharpe']:+.2f} | "
              f"H@25 net={v['0.0025']['net']:+.3f} sh={v['0.0025']['sharpe']:+.2f} | "
              f"naive H@10 net={v['0.001']['naive_net']:+.3f} sh={v['0.001']['naive_sharpe']:+.2f}",
              flush=True)
    return out


if __name__ == '__main__':
    t0 = time.time()
    P, variants, results, h_dates = main()
    verd = verdicts(P, variants, results, h_dates)
    results['_verdicts'] = verd
    results['_pre_registration'] = ('designs fixed a priori; judged only on 2025-2026 '
                                    'holdout vs naive_daily at 10 and 25 bps')
    (ROOT / "reports/phase5_results.json").write_text(json.dumps(results, indent=2))
    print(f"DONE in {time.time()-t0:.0f}s -> reports/phase5_results.json", flush=True)

