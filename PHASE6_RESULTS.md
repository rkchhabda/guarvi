# PHASE 6 RESULTS — 5-Day Horizon Labels (Pre-Registered Fresh Walk-Forward)

**Date:** 2026-08-28 · **Script:** `scripts/phase6_horizon5.py` · **Runtime:** 439s walk-forward + 18s evaluation
**Data:** 249,156 rows · 98 symbols · 2015-12-31 → 2026-08-17 · 62 features · XGB (Phase-3 hyperparameters)

## Pre-registration (declared before holdout evaluation)

| Item | Value |
|---|---|
| Label | `target_5d = close[t+5]/close[t] - 1 > 0` per symbol; \|5d ret\| ≤ 0.5 filter |
| Retrain | every 42 trading days, predict daily, 500-day warmup, 200K train cap (Phase-3 protocol) |
| Designs | **F0** naive hold-all frozen w5 (reference) · **F1** xgb≥0.5 frozen w5 · **F2** tercile tilt 1.5/1/0.5 frozen w5 |
| PASS rule | F1 or F2 beats F0 on 2025–26 holdout **net AND Sharpe at BOTH 10 and 25 bps** |

## Third bug found and fixed this phase (compounding family, again)

First run reported `F0 net=+734, MDD=-0.88` — impossible for a diversified portfolio. **Root cause:** the
backtest compounded **overlapping 5-day forward returns daily** (each price move t→t+5 counted 5×, once per
prediction date). Same failure class as the original 4.37e21 audit bug (Phase 9A) and the Phase-3 BRITANNIA
bug. **Fix:** portfolio daily return now uses each held symbol's **next-day (1-day) return** on each held
day; the 5-day label only determines membership. A 5-day position earns 5 daily returns ≈ its 5-day return,
each counted exactly once. New assertions: `|daily net| ≤ 0.15`, `turnover < 5000`, plus the full Phase-3
assertion set (exposure ≤ 1, MDD ∈ [-1,0], equity finite/positive, no NaN/inf, no duplicate date-symbol).

**Validation anchor:** corrected F0 = **+264.6%** net, Sharpe 0.95, turnover 0.15 — consistent with Phase 5's
independent D1 daily equal-weight implementation (+287.2%, Sharpe 0.97, turnover 10.7; small gap = 5-day
frozen weights). Accounting independently corroborated. ✓

## Model quality on 5-day labels (205,221 OOS predictions, 51 retrain blocks)

| Metric | 5-day | 1-day (Phase 3, for reference) |
|---|---|---|
| Accuracy @0.5 | **51.93%** | 51.85% |
| Base rate (up) | 52.30% (drift-inflated) | 50.40% |
| z vs base rate (overlap-adj ÷√5) | **−1.5** (nominal −3.4) | +16.8 |
| AUC | **0.5113** | 0.5253 |
| Tercile directional accuracy | 51.05% → 52.72% → 53.13% | monotonic, stronger |
| Top−bottom tercile spread | **+10.6 bps per 5 days (gross)** | larger |

**Interpretation:** at the 5-day horizon the threshold-0.5 accuracy is *below* the drift-inflated base rate
(z=−1.5). The only surviving signal is the **cross-sectional ranking** (monotone terciles, ~10.6 bps/5d
gross) — roughly 2 bps/day gross before costs, which is the same weak edge already known from 1-day labels.

## Corrected portfolio results (net; gross in parentheses where useful)

| Design | Full net | Full Sharpe | MDD | Turnover | Avg names | Holdout net @10 | Holdout Sharpe @10 | Holdout net @25 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| F0 naive hold-all w5 | +264.6% | 0.95 | −0.377 | 0.15 | 98.6 | **+5.3%** | **+0.28** | +5.3% | reference |
| F1 xgb≥0.5 w5 | +305.2% | 0.97 | −0.418 | 119.0 | 72.0 | −0.9% | +0.04 | −3.7% | ❌ FAIL |
| F2 tercile tilt w5 | +305.6% (+338.5% gross) | 0.99 | −0.390 | 78.0 | 96.3 | +5.3% (tie) | +0.28 (tie) | **+3.2%** | ❌ FAIL |

**Pre-registered verdict: 0/2 PASS.** F1 loses outright on the holdout. F2 exactly ties F0 at 10 bps
(+5.3%/+0.28) and loses at 25 bps (+3.2%/+0.20) — the ~4.1% full-period gross edge over F0 (+338.5% vs
+264.7%) is consumed by costs and does not survive the holdout. In-sample vs holdout Sharpe for F2:
1.13 → 0.28 (same regime dependence as Phases 4–5).

## Program conclusion after Phases 3–6 (four independent fresh walk-forwards)

1. **The statistical edge is real and robust**: monotone probability ranking, AUC 0.51–0.53, z up to +16.8,
   reproduced across horizons, thresholds, and 51-block continuous walk-forwards.
2. **It is not monetizable at Indian retail costs (~20–30 bps round-trip)** at daily or 5-day rebalance
   frequency: the ranking carries only ~2 bps/day of gross alpha, and every portfolio design that
   concentrates it fails the 2025–26 holdout (Phases 4, 5, 6: 0 of 12 designs PASS).
3. **Naive daily/every-5-day equal-weight NIFTY100 remains the best strategy** on the holdout
   (+5.3% net, Sharpe +0.28 at 25 bps).
4. The honest use of the model is as a **research/screening signal** (e.g., tilting within a low-turnover
   core would need to be evaluated on a fresh pre-registered holdout after 2026-08 data accrues).

## Deliverables

- `scripts/phase6_horizon5.py` — pre-registered walk-forward + overlap-safe backtest
- `reports/phase6_predictions.csv` — 205,221 OOS predictions (date, symbol, actual, ret_5d, ret_1d, prob)
- `reports/phase6_results.json` — all metrics, terciles, cost sensitivity, verdicts, pre-registration text
- `reports/phase6_daily_equity.csv` — daily net returns per design
- `reports/phase6_run.log`, `reports/phase6_run2.log` — run logs (run1 = buggy accounting, kept for audit trail)
