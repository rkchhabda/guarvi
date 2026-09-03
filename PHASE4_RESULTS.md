# Phase 4: Turnover Reduction & Cost-Aware Strategy Design

**Date:** 2026-08-28 · **Input:** 205,948 OOS predictions from Phase 3 (no retraining)
**Script:** `scripts/phase4_turnover.py` · **Outputs:** `reports/phase4_results.json`, `phase4_daily_equity.csv`
**Cost convention (same as Phase 3):** turnover = 0.5·Σ|Δw| per day, cost = rate × turnover.

## Headline verdict

**The spectacular high-confidence backtests are in-sample artifacts. Do not deploy.**

The monotonic accuracy edge is real (verified below), but every attempt to monetize it
as a daily-rebalanced concentrated portfolio fails the 2025–2026 holdout and dies at
realistic Indian transaction costs.

## 1. Diagnostic bug found and fixed (methodology note)

The first run reported `held_acc = 0.5040` for **every** variant. Root cause: in the
installed pandas version, `DataFrame.stack()` no longer drops NaN, so the "held pairs"
index silently expanded from 205,948 real positions to the full 217,872-cell
date×symbol grid (empty IPO/pre-listing cells included) — every row looked "held".
Fixed with an explicit `np.where(W.values > 0)` pair construction plus a hard
count-assertion. Verified against Phase 3 confidence buckets:

| Held set | n | Held accuracy | Phase 3 reference |
|---|---|---|---|
| naive (all) | 205,948 | 50.40% | 50.40% ✓ |
| xgb ≥ 0.50 | 119,690 | 51.94% | 51.85% ✓ |
| xgb ≥ 0.55 | 19,108 | 54.88% | ~54.2% ✓ |
| xgb ≥ 0.60 | 2,447 | 59.38% | ~59.1% ✓ |

**The probability→accuracy relationship is genuine and monotonic.** That part survives audit.

## 2. Variant results (full period, net of 10 bps)

| Variant | Net cum | Gross cum | Sharpe | MDD | Turnover | Avg names held | IS Sharpe 18–24 | Holdout Sharpe 25–26 |
|---|---|---|---|---|---|---|---|---|
| naive (buy all) | +2.87 | +2.91 | **0.97** | -0.39 | 10.7 | ~97 | 1.10 | **+0.31** |
| xgb@0.50 | +3.05 | +9.06 | 0.92 | -0.38 | 910 | ~56 | 1.17 | -0.35 |
| xgb@0.52 | +2.58 | +12.86 | 0.74 | -0.43 | 1,356 | — | 1.02 | -0.87 |
| xgb@0.55 | +36.7 | +163.8 | 1.35 | -0.64 | 1,477 | 8.9 | 1.83 | -1.44 |
| xgb@0.58 | +806.6 | +1,884 | 2.10 | -0.47 | 852 | ~1.3 | 2.39 | -0.06 |
| xgb@0.60 | +884.6 | +1,480 | 2.11 | -0.36 | 517 | **1.1** | 2.42 | -0.33 |
| topK25 | +4.41 | +18.5 | 1.02 | -0.42 | 1,282 | 25 | 1.37 | -0.76 |
| topK50 | +3.47 | +11.4 | 0.96 | -0.37 | 1,021 | ~50 | 1.22 | -0.37 |
| band55/50 | +10.99 | +24.0 | 1.20 | -0.42 | 735 | — | 1.53 | -0.65 |
| band55/50 hold5 | +6.72 | +11.8 | 1.11 | -0.42 | 505 | — | 1.36 | -0.35 |
| band60/55 | +687.1 | +1,117 | 2.04 | -0.35 | 488 | 1.7 | 2.30 | +0.14 |

## 3. Why the +88,000% numbers are not real edge — three independent kills

**(a) Concentration.** xgb@0.60 holds **1.1 names on average** at 100% exposure — a
lottery, not a portfolio. Daily gross std 2.6%.

**(b) Regime dependence.** Yearly net returns:

| Variant | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025* | 2026* |
|---|---|---|---|---|---|---|---|---|---|
| naive | -0.10 | +0.10 | +0.32 | +0.52 | +0.11 | +0.37 | +0.21 | +0.05 | +0.01 |
| xgb@0.50 | -0.12 | +0.45 | +0.57 | +0.34 | +0.07 | +0.35 | +0.19 | -0.06 | -0.06 |
| xgb@0.55 | +0.30 | +0.57 | **+5.20** | +1.01 | +0.45 | +0.78 | +0.15 | -0.33 | -0.26 |
| xgb@0.60 | +1.42 | +1.30 | **+7.01** | +2.85 | +1.61 | +0.34 | +0.64 | -0.07 | -0.03 |
| band60/55 | +0.63 | +1.67 | **+6.64** | +2.50 | +1.25 | +0.46 | +0.77 | -0.09 | +0.12 |

*2025–2026 = holdout (thresholds/designs were never fit to it, but were selected with
knowledge of 2018–2024 = multiple testing). Essentially all model profits come from the
2020–2022 extreme-volatility bull; every model variant is flat-to-negative in holdout
while naive buy-and-hold stays positive.

**(c) Cost fragility.** Cumulative net return vs cost level:

| Variant | 5 bps | 10 bps | 25 bps | 50 bps |
|---|---|---|---|---|
| naive | +2.89 | +2.87 | +2.81 | **+2.71** |
| xgb@0.50 | +5.38 | +3.05 | **+0.03** (break-even) | -0.89 |
| xgb@0.55 | +77.9 | +36.7 | +3.13 | -0.90 |
| topK50 | +6.44 | +3.47 | -0.03 | -0.92 |
| band55/50 | +16.3 | +10.99 | +2.98 | -0.37 |

At realistic Indian retail costs (brokerage+STT+slippage ≈ 20–30 bps round-trip) the
daily-rebalanced model strategy ≈ **zero**, while naive buy-and-hold is cost-immune.

**(d) Concentration source (data check).** Top contributors of xgb@0.60 are
ultra-volatile momentum names — IRFC (contribution +3.46 of daily-return mass,
mean |daily move| 2.35%, max +48.3%), SHRIRAMFIN (mean |move| 3.17%), ADANIPOWER,
CGPOWER. Mean |ret| values are high but plausible for these names (no smoking-gun bad
data), i.e. the model genuinely concentrates in the highest-vol stocks — exactly the
positions where a 59% accuracy edge translates into huge in-sample compounding and
into ruin when the regime turns.

## 4. What remains true after Phase 4

1. **Accuracy edge (Phase 3): REAL** — +1.44% vs naive, z=+16.8, 8 of 9 years, SE 0.11%.
2. **Confidence conditioning: REAL** — prob≥0.60 ⇒ 59.4% accuracy (n=2,447).
3. **Daily-rebalanced monetization: NOT REAL** — negative holdout Sharpe, dead at ≥25 bps.
4. **Best risk-adjusted full-period strategy is naive buy-and-hold** (Sharpe 0.97).

## 5. Recommendations (next phase candidates)

- **A. Tilt overlay (recommended):** keep a buy-and-hold NIFTY100 core; use model
  probability as a small over/underweight tilt. Turnover stays ~10 (cost-immune),
  edge is harvested without daily churn.
- **B. Weekly/monthly rebalance:** cuts turnover 5–20×; test with holdout discipline.
- **C. Longer-horizon labels:** predict 5–10-day direction to structurally cut turnover.
- **D. Any new design must pre-register its rule and be judged ONLY on 2025–2026.**

**Stop condition honored:** no new models trained; audit and honest evaluation only.
