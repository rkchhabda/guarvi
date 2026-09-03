# Phase 5: Pre-Registered Low-Turnover Monetization — ALL DESIGNS FAIL HOLDOUT

**Date:** 2026-08-28 · **Input:** Phase-3 OOS predictions (no retraining) · **Script:** `scripts/phase5_tilt.py`
**Outputs:** `reports/phase5_results.json`, `phase5_daily_equity.csv`

## Pre-registration (declared before evaluation)

8 designs fixed a priori (no search): daily naive, static buy-and-hold, continuous
tilt (t=0.5), tercile tilt (1.5/1.0/0.5) daily + weekly-frozen, binary xgb≥0.50
portfolio frozen weekly (w5) and every 21 days (w21), plus the Phase-4 daily binary
as a reproduction check.
**PASS rule:** beat naive daily equal-weight on 2025–2026 holdout **net return AND
Sharpe at BOTH 10 bps and 25 bps**. Judgment data: holdout only.

## Full period 2018–2026 (net @10 bps) — turnover reduction works mechanically

| Design | Net | Sharpe | MDD | Turnover | Holdout Sharpe |
|---|---|---|---|---|---|
| D0 xgb50 daily (reproduces Phase 4 ✓) | +3.049 | 0.92 | -0.38 | 910.1 | -0.35 |
| D1 naive daily | +2.872 | 0.97 | -0.39 | 10.7 | **+0.31** |
| D2 naive static buy&hold | +2.870 | 0.87 | -0.44 | 19.8 | +0.28 |
| D3 tilt continuous t=0.5 | +3.015 | 0.99 | -0.39 | **32.7** | +0.29 |
| D4 tercile tilt daily | +2.967 | 0.96 | -0.39 | 365.8 | +0.08 |
| D5 tercile tilt frozen w5 | +2.920 | 0.96 | -0.39 | 91.8 | +0.18 |
| D6 xgb50 frozen w5 | +3.337 | 0.98 | -0.45 | 210.7 | -0.07 |
| **D7 xgb50 frozen w21** | **+3.647** | **1.02** | -0.38 | **61.7** | +0.17 |

## Pre-registered holdout verdict (2025-01 → 2026-08)

| Design | H net @10 | H Sharpe @10 | H net @25 | H Sharpe @25 | Verdict |
|---|---|---|---|---|---|
| naive daily (reference) | **+0.061** | **+0.31** | **+0.056** | **+0.30** | — |
| D7 xgb50 w21 | +0.024 | +0.17 | +0.009 | +0.12 | **FAIL** |
| D3 tilt cont t0.5 | +0.057 | +0.29 | +0.050 | +0.27 | **FAIL** |
| D6 xgb50 w5 | -0.040 | -0.07 | -0.093 | -0.28 | **FAIL** |
| D5 tercile w5 | +0.026 | +0.18 | +0.000 | +0.08 | **FAIL** |
| D4 tercile daily | +0.000 | +0.08 | -0.110 | -0.36 | **FAIL** |
| D2 naive static | +0.057 | +0.28 | +0.053 | +0.27 | **FAIL** |
| D0 xgb50 daily | -0.115 | -0.35 | -0.328 | -1.34 | **FAIL** |

**7 of 7 designs FAIL.** No tested monetization of the model signal beats plain
naive daily equal-weight out-of-sample on the pre-registered rule.

## Interpretation

1. **Turnover reduction is achievable and cheap:** D7 cuts turnover 910 → 62 (15×)
   and converts the full-period picture to "best model variant" (+3.65 net, Sharpe
   1.02 vs naive 0.97). D3 tilt harvests the signal with only 32.7 turnover.
2. **But the signal itself was weak in 2025–2026.** Cost is no longer the binding
   constraint (D7 @25 bps still nets +0.9%) — the model's high-probability picks
   simply did not outperform in the holdout, consistent with the Phase-3 yearly
   table (2025–26 was the weakest stretch, and Phase 4 showed holdout Sharpe < 0
   for all binary variants).
3. **The pre-registration prevented a false positive.** Judged on the full period,
   D7 looks like a success (+0.78pp over naive, higher Sharpe). Judged on holdout
   as pre-committed, it fails. This mirrors the Phase-2 threshold lesson.
4. D0 exactly reproduces Phase 4's xgb@0.50 (net +3.049, turnover 910.1) — the
   audited backtest implementation is consistent across phases.
5. Naive daily equal-weight remains the best holdout strategy (Sharpe +0.31);
   daily rebalancing itself adds ~0.10 Sharpe over static buy-and-hold (0.97 vs 0.87).

## Program-level conclusion (Phases 1–5)

- The XGB model has a **real, statistically significant directional edge**
  (+1.44% accuracy vs naive, z=+16.8, SE 0.11%, monotonic in confidence: ≥0.60 ⇒ 59.4%).
- The edge has **not** been monetized: every portfolio design tested (daily binary,
  concentrated thresholds, hysteresis bands, top-K, tilts, frozen rebalances) fails
  the 2025–2026 holdout or dies at realistic Indian costs.
- Honest status: **research-grade signal, no deployable strategy.**

## Viable next options (require new walk-forward runs, i.e., retraining)

- **Longer-horizon labels (5–10 day direction):** structurally lower turnover and
  more learnable signal-to-noise; the most promising untried lever.
- **New information:** macro/sector/flow features — current 62 features are all
  price/volume, likely saturated.
- **Accept the negative result** and keep the model as a research/screening tool.

**Stop condition honored:** no new advanced models; audit-grade evaluation only.
