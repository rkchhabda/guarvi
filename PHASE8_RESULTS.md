# Phase 8 Results — External Macro Features: GATE 1 FAIL

**Status: research program conclusion reached.** Pre-registered two-gate test, 2026-08-29.

## What was tested

Nine free external macro features (India VIX level/change/z-20/high-regime, S&P 500 1d & 5d lagged returns, USD/INR 1d lagged return, crude 1d & 5d lagged returns), all downloaded via yfinance and lagged at least one session to guarantee no lookahead. Merged onto the NIFTY100 panel by date (coverage ≥ 99.66%). Compared against the exact Phase-3 base protocol (62 features, XGB 150×depth-4, retrain every 42 days, 205,948 paired OOS predictions, 51 blocks).

## Results (Gate 1 — paired, sanity-alarmed, leakage-guarded)

| Arm | Features | AUC | Accuracy | z |
|---|---|---|---|---|
| Base | 62 | 0.5253 | 0.5185 | +13.11 |
| + Macro | 71 | 0.5200 | 0.5161 | +10.93 |

ΔAUC = **−0.0053** · McNemar b=39,734 / c=39,239, z=+1.76 (favors base) · **GATE 1: FAIL** → Gate 2 (holdout portfolios) never evaluated by design.

## Bugs caught during this phase (running tally: 6)

1. Feature-name mismatch (`usdinr_chg1d_lag` vs generated `usdinr_ret1d_lag`) — KeyError, fixed.
2. Protocol guard correctly tripped when merged macro columns entered the base feature set (71 ≠ 62) — base set explicitly excluded macro columns; assertions now enforce both the 62-feature protocol and the label denylist.

## Final program scoreboard (Phases 3–8, six walk-forwards, 20+ designs)

| Phase | Lever | Result |
|---|---|---|
| 3 | 1-day technicals (62 feats) | ✅ Best signal: AUC 0.5253, acc 51.85%, z +16.8 — research-only |
| 4 | Turnover/cost engineering | ❌ All designs fail holdout |
| 5 | Pre-registered portfolio designs | ❌ 0/7 pass holdout |
| 6 | 5-day horizon labels | ❌ 0/2 pass; ranking only (~2 bps/day gross) |
| 7 | Market-state & sector features | ❌ Gate 1 fail (ΔAUC −0.0043) |
| 8 | External macro (VIX/SPX/FX/Crude) | ❌ Gate 1 fail (ΔAUC −0.0053) |

**Bottom line:** the validated edge never exceeded ~2 bps/day gross at AUC ≈ 0.525; nothing monetizes at Indian retail costs; **naive daily equal-weight NIFTY100 remains the best strategy** (2025–26 holdout: +5.3–6.1% net, Sharpe +0.28–0.31). Further gains require new data classes (FII/DII flows, fundamentals, news sentiment — paid/licensed), not more modeling.
