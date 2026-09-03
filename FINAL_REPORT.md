# FINAL REPORT — NIFTY100 Market-Movement Prediction Research Program

**Project:** Indian Stock Market Movement Prediction (Guarvi1)
**Period:** 2026-08-26 → 2026-08-29 · **Status:** CLOSED (conclusion reached)
**Data:** NIFTY100, 98 symbols, 2015-12-31 → 2026-08-17, 249,983 clean rows, 62 base features
**Discipline:** every claim below is out-of-sample, pre-registered, and independently verified

---

## 1. Executive Summary

1. **A real but tiny directional edge exists**: XGBoost on 62 price/volume features achieves **51.85% accuracy vs 50.40% naive base rate (z = +16.8), AUC 0.5253** over 205,948 fully out-of-sample predictions in a continuous 51-block walk-forward. The edge is monotonic in model confidence (prob ≥ 0.60 ⇒ 59.4% accuracy on 2,447 predictions).
2. **The edge is not monetizable**: ~2 bps/day *gross*, while Indian retail round-trip costs are ~20–30 bps. Across Phases 4–8, **20+ portfolio designs were tested; zero passed** the pre-registered 2025–26 holdout at either 10 or 25 bps.
3. **All free information sources are exhausted**: 1-day technicals, 5-day horizons, market-state/sector features, and external macro series (VIX, S&P 500, USD/INR, crude) all fail to improve the base model (ΔAUC = −0.0053 to −0.0043, all negative or null).
4. **Best deployable strategy remains naive daily equal-weight NIFTY100**: 2025–26 holdout +5.3–6.1% net, Sharpe +0.28–0.31, ~zero turnover, cost-immune.
5. **The audit trail found and fixed 6 independent bugs**, several producing spectacular false results (4.37e21 "returns", AUC = 1.0) that would have invalidated the project had they gone undetected.

## 2. Program Timeline

| Phase | Lever tested | Result |
|---|---|---|
| Audit | Backtest verification | 🐛 Bugs #1–2 found; corrected: model −35.5%, MDD −47.9%, Sharpe −0.42 |
| 1A–1C | Feature additions, model tuning | Peak "53.4%" — later shown to be sampling noise |
| 2 | Honest re-evaluation | ❌ Threshold/ensemble/feature-selection gains = overfitting |
| 3 | Definitive walk-forward (205,948 OOS) | ✅ 51.85% / AUC 0.5253 / z +16.8; backtest bug #2 re-caught |
| 4 | Turnover & cost engineering | ❌ 11 variants fail holdout; 🐛 bug #3 (stack NaN) |
| 5 | Pre-registered portfolio designs | ❌ 0/7 pass; turnover 910→62 but signal absent in holdout |
| 6 | 5-day horizon labels | ❌ 0/2 pass; 🐛 bug #4 (overlap compounding); ranking-only ~2 bps/day |
| 7 | Market-state & sector features | ❌ Gate 1 fail (ΔAUC −0.0043, McNemar z −0.76); 🐛 bug #5 (label leakage, AUC=1.0 caught by alarm) |
| 8 | External macro (VIX/SPX/FX/Crude) | ❌ Gate 1 fail (ΔAUC −0.0053); protocol guard caught 71≠62 feature violation |

## 3. Bug Registry (audit value of the program)

| # | Bug | Symptom | Detection | Fix |
|---|---|---|---|---|
| 1 | Per-symbol returns compounded across 128,237 rows | Cum. return 4.37e21, MDD −1.0 | Mathematical impossibility audit | Aggregate by date first; equity_t = equity_{t-1}·(1+r_t) |
| 2 | Bad price data (BRITANNIA) | +14,593% single "return", equity 4.1e31 | Sanity assertions on daily returns | |ret| ≤ 0.5 filter + hard assertions |
| 3 | pandas stack() no longer drops NaN | Held-set silently inflated 205,948 → 217,872 | Phase-3 cross-validation mismatch | np.where + count assertions |
| 4 | Overlapping 5-day returns compounded daily | F0 "+73,400%", MDD −0.88 | Turnover 0.1 impossible for diversified book | 1-day returns of held names; 5d label only for membership |
| 5 | Label columns in feature set | OOS AUC = 1.0000, acc 100% | Pre-registered sanity alarm (acc > 0.60 abort) | Unified label denylist asserted on every feature set |
| 6 | Merged macro columns entered base set | Base 71 ≠ 62 protocol guard | 62-feature protocol assertion | Explicit macro-column exclusion from base set |

## 4. Methodology Standards Established

- **Audited backtest contract** (all phases): equal-weight by date, exposure ≤ 1.0, turnover = 0.5·Σ|Δw|, costs charged on turnover, Sharpe = mean/σ·√252 (NaN if σ=0), MDD ∈ [−1, 0], hard assertions on NaN/inf, return > −1, finite positive equity.
- **Pre-registration**: designs and PASS/FAIL rules declared before any holdout evaluation (Phases 5–8); gates stop evaluation early on failure — no p-hacking path.
- **Two-gate protocol**: Gate 1 = paired OOS model comparison (McNemar + ΔAUC, sanity alarm aborts on acc > 0.60 / AUC > 0.65); Gate 2 = holdout portfolios, only if Gate 1 passes.
- **Holdout discipline**: 2025-01-01 → end never used for selection; verdicts judged only there, at both 10 and 25 bps.
- **Independent verification**: every corrected implementation anchored against an independent prior result (e.g., Phase 6 naive F0 = +264.6% vs Phase 5 D1 = +287.2%, consistent to methodology differences).

## 5. Final Validated Numbers

**Model (62-feature XGBoost, continuous walk-forward, 205,948 OOS predictions):**

| Metric | Value |
|---|---|
| Accuracy @ 0.5 | 0.5185 (naive base rate 0.5040; z = +16.8) |
| ROC-AUC | 0.5253 |
| Confidence ≥ 0.55 / 0.60 / 0.70 | 54.9% (n=16,737) / 59.4% (n=2,447) / 62.7% (n=185) |
| Years beaten naive | 8 of 9 (2018–2026) |

**Strategies (2025–26 holdout, net):**

| Strategy | Net | Sharpe | Turnover | Verdict |
|---|---|---|---|---|
| Naive daily equal-weight NIFTY100 | **+5.3–6.1%** | **+0.28–0.31** | ~0 | ✅ benchmark stands |
| Best model design (frozen-21d, Ph5 D7) | +2.4% | +0.17 | 61.7 | ❌ loses to naive |
| All other tested designs (Ph4–8) | — | — | — | ❌ 0 passes |

## 6. Conclusions & Recommendations

1. **Keep**: the 62-feature XGBoost as a research/screening signal; the audited backtest library; the pre-registration discipline; all phase reports as negative-result documentation.
2. **Use**: naive daily equal-weight NIFTY100 as the production benchmark; do not deploy any model-driven strategy tested so far.
3. **Do not**: iterate further on price/volume features, horizons, sectors, or free macro series — each is now provably a dead end (6 consecutive Gate-1/holdout failures).
4. **If resuming**: only genuinely new data classes (FII/DII flows, fundamentals/earnings, news/sentiment — paid or licensed), with the same pre-registered two-gate protocol and the 2025–26 holdout now permanently retired as training data.

