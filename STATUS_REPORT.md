# STATUS REPORT — Indian Stock Market Movement Prediction

**Generated:** 2026-08-29 · **Project root:** `c:\Users\r_chh\OneDrive\Apps\Guarvi1`
**Overall status:** RESEARCH PROGRAM CLOSED — conclusion reached after 8 pre-registered phases
**Health:** all 61 tests pass · no stray processes · all deliverables on disk

---

## 1. Project Overview

Machine-learning research project predicting next-day directional movement of NIFTY100 stocks (98 symbols, 2015-12-31 → 2026-08-17, 249,983 clean rows, 62 base features) and evaluating whether the signal is monetizable under Indian retail transaction costs.

- **Stack:** Python 3.14, pandas/numpy, XGBoost (hist), scikit-learn, yfinance, pytest
- **Entry points:** `scripts/run_nifty100_pipeline.py` (original), `scripts/phase3_definitive_eval.py` (reference walk-forward), `scripts/phase8_external_macro.py` (latest)
- **Validation:** continuous walk-forward, retrain every 42 trading days, 500-day warmup, 200K-row train cap, 205,948 OOS predictions, 2025-01-01→end holdout never used for selection

## 2. Current Status at a Glance

| Item | Status |
|---|---|
| Data pipeline (download → features → labels) | ✅ complete, validated |
| Model with real OOS edge | ✅ 62-feat XGBoost: 51.85% acc vs 50.40% naive (z=+16.8), AUC 0.5253 |
| Monetizable strategy | ❌ 20+ designs tested, 0 pass holdout at 10/25 bps |
| Naive benchmark | ✅ best strategy: holdout +5.3–6.1% net, Sharpe +0.28–0.31 |
| Free information levers | ❌ exhausted (technicals, 5-day, sector, macro) |
| Test suite | ✅ 61/61 pass |
| Documentation | ✅ 14 root MD files + per-phase reports |

## 3. Phase-by-Phase Status

| Phase | Lever | Outcome |
|---|---|---|
| Audit (9A) | Backtest verification | 2 bugs fixed (4.37e21 compounding; bad prices); corrected model: −35.5%, Sharpe −0.42 |
| 1A–1C | Features + tuning | Early "53.4%" later shown to be sampling noise (3K-pred windows) |
| 2 | Honest re-eval | Threshold/ensemble/feature-selection gains = overfitting |
| 3 | Definitive walk-forward | ✅ validated edge: 51.85%, AUC 0.5253, z +16.8, 8/9 years |
| 4 | Turnover/cost engineering | ❌ 11 variants fail holdout (concentration + regime + cost fragility) |
| 5 | Pre-registered designs | ❌ 0/7 pass; turnover 910→62 didn't help |
| 6 | 5-day horizon | ❌ 0/2 pass; ranking-only (~2 bps/day gross) |
| 7 | Market-state/sector | ❌ Gate 1 fail (ΔAUC −0.0043) |
| 8 | External macro | ❌ Gate 1 fail (ΔAUC −0.0053) |

## 4. Quality & Audit Summary

Six bugs were found and fixed during the program — the single most valuable byproduct of this work:

1. **Per-symbol compounding** across 128,237 rows → 4.37e21 "cumulative return"
2. **Bad price data** (BRITANNIA) → fake +14,593% returns, equity 4.1e31
3. **pandas stack() NaN change** → held-set silently inflated 205,948 → 217,872
4. **Overlapping 5-day returns compounded daily** → impossible "+73,400%" with MDD −0.88
5. **Label leakage** in feature set → OOS AUC = 1.0000 (caught by pre-registered sanity alarm)
6. **Protocol violation** after data merge → base feature count 71 ≠ 62 (caught by assertion)

Every current number is backed by hard assertions (no NaN/inf, returns > −1, exposure ≤ 1.0, MDD ∈ [−1,0], finite positive equity) and independently cross-checked between phase implementations.

## 5. Repository Layout (key artifacts)

```
Guarvi1/
├── STATUS_REPORT.md, FINAL_REPORT.md, PROJECT_STATUS.md   ← status & conclusions
├── PHASE3..8_RESULTS.md                                   ← per-phase findings
├── data/processed/nifty100_features.parquet               ← panel (249,983 rows, 62 feats)
├── reports/
│   ├── backtest_audit.md/.json/.csv                       ← Phase 9A audit
│   ├── phase3_predictions.csv (205,948 OOS, reference)
│   ├── phase{4..8}_results.json, _summary.md, _run.log    ← per-phase deliverables
│   └── phase6/7/8_predictions.csv                         ← OOS prediction caches
├── scripts/
│   ├── audit_backtest.py, phase3_definitive_eval.py       ← audited core
│   ├── phase4_turnover.py … phase8_external_macro.py      ← lever tests
│   └── enhanced_train.py, data_quality.py                 ← utilities
├── src/market_ml/ (baselines, evaluation, features, labels, splits)
└── tests/ (61 tests, all passing)
```

## 6. Limitations

- Edge is cross-sectional ranking only (~2 bps/day gross); threshold accuracy at 5-day horizon is *below* base rate.
- All results are specific to NIFTY100, 2018–2026, daily frequency, yfinance data quality.
- The 2025–26 holdout is now burned for selection purposes; future tests need data accruing after 2026-08-29 or a fresh holdout.
- No shorting, no intraday, no market-impact modeling; costs assumed proportional to turnover.

## 7. Recommended Next Actions

1. **Archive** the program as-is (documents are the deliverable) — recommended.
2. If resuming, **acquire new data classes only** (FII/DII flows, fundamentals, news sentiment), re-using the audited backtest and two-gate protocol.
3. Re-run `python -m pytest tests/` after any change; regenerate phase reports only via the committed scripts.

## 8. Reproduction (validated commands)

```powershell
cd c:\Users\r_chh\OneDrive\Apps\Guarvi1
python -m pytest tests/ -q                 # 61 passed
python scripts/phase3_definitive_eval.py   # reference walk-forward (~20 min)
python scripts/phase8_external_macro.py    # latest lever test (~25 min)
```

