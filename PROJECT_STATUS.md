# Indian Stock Market Movement Prediction Project
## Last Update and Current Status Report
**Generated:** August 28, 2026

---

## 1. Project Overview

This project builds a directional prediction system for Indian stocks/indices using OHLCV data, technical indicators, and walk-forward validation. The goal is to predict next-day direction (up/down) where 1 indicates tomorrow's close > today's close.

### Key Features:
- **Validation Methodology:** Expanding-window walk-forward only (no random train/test splits)
- **Target:** Directional accuracy (1 if tomorrow's close > today's close, else 0)
- **Models Evaluated:** Logistic Regression, Random Forest, XGBoost, plus baselines
- **Features:** Returns, log returns, SMA/EMA ratios, RSI, MACD, Bollinger Bands, ATR, rolling volatility, volume features
- **Data Source:** Yahoo Finance OHLCV data (adjusted) via jugaad-data/yfinance (no credentials bundled)

---

## 2. Last Update Summary

Based on file modification timestamps, the most recent activity occurred on **August 28, 2026**:

### Most Recently Modified Files:
- `src/market_ml/baselines.py` - **August 28, 2026, 08:51:41** (43,369 bytes)
- `src/market_ml/__pycache__/baselines.cpython-314.pyc` - **August 28, 2026, 08:51:49** (54,816 bytes)

### Recent Activity Timeline:
- **August 26, 2026:** Major data processing and model training batch
  - Feature engineering completed (18:29:32)
  - NIFTY100 feature dataset generated (291 MB CSV, 126 MB Parquet)
  - Evaluation reports created (12:59:13 for NIFTY100)
  - Baseline model training scripts updated
- **August 27-28, 2026:** Baseline model refinement and testing
  - Baselines module significantly updated (43KB file)
  - Likely involved testing/tuning of baseline models

---

## 3. Current Status

### ✅ Completed Work:
1. **Data Pipeline:** Fully functional
   - Raw data download scripts (`scripts/build_dataset.py`, `download.py`)
   - Feature engineering pipeline (`scripts/build_features.py`, `feature_engineering.py`)
   - Quality validation and reporting

2. **Model Training & Evaluation:**
   - Baseline models implemented and evaluated (Always-up, Previous-direction, Logistic Regression, Random Forest, XGBoost)
   - Walk-forward validation framework with 12-month holdout period
   - Comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, Brier score, calibration)
   - Trading simulation metrics (hit rate, returns, drawdown, Sharpe ratio)

3. **Reporting System:**
   - Detailed markdown and JSON evaluation reports
   - Feature quality reports
   - Data snapshots and schema documentation
   - Symbol-wise and monthly performance analysis

4. **Infrastructure:**
   - Modular code structure in `src/market_ml/`
   - Configuration management (`config/`)
   - Test suite (`tests/`)
   - Processed data storage (`data/processed/`)
   - Model artifacts storage (`models/`)

### 📊 Latest Results (NIFTY100 Evaluation - August 26, 2026):

#### Overall Performance:
- **Directional Accuracy:** 0.5048 (50.48%)
- **Always-Up Baseline:** 0.5092 (50.92%)
- **Edge vs Baseline:** -0.0044 (-0.44 percentage points)
- **ROC-AUC:** 0.5063
- **Brier Score:** 0.2729
- **Calibration:** Well-calibrated

#### Trading Simulation (Before Costs):
- **Hit Rate:** 0.5173
- **Average Return per Buy Signal:** 0.000957 (0.0957%)
- **Cumulative Return:** 4.37×10²¹ (extreme value due to compounding)
- **Max Drawdown:** -1.0
- **Sharpe Ratio (Annualized):** 0.544

#### Model Comparison:
*Note: Specific baseline model results from the August 28 update are not yet in reports, but the baselines.py file indicates recent work on:*
- Always-up baseline
- Previous-direction baseline  
- Logistic Regression (with StandardScaler)
- Random Forest
- XGBoost

#### Data Characteristics:
- **Evaluation Period:** 2021-03-23 to 2026-08-25
- **Symbols Evaluated:** 98 (from NIFTY100 universe)
- **Out-of-Sample Predictions:** 128,237
- **Features Engineered:** 62 columns
- **Total Rows Processed:** ~250,000 (after cleaning)

---

## CURRENT STATUS — Post Phase 5 (2026-08-28, evening)

> Phases 2–5 supersede the original numbers above. Authoritative docs:
> `PHASE3_RESULTS.md`, `PHASE4_RESULTS.md`, `PHASE5_RESULTS.md`, `reports/backtest_audit.md`.

### Definitive model evaluation (Phase 3 — low-noise continuous walk-forward)
- 205,948 OOS predictions, 2,136 days, 51 retrain blocks, binomial SE 0.11%
- XGBoost: **51.85%** accuracy, AUC 0.5253, **edge vs naive +1.44%, z=+16.8** (8 of 9 years)
- Naive (always-up): 50.40% · Logistic: 50.74% · Confidence monotonic: prob≥0.60 ⇒ 59.4% (n=2,447)

### Bugs found & fixed during audit phases
1. **4.37e21 cumulative return**: compounded 128K symbol-day rows sequentially → fixed to
   daily equal-weight portfolio aggregation (corrected: -35.5%, MDD -47.9%, Sharpe -0.42).
2. **Bad price data** (BRITANNIA): fake +14,593% returns → equity 4.1e31 → fixed with
   |ret|≤0.5 filter + hard assertions (equity finite, MDD∈[-1,0], exposure≤1).
3. **pandas stack() NaN behavior**: silently expanded "held sets" to the full date×symbol
   grid → fixed with np.where + count assertions (held accuracies now verified).

### Monetization status (Phases 4–5): **no deployable strategy**
- Daily-rebalanced binary portfolios die at ~25 bps costs (turnover 910).
- High-threshold concentration (+88,000% full-period) = in-sample artifact: 1.1 avg names,
  2020-regime dependent, negative 2025–26 holdout Sharpe.
- Pre-registered Phase 5 (8 designs: tilts, terciles, frozen w5/w21): **7/7 FAIL** the
  2025–26 holdout vs naive daily equal-weight (best: D7 frozen-21d, net +2.4% vs naive +6.1%).
- Turnover reduction works mechanically (910 → 32–62) but the signal was weak in holdout.

### Honest bottom line
Research-grade directional edge (statistically robust) exists; **it does not survive to
positive holdout active returns under any tested portfolio design at Indian retail costs.**
Naive daily equal-weight NIFTY100 remains the best holdout strategy (Sharpe +0.31).

### Remaining levers (require fresh walk-forward = retraining; user approval needed)
1. **5–10-day horizon labels** — lower turnover, better SNR (most promising, untried).
2. **New information sources** — macro/sector/fundamental/flow features (current 62 are all price/volume).
3. Accept negative result; keep model as research/screening tool only.

---

## CURRENT STATUS — Post Phase 6 (2026-08-28, late evening)

> Phase 6 executed lever #1 above. Authoritative doc: `PHASE6_RESULTS.md`, `reports/phase6_summary.md`.

### Phase 6 — 5-day horizon labels (pre-registered fresh walk-forward): **0/2 PASS**
- Fresh 51-block walk-forward, 205,221 OOS predictions, XGB with Phase-3 hyperparameters
- Model @5d: accuracy 51.93% vs 52.30% drift-inflated base rate (z overlap-adj = **−1.5**), AUC 0.5113
- Tercile ranking monotone (51.05% → 52.72% → 53.13% dir. acc.; +10.6 bps/5d gross top−bottom) — the only surviving signal
- Corrected portfolios (holdout 2025–26): F1 xgb≥0.5 w5 → −0.9%/+0.04 Sharpe (FAIL); F2 tercile w5 → +5.3%/+0.28 at 10 bps (ties naive) and +3.2%/+0.20 at 25 bps (loses) → **FAIL** per pre-registered rule
- F0 validation anchor: +264.6% net, Sharpe 0.95, turnover 0.15 ≈ Phase-5 D1 (+287.2%, 0.97) ✓

### Bug #4 fixed (compounding family)
Phase 6's first backtest compounded **overlapping 5-day forward returns daily** (each price move counted 5×)
→ F0 "net +73,400%". Fixed to overlap-safe accounting: held symbols earn their **1-day return on each held
day**; 5d label only drives membership. New assertions: |daily net| ≤ 0.15, turnover < 5000, plus the full
Phase-3 set. Running tally of audit bugs: 4.37e21 sequential compounding · BRITANNIA bad prices ·
pandas stack() NaN · **5-day overlap compounding**.

### Final program conclusion (Phases 3–6, four independent walk-forwards, 12 portfolio designs)
- ✅ Real, robust, monotone cross-sectional ranking signal (AUC 0.51–0.53, z up to +16.8, survives
  horizon/threshold changes) worth ~2 bps/day **gross**.
- ❌ Not monetizable at Indian retail costs (~20–30 bps round-trip): 0 of 12 pre-registered designs beat
  naive equal-weight on the 2025–26 holdout; concentration/2020-regime dependence kills the rest.
- **Best strategy remains naive daily (or 5-day frozen) equal-weight NIFTY100**: holdout +5.3–6.1% net,
  Sharpe +0.28–0.31, turnover ≈ 0.
- Recommended use of the model: research/screening signal only; any future design must be pre-registered
  and judged on a holdout period accruing after 2026-08-28.

---

## CURRENT STATUS — Post Phase 7 (2026-08-29, morning)

> Phase 7 executed lever #2 (new information: market-state & sector features). Authoritative doc:
> `PHASE7_RESULTS.md`, `reports/phase7_summary.md`.

### Phase 7 — market-state & sector features (pre-registered two-gate test): **GATE 1 FAIL**
- 13 new features (market regime, breadth, sector relative strength/momentum, 18 sectors, 0 unmapped)
- Paired evaluation n = 205,948 OOS predictions, 51 retrain blocks, identical walk-forward protocol
- Base (62 feats): AUC **0.5253**, acc 51.85%, z +13.1 · Extended (75 feats): AUC 0.5210, acc 51.75%, z +12.2
- ΔAUC = **−0.0043** · McNemar b=34,223 / c=34,422, z = −0.76 → no improvement; **Gate 2 never evaluated**

### Bug #5 fixed (label leakage)
New label columns (`next_ret_1d`/`target_1d`) were missing from the feature exclusion list → first run
showed AUC 1.0000. Caught by sanity alarm; fixed with a unified label denylist asserted against every
feature set, a 62-feature protocol guard, and an OOS-metrics alarm (acc > 0.60 ⇒ abort). Tainted cache
deleted; clean rerun performed (base arm reproduces Phase 3 exactly, validating the pairing harness).

### Program conclusion after 7 phases (5 walk-forwards, 20 designs, 0 holdout passes)
- All information available in the current dataset is now **exhausted**: 1-day technicals (Ph3),
  5-day horizon (Ph6), market-state/sector (Ph7) — none beats the 62-feature base.
- The validated residual signal (~51.85%, AUC 0.525, ~2 bps/day gross) remains **research-only**.
- Best strategy: **naive equal-weight NIFTY100** (holdout +5.3–6.1% net, Sharpe +0.28–0.31).
- Further gains require genuinely new external data (macro, flows, fundamentals, sentiment) — not more
  modeling on price/volume data. Audit bug tally: 4.37e21 compounding · bad prices · stack() NaN ·
  5-day overlap compounding · label leakage.

---

## Phase 8 (2026-08-29): External Macro Features — GATE 1 FAIL

- Tested 9 free external features (India VIX, S&P 500, USD/INR, Crude — all lagged ≥1 session)
- Extended model (71 feats): AUC 0.5200, acc 51.61% vs base 62 feats: AUC 0.5253, acc 51.85%
- dAUC = -0.0053, McNemar z = +1.76 (favors base) -> Gate 1 FAIL, Gate 2 never evaluated
- Deliverables: scripts/phase8_external_macro.py, reports/phase8_results.json, reports/phase8_summary.md, PHASE8_RESULTS.md
- **Program conclusion:** all free information sources exhausted (technicals, horizon, sector, macro). Validated residual signal ~2 bps/day gross is research-only. Naive daily equal-weight NIFTY100 remains the best strategy (2025-26 holdout +5.3-6.1% net, Sharpe +0.28-0.31). Further gains require new data classes (FII/DII flows, fundamentals, sentiment).

---

## Phase 9 (2026-08-30): FII/DII Institutional Flow Features — DATA-LIMITED PROBE, Gate 1 INCONCLUSIVE / Gate 2 FAIL

> **BLOCKER:** NSE/NSDL are unreachable from this host (403/503/connection-reset). Only 142
> FII/DII sessions (2026-01-14..2026-08-28) were reachable via the free NSE/NSDL-sourced
> aggregator. A validated multi-year walk-forward is therefore NOT possible here.

- Rewrote `fetch_fii_dii_flows` (NSDL → aggregator fallback, TLS/cookie handling, coverage warning); added `scripts/phase9_fii_dii.py` (faithful Phase-8 mirror, scoped to the FII/DII window).
- 30 FII/DII-derived features added to 62-feature base. Lookahead-safe (FII/DII net merged on session t, predicts t+1).
- PROBE parameters (deviation from protocol): warmup=40 (protocol 500), whole 2026 window as test (no 2025-26 holdout). n_paired = 11,014 OOS.
- Base AUC 0.5094 / acc 51.26% · Extended AUC **0.5316** / acc 52.19% · **dAUC = +0.0222** (largest raw lift of any external source: macro −0.0053, sector −0.0043).
- **McNemar b=2257, c=2359, z = −1.50 → NOT significant.** Gate 1 passes the raw-AUC rule but is statistically inconclusive.
- Gate 2 (2026 probe window, not a true holdout): G0 naive +0.111 net / 1.54 Sharpe; G1 +0.109 / 1.65 (costs cut to +0.025 / 0.46); G2 +0.109 / 1.49 → **both FAIL vs G0**.
- **Interpretation:** FII/DII is the first external class to show a meaningful raw signal, consistent with its role as the dominant driver of Indian market direction — but on reachable 142-day data the lift is not significant and not monetizable. Phase 9 is **not closed**.
- **To run the definitive Phase 9:** fetch full FII/DII history from an environment where NSE/NSDL are reachable (or drop `data/external/fii_dii_flows.csv` in place), then raise `START_DAYS` to 500 and remove the window restriction in `scripts/phase9_fii_dii.py`. Harness + features + anti-leakage assertions are ready.
- Deliverables: scripts/phase9_fii_dii.py, reports/phase9_results.json, reports/phase9_predictions_base.csv, reports/phase9_predictions_ext.csv, PHASE9_RESULTS.md

