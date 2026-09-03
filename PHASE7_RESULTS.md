# Phase 7 — Market-State & Sector Features: **GATE 1 FAIL (stopped by pre-registration)**

**Date:** 2026-08-29 · **Script:** `scripts/phase7_market_sector.py` · **Runtime:** 3,089s (51 retrain blocks)
**Data:** 249,983 rows, 2,633 dates (2015-12-31 → 2026-08-17), 98 symbols · **OOS predictions:** 205,948 (paired)

## Pre-registered protocol (declared before evaluation)

Two-gate test, same discipline as Phases 5–6:

- **Gate 1 (this phase):** extended model (62 base + 13 market-state/sector features = 75) must beat
  the base model's **AUC on paired OOS predictions** (identical dates/symbols, identical walk-forward:
  retrain every 42 days, 500-day warmup, 200K train cap, XGB hyperparameters identical to Phase 3).
- **Gate 2 (conditional):** only if Gate 1 passes, evaluate portfolio designs vs naive reference on the
  2025–26 holdout (net AND Sharpe at 10 and 25 bps). **Gate 1 failed ⇒ Gate 2 never evaluated.**

New features (13): market regime (NIFTY100 equal-weight index trend/vol/volume state), market breadth
(% symbols above own 20d/50d MA), sector relative strength and sector-momentum (18 sectors mapped from
symbol prefixes; 0 rows unmapped).

## ⚠️ Incident: label leakage caught and fixed (Bug #5 family — data/label hygiene)

The **first run** produced AUC = 1.0000, accuracy = 100.00% — mechanically impossible. Root cause:
`load_data()` creates label columns `next_ret_1d` / `target_1d`, but the feature exclusion list (copied
from Phase 3, which used different label names) did not contain them → the model trained on the label.

**Defenses added (institutionalized):**
1. Unified label-column denylist (`next_ret_1d`, `target_1d`, `next_ret`, `target_direction`,
   `next_ret_5d`, `target_5d`) asserted absent from every feature set.
2. Protocol guard: base feature count must equal 62 (Phase-3 protocol).
3. Sanity alarm: OOS accuracy > 0.60 or AUC > 0.65 ⇒ abort before any downstream analysis.
4. Tainted prediction cache deleted; full clean rerun performed.

Base-arm reproduction on both runs (AUC 0.5253, acc 0.5185) confirms the pairing harness is exact.

## Clean results (leakage-free, paired n = 205,948, 51 blocks)

| Model | Features | AUC (MW) | Accuracy | z vs 0.5 base rate |
|---|---|---|---|---|
| Base (Phase-3 protocol) | 62 | **0.5253** | **51.85%** | +13.1 |
| Extended (+ market/sector) | 75 | 0.5210 | 51.75% | +12.2 |

- **ΔAUC = −0.0043** (extended is *worse*)
- **McNemar:** b = 34,223 (base right/ext wrong), c = 34,422 (ext right/base wrong), **z = −0.76**
  → no statistically significant difference in paired errors; the direction favors base.
- **GATE 1: FAIL** → per pre-registration, portfolio stage skipped (no p-hacking path).

## Interpretation

1. Market-state and sector one-hot/relative-strength features add **no incremental discriminating
   information** beyond the existing 62 price/volume features on NIFTY100 daily data.
2. The slight degradation (−0.004 AUC) is consistent with the variance cost of +13 features with no
   compensating signal — 62 features is already at/ past the information ceiling of this dataset.
3. Combined with Phase 6 (5-day horizon: z = −1.5 vs drift base rate; ranking-only ~2 bps/day gross)
   and Phases 4–5 (12 portfolio designs, 0 holdout passes), **all information sources available in the
   current dataset have now been tested and exhausted**.

## Bug ledger (audit trail of the program)

| # | Bug | Symptom | Caught by |
|---|---|---|---|
| 1 | Compounding across 128K symbol-dates as sequential trades | 4.37e21 cumulative return | Phase 9A audit |
| 2 | BRITANNIA bad prices (fake +14,593%/period) | equity 4.1e31 | Phase 3 \|ret|≤0.5 filter |
| 3 | Overlapping 5-day returns compounded daily | naive showed +73,400%, MDD −0.88 | Phase 6 equity sanity assertions |
| 4 | Label columns absent from exclusion list | extended AUC 1.0000 | Phase 7 sanity alarm |

## Conclusion and program status

- The **only** validated signal remains the 62-feature XGB ranking edge (~51.85% / AUC 0.525,
  ~2 bps/day **gross**) — real (z ≈ +13–17) but **not monetizable** at Indian retail costs.
- Best deployable strategy is still **naive daily equal-weight NIFTY100** (holdout +5.3–6.1% net,
  Sharpe +0.28–0.31, near-zero turnover).
- Any further improvement requires **new information not present in the current dataset**
  (macro releases, FII/DII flows, fundamentals, news/sentiment) — not more modeling on the same data.
