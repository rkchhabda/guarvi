# Phase 9 — FII/DII Institutional Flow Features (DATA-LIMITED PROBE)

**Generated:** 2026-08-30
**Status:** Probe only. Definitive Phase 9 blocked: NSE/NSDL are unreachable from this
host (403/503/connection-reset); only 142 FII/DII sessions (2026-01-14..2026-08-28)
were reachable via the free NSE/NSDL-sourced aggregator.

## What changed
- `src/market_ml/external_data.py::fetch_fii_dii_flows` rewritten to try NSDL (authoritative,
  full history) then fall back to the reachable aggregator. Added `_tls_context`, `_open_url`,
  `_parse_nsdl_csv`, `_fetch_nsdl`, `_fetch_aggregator`. Emits an explicit coverage warning
  when history < 1 year (so a short series is never silently treated as a holdout).
- New `scripts/phase9_fii_dii.py`: faithful mirror of Phase 8 (paired base vs extended
  walk-forward, AUC + McNemar Gate 1, portfolio Gate 2) scoped to the FII/DII window.

## Pre-registration (as declared in the script header)
- 30 FII/DII-derived features added to the 62-feature base (net-flow MAs, z-scores, FII-DII
  spread, buy ratio, momentum, rolling correlation, consecutive buy/sell streaks).
- Lookahead rule: FII/DII net for session t is published post-close t → merged on date t, used
  only to predict target_1d (close t+1 > close t). No leakage (asserted).
- Gate 1: ext AUC > base AUC on paired OOS. Sanity alarm acc>0.60/AUC>0.65 aborts.
- Gate 2 (if Gate 1): G1/G2 daily designs beat G0 naive equal-weight on net AND Sharpe @10/25 bps.

## *** PROTOCOL DEVIATION (must be stated) ***
The validated Phase-3 protocol uses a **500-day warmup** and a **2025-26 holdout**. Neither is
possible here: only 142 FII/DII sessions exist. The probe therefore uses warmup=40 and treats
the whole 2026 window as the test period. AUC/direction numbers are **not comparable** to
Phase 3's 0.5253 (different warmup, different era). This is an indicative directional probe,
not a holdout pass.

## Results (n_paired = 11,014 OOS predictions, 16 retrain blocks, 2026 window)
| Arm | AUC | Acc | z vs base-rate |
|-----|-----|-----|----------------|
| Base (62 feats) | 0.5094 | 0.5126 | +0.7 |
| Extended (+FII/DII) | 0.5316 | 0.5219 | +2.6 |

- **ΔAUC = +0.0222** — the largest raw lift of any external source tested (macro −0.0053 in
  Phase 8, sector −0.0043 in Phase 7).
- **McNemar b=2257, c=2359, z = −1.50** → the lift is **NOT statistically significant**; the
  paired win/loss is essentially a coin-flip (slightly favors base on swaps).
- Sanity alarm: acc 0.5219 < 0.60, AUC 0.5316 < 0.65 → no leakage flag.

### Gate 1: PASS on the raw-AUC rule, but McNemar non-significant
By the literal AUC rule (ext > base) Gate 1 "passes", but the McNemar z=−1.50 means the
improvement is not distinguishable from noise on this short window. Treat Gate 1 as
**inconclusive**, not a real pass.

### Gate 2 (evaluated on the 2026 probe window, NOT a true holdout): FAIL
| Design | net | Sharpe | turnover | @25bps net | @25bps Sharpe |
|--------|-----|--------|----------|-----------|---------------|
| G0 naive daily EW | +0.111 | 1.54 | 0.0 | +0.111 | 1.54 |
| G1 xgb≥0.5 daily | +0.109 | 1.65 | 52.0 | +0.025 | 0.46 |
| G2 tercile tilt | +0.109 | 1.49 | 17.9 | +0.079 | 1.13 |

Neither G1 nor G2 beats G0 on net or Sharpe at 10/25 bps.

## Honest bottom line
- FII/DII flows are the **first external source to show a meaningful raw AUC lift** (+2.2 pts),
  consistent with their known economic role as the dominant driver of Indian market direction.
  But on the only reachable 142-day window the lift is **not statistically significant** and
  **not monetizable** vs naive equal-weight.
- This does **not** close Phase 9. A definitive test requires full FII/DII history (2021→2026)
  to run the validated 500-day-warmup walk-forward with a real 2025-26 holdout.
- **To run the definitive Phase 9:** execute `scripts/fetch_fii_dii.py` (or
  `fetch_fii_dii_flows`) from an environment where NSE/NSDL are reachable, OR drop a
  `data/external/fii_dii_flows.csv` (columns date, fii_net, dii_net, …) into place, then run
  `python -m scripts.phase9_fii_dii` (raise START_DAYS back to 500 and remove the 2026 window
  restriction). The harness, features, and anti-leakage assertions are already in place.

## Audit / process notes
- FII/DII is index-level (one value per session, broadcast to all symbols). Correctly merged on
  `date` with `how='left'` and the frame restricted to the coverage window so train and test
  both carry the feature — no train/test contamination from NaN-filled pre-history.
- Reuses the Phases 3-8 backtest assertions (exposure≤1, turnover, |1d ret|≤0.5, MDD∈[-1,0],
  finite equity).
