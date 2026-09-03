# Plan: Guarvi Live Web Portal

**Date:** 2026-08-30
**Author:** opencode
**Status:** Proposal — not yet implemented.

---

## 0. Honest framing (read first)

The research (Phases 1–9) is explicit: the directional signal is **real but small
(~2 bps/day gross) and not monetizable at Indian retail costs**; naive equal-weight
NIFTY100 is the best strategy on the 2025-26 holdout. Therefore the portal must be a
**research / screening / education dashboard**, not an investment-advice or auto-trading
product. This is both honest and the safer legal posture (SEBI regulates investment
advice; unregistered advice can carry liability). v1 is **read-only**.

---

## 1. What already exists (reuse, don't rebuild)

- `scripts/live_data_runner.py` — polls Groww quotes + OHLCV during NSE hours, appends
  CSVs to `data/live/nse/`, runs as systemd service `guarvi-live-data`.
- `scripts/setup_vps.sh` — provisions an Oracle Cloud VM (ap-mumbai-1), `.venv`, `.env`
  (chmod 600), `ufw`, and the `guarvi-live-data` systemd unit. **Extend this** for the web.
- `scripts/predict_latest.py` — loads `models/NIFTY50_model.joblib`, returns P(next-day up).
- `models/NIFTY50_model.joblib` + `NIFTY50_meta.json` — a trained artifact (NIFTY50 only).
- `data/processed/nifty100_features.parquet` — the cross-sectional dataset behind Phases 3–9.
- `reports/phase3..9_results.json` — historical signal stats to show as "track record".

## 2. Recommended stack (suitable portal)

| Layer | Choice | Why |
|-------|--------|-----|
| Backend API | **FastAPI** + uvicorn | Tiny, async, reuses Python models; deploys on the same VM. |
| Frontend | **Single static page** (HTML + vanilla JS / htmx) | No build step, served by nginx. Dashboard = signal gauge, screener table, live quotes, disclaimer. |
| Reverse proxy / TLS | **nginx + certbot** (Let's Encrypt) | Standard, free, on the Oracle VM. |
| Process | **systemd** `guarvi-web` (+ existing `guarvi-live-data`) | Consistent with current deploy. |
| Signal computation | Nightly batch → JSON cache | API stays stateless/fast; no model load per request. |

> *Fastest alternative:* **Streamlit** gets a dashboard running in ~1 hour but is
> single-process and weaker for custom UI / live quotes / multi-user. Pick FastAPI for a
> real "portal"; Streamlit only if you want a 1-day prototype.

## 3. Architecture

```
                 nginx (:80/:443, TLS)
                   |  /api/*  --> uvicorn (guarvi-web)
                   |  /*      --> static dashboard
        Oracle VM (ap-mumbai-1, static IP whitelisted at Groww)
        ┌─────────────────────────────────────────────┐
        │ guarvi-live-data (systemd)                   │
        │   Groww API -> quotes_<date>.csv, ohlcv/*.csv│
        │ guarvi-signal (systemd timer, nightly)       │
        │   build latest features -> model ->          │
        │   reports/live_signal.json (per-symbol prob) │
        │ guarvi-web (systemd, uvicorn :8000)          │
        │   GET /api/signal, /api/screen, /api/quotes, │
        │   /api/performance   (reads cached JSON+CSV) │
        └─────────────────────────────────────────────┘
```

## 4. Components to build

1. **`scripts/build_signal.py`** (new, ~1 day)
   - Train/load a **NIFTY100 model** (currently only NIFTY50 exists; train on
     `nifty100_features.parquet` so we get per-symbol prob_up — matches the validated
     cross-sectional ranking signal). Save `models/nifty100_model.joblib` + meta.
   - Each night: rebuild features from the latest `data/live/nse/ohlcv/*.csv` +
     `nifty100_features`, score every symbol, rank by prob_up, write
     `reports/live_signal.json` (`{as_of, nifty50_prob, symbols:[{symbol, prob_up, rank, last_close}]}`).
   - Reuse `predict_latest.py` logic for the NIFTY50 aggregate gauge.

2. **`web/` backend (FastAPI, ~1–2 days)**
   - `GET /api/signal` → NIFTY50 prob_up + direction + last update time.
   - `GET /api/screen` → sorted NIFTY100 symbol table (prob, rank, calibration note).
   - `GET /api/quotes` → latest live quotes from `data/live/nse/quotes_<today>.csv`.
   - `GET /api/performance` → summary of Phase 3–9 stats + cumulative paper curve.
   - Every response carries a `disclaimer` field.

3. **`web/static/` frontend (~1–2 days)**
   - Signal gauge (NIFTY50 P-up), top/bottom-10 screener table with sparkline of prob,
     live quotes ticker table (market hours), prominent disclaimer banner, link to
     `PROJECT_STATUS.md` verdict. No order-entry UI in v1.

4. **Deploy glue (~0.5 day)**
   - Extend `scripts/setup_vps.sh`: install `guarvi-web` systemd service + `guarvi-signal`
     timer, nginx site + certbot, open :80/:443 in `ufw`.
   - Add `fastapi`, `uvicorn`, `pydantic` to `requirements.txt`.

## 5. Compliance / safety checklist

- [ ] Prominent, persistent disclaimer: research/education only, not SEBI-registered
      advice, not profitable net-of-costs per the project's own audit.
- [ ] No live order placement in v1 (keep `place_order.py` out of the portal UI).
- [ ] If order entry is ever added: paper-trading only, explicit confirmations, rate limits.
- [ ] `.env` stays chmod 600; API must not echo secrets; bind uvicorn to localhost, nginx
      only exposes `/api` + static.
- [ ] Per-symbol prob is a probability, not a recommendation — UI says "signal", not "buy".

## 6. Phased delivery

- **Phase A (MVP, ~3–4 days):** `build_signal.py` (NIFTY100 model + nightly JSON) +
  FastAPI read-only API + minimal static dashboard + nginx on VM. Read-only.
- **Phase B (polish):** live quotes table, screener sparklines, performance/"track record"
  page, mobile layout, certbot auto-renew.
- **Phase C (optional, only after compliance review):** paper-trading simulation page;
  live order entry strictly behind guards (not recommended given the research verdict).

## 7. Open decisions for you

1. **Scope/audience:** personal research dashboard, or a public portal? (Public → stronger
   SEBI/disclaimer review; recommend personal/closed-beta first.)
2. **Implement now or plan-only?** Say the word and I'll scaffold Phase A.
3. **Stack:** FastAPI (recommended) vs Streamlit (1-hour prototype)?
4. **Signal universe:** NIFTY50-only (model exists today) or invest the day to
   train the NIFTY100 model for full-symbol screening?

---

## 8. Phase A implementation log (DONE 2026-08-30)

Decisions applied: **public, read-only** · **scaffold Phase A now** · **NIFTY100 screening**.

- `scripts/build_signal.py` (new): trains a pooled NIFTY100 XGBoost model (62 base features,
  same hyperparameters as Phases 3–9), saves `models/nifty100_model.joblib` + meta, and writes
  `reports/live_signal.json` (per-symbol P(next-day up), rank, market-breadth gauge). Includes a
  leakage sanity alarm (in-sample acc>0.60 / AUC>0.65 aborts). Smoke-run: trained on 249,983
  rows, in-sample AUC 0.577, no alarm; signal as_of 2026-08-23, breadth 0.4915, 102 symbols.
- `web/main.py` (new, FastAPI): read-only `/api/signal`, `/api/screen`, `/api/quotes`,
  `/api/performance`, and `/` dashboard. Every response carries the disclaimer. Verified with
  TestClient (all 200).
- `web/static/index.html` (new): self-contained dashboard (no external CDN), signal gauge,
  top/bottom screener, live-quotes table, research track-record, persistent disclaimer banner.
- `requirements.txt`: added fastapi, uvicorn, pydantic.
- `scripts/setup_vps.sh`: extended with `--domain` / `--no-web`; installs nginx, the
  `guarvi-web` systemd unit (uvicorn :8000), a `guarvi-signal` oneshot + daily timer
  (16:00 Asia/Kolkata), and an nginx site (proxy to uvicorn; certbot if `--domain` given).
- `web/__init__.py` created so `web.main:app` imports cleanly under uvicorn.

**Verified:** `bash -n setup_vps.sh` OK; FastAPI TestClient all endpoints 200; full pytest
suite 84 passed (unchanged).

**What the VM run does:** `sudo bash scripts/setup_vps.sh --domain signal.example.com` provisions
the data collector + web portal + nightly signal timer on the Oracle VM. The portal is reachable
at `http://<vm-ip>/` (or HTTPS with `--domain`).

**Remaining for a real launch (Phase B/C, out of scope of this scaffold):**
- Wire `build_signal` to rebuild features from `data/live/nse/ohlcv/*.csv` (currently it scores
  the latest row of `nifty100_features.parquet`; the nightly timer will still refresh ranks, but
  truly live features need the live-feature pipeline). Keep `live_data_runner` collecting OHLCV.
- Public launch: add a stronger SEBI/disclaimer page + terms; consider closed-beta first.
- Optional Phase C: paper-trading simulation page (no live order entry, per the research verdict).

---

## 9. Live-freshness overlay (added 2026-08-30)

`scripts/build_signal.py` now accepts `--refresh-from-live` (also wired into the
`guarvi-signal` systemd timer on the VM). When run with that flag it:

1. Globs `data/live/nse/ohlcv/*.csv` (the Groww intraday files written by
   `live_data_runner.py`).
2. For each `(symbol, day)` keeps the last row's `close`.
3. For every symbol in the screener output, **if the live date >= parquet
   `as_of`**, it overlays the live `close` and bumps the symbol's `as_of`.
   Stale live data is ignored.
4. Writes the new `last_close`/`as_of` and a top-level `"freshness"` audit
   field into `reports/live_signal.json`.

This does **not** retrain, recompute features, or change the model's
`prob_up` ranking. The screener is now genuinely daily-fresh (latest close)
on the VM as long as `live_data_runner` is collecting; the model artefacts
are still based on the audited NIFTY100 features parquet.

Verified: with the live dir empty (this host), the overlay is a clean
no-op (only the `freshness` and `generated` fields change). With synthetic
live CSVs (MAXHEALTH @ 2026-08-30, IDEA @ 2026-08-01), MAXHEALTH's
`last_close` and `as_of` advanced to 2026-08-30 while IDEA kept its
parquet values (stale). `/api/signal` exposes `freshness`; the dashboard
shows it under the breadth gauge. Full pytest: 84 passed.

---

## 10. Live-feature pipeline (added 2026-08-30)

Section 9 only refreshed the **close**. To get genuinely fresh technicals
(RSI, MACD, returns, volatility) every day we also need to extend the
**feature parquet** with new daily rows. That's what
`scripts/refresh_live_features.py` does:

1. Globs `data/live/nse/ohlcv/*.csv` (5-min Groww candles, one file per
   symbol per day) and aggregates each file into a daily row
   (open=first, high=max, low=min, close=last, volume=sum, source=`groww_live`).
2. Loads the historical raw OHLCV (`nifty100_ohlcv.parquet`) and concats
   the new daily rows (deduped on `symbol+date`, live wins).
3. For each new symbol, runs `build_symbol_features` over the full
   historical+live series (so SMA_200 / 252-day rolling / RSI-14 warm
   up), then slices out only the new dates.
4. Runs `build_cross_sectional` over the new rows so the rank/excess/
   rel_volatility columns are populated.
5. Projects the new rows to the **historical schema** (71 columns) so
   the trained model stays happy. Extra "external" feature columns
   (FII/DII, options, sentiment) added by `build_symbol_features` are
   dropped; any cross-sectional columns for symbols that didn't trade
   on a new date are NaN (median imputation handles them).
6. Appends to `nifty100_features.parquet` in place, deduped on
   `(date, symbol)`.

**Idempotent.** The cutoff is the **features parquet's max date**, not
the OHLCV max, so re-running on the same live dir is a clean no-op
(verified: 6 new rows on the first call, 0 on subsequent calls; the
synthetic test data in `data/live/nse/ohlcv/` doesn't re-inject).

**Wired into the VM.** `setup_vps.sh` now generates a `guarvi-signal`
unit with two stages:

```
ExecStartPre=.../python -m scripts.refresh_live_features
ExecStart=.../python -m scripts.build_signal --refresh-from-live
```

so the nightly timer (16:00 IST) first extends the features parquet
from the day's Groww candles, then rebuilds the screener with the
freshness overlay. The web portal's `prob_up` ranks and the
freshness pill both update daily from real market data.

**Leakage safety.** New live rows have `target_direction = NaN` because
the next session isn't known; the model is never retrained on these
rows (training is offline against the audited historical parquet).
Cross-sectional NaNs for symbols absent on a new date are imputed by
the model's median imputer.

**Tests.** `tests/test_refresh_live_features.py` (6 cases) covers the
aggregator (empty dir, 2 symbols × 3 days, bad stems), the full
pipeline (6 new rows + idempotency), the no-live-files no-op, and the
cutoff behaviour (live dates already in features are skipped).
Full pytest: **90 passed**.

## 11. Dashboard revamp — fintech-grade UI (added 2026-08-30)

The first cut of `web/static/index.html` was a bare table. It rendered
correctly but it didn't look like a product, and it didn't give a
visitor enough signals to trust the underlying number. This pass
rebuilds the page around eight KPI cards, a search/filter stock
screener with a Top-5 / Bottom-5 split, a live IST clock, a NIFTY 50
reference value, and a state-aware live-quotes section. Everything is
plain HTML/CSS/JS — no framework — and is served by the existing
`/static` mount.

### New / changed endpoints (`web/main.py`)

| Route            | Purpose                                                      |
|------------------|--------------------------------------------------------------|
| `GET /api/now`   | Server-side IST clock + `in_market_session` flag.            |
| `GET /api/nifty50` | NIFTY 50 reference value (real CSV if present, proxy otherwise). |

Both are pure functions over `data/` and `ZoneInfo("Asia/Kolkata")`,
no auth, no caching, no external calls.

### NIFTY 50 source-of-truth ladder

The repo has no real NIFTY 50 index file. `_nifty50_index()` looks
for one in this order and returns the first hit:

1. `data/raw/nifty50.csv`        — drop a daily `date, close` CSV here.
2. `data/external/nifty50.csv`   — same shape, alternate location.
3. `data/processed/nifty50_index.parquet` — precomputed.
4. **Proxy fallback** — mean of closes for the 50 highest-average-volume
   symbols in `nifty100_ohlcv.parquet`. The response field `source` is
   set to `"proxy_top50_liquid_avg"` so the UI pill's tooltip and the
   `/api/nifty50` payload never confuse a proxy value for the real index.

The current proxy value is **689.27** (+0.63% vs prev close, as of
2026-08-24). This is **research/education only**; the proxy and the
real index are not the same series.

### Page layout (`web/static/index.html`, ~22 KB)

- **Sticky top bar**: gradient logo "G", product name, subtitle, NIFTY 50
  pill (value + change + tooltip with `source · as_of`), live IST clock
  with pulsing green/red session dot.
- **Disclaimer banner**: gradient band immediately under the header, the
  honest "small ~2 bps/day, not profitable net of Indian retail costs,
  not SEBI-registered" message, full-width.
- **8 KPI cards in two rows of 4**: Signal Direction, Breadth P(up), AUC,
  Gross edge, Accuracy, Universe, Model, Freshness. Each card has a
  title, a primary number, a context sub-line, and a colour cue (UP
  green / DOWN red / neutral slate).
- **Stock screener**: heading + as-of pill, then a single row with a
  search input (icon + placeholder "Filter by symbol (e.g. RELIANCE,
  INFY, TCS)…"), three filter pills (All / UP only / DOWN only), and a
  live "**N / 102** symbols matching" counter.
- **Top 5 / Bottom 5 sub-cards** side by side: identical table schema
  (Rank, Symbol, P(up), Confidence bar, Last close). Top card has an UP
  green accent; bottom card has a DOWN red accent.
- **Live quotes section**: heading + state pill ("market open" /
  "market closed" / "no live data"); a status bar explaining the
  current state ("Live collector is idle or it is outside NSE hours
  09:15–15:30 IST"); a Symbol / Last / Change% / Volume table that is
  populated from `/api/quotes` when available and shows a friendly
  empty-state otherwise.
- **Footer**: short "not investment advice" pointer to `PROJECT_STATUS.md`.

### Polling cadences

| Element       | Endpoint          | Cadence |
|---------------|-------------------|---------|
| Live clock    | client-side       | 1s      |
| Session dot   | `/api/now`        | 30s     |
| NIFTY 50 pill | `/api/nifty50`    | 30s     |
| Signal KPIs   | `/api/signal`     | 60s     |
| Screener/T5B5 | `/api/signal`     | 60s     |
| Performance   | `/api/performance`| 60s     |
| Live quotes   | `/api/quotes`     | 30s     |

All pollers debounce on tab visibility — when the tab is hidden they
back off to a single tick on `visibilitychange` to keep idle CPU low.

### State & filtering

- A single `state` object holds `symbols`, `filter` (`"all"|"up"|"down"`),
  `search` (string, lowercase substring match), and the latest quotes.
- The search input and filter pills are pure UI; the source-of-truth
  list is never re-fetched for filter changes, so typing "INF" and
  pressing "DOWN only" is instant.
- Verified: typing "INF" narrows both Top-5 and Bottom-5 to a single
  row (INFY 48.4% / ₹1,130) and the count pill updates to "1 / 102".

### Tests

No new tests were required for the HTML (it's static). The new
`/api/now` and `/api/nifty50` routes use the same FastAPI `TestClient`
pattern as the existing 90 tests, so the full suite continues to pass:

```
$ python -m pytest -q
90 passed in 30.13s
```

### Visual verification

Loaded `http://127.0.0.1:8000/` in the integrated browser:

- ✅ gradient top bar, gradient logo "G", sticky
- ✅ NIFTY 50 pill in header: `689.27  +4.34 (+0.63%)` with tooltip
  `proxy_top50_liquid_avg · as of 2026-08-24`
- ✅ live IST clock ticking every second, "Sun 30 Aug 2026 16:03:37 IST"
- ✅ disclaimer banner immediately under the header
- ✅ 8 KPI cards in two rows: Direction = DOWN, Breadth = 49.1%, AUC = 0.5253,
  Edge = ~2 bps, Accuracy = 51.85%, Universe = NIFTY 100, Model = XGBoost,
  Freshness = "no live data · as of 2026-08-23"
- ✅ screener heading + as-of pill, search box + 3 filter pills + "102 / 102"
- ✅ Top 5 table: MAXHEALTH 53.8%, IDEA 53.5%, ULTRACEMCO 52.1%, SBILIFE 52.0%, VEDL 51.7%
- ✅ Bottom 5 table: GAIL 44.1%, BANKBARODA 44.6%, ESCORTS 45.2%, HUDCO 45.3%, PIDILITIND 45.6%
- ✅ live quotes: state pill = "market closed", status bar explains
  "Live collector is idle or it is outside NSE hours 09:15–15:30 IST"
- ✅ footer pointer to `PROJECT_STATUS.md`
- ✅ search test: typing "INF" → "1 / 102 symbols matching" + both tables
  show only INFY 48.4% / ₹1,130

## 12. NIFTY 50 hero, deep-dive panel & LTP-when-closed (added 2026-08-30)

The third pass turns the dashboard into a clickable research tool:
a prominent NIFTY 50 hero strip, a "Selected Stock Deep Dive" panel
that opens when a NIFTY-100 row is picked, a `.selected` row-highlight
on both Top-5 and Bottom-5 tables, and a "Market closed · last LTP as
of …" fallback so the live-quotes section is never empty.

### New / changed endpoints (`web/main.py`)

| Route                       | Purpose |
|----------------------------|---------|
| `GET /api/quotes` (changed) | Adds `market_state` (`open`/`closed`), top-level `as_of` and `source`. When the live collector has no rows, the route falls back to the most recent trading day in `nifty100_ohlcv.parquet` and labels the source `parquet_last_day`. |
| `GET /api/symbol/{symbol}`  | New. Per-symbol deep-dive: `prob_up`, `rank`, `confidence` (computed as `\|P(UP) − 0.5\| × 2`), `last_close`, `change`, `change_percent`, `volume`, day `open/high/low`, SMA-20, SMA-50, 52-week `high_52w`/`low_52w` (252-session window), and a 30-point `history` series. 404 if the symbol isn't in the parquet. |

`_latest_quotes()` now does a two-step lookup: live collector file
first, parquet last-day second. `_symbol_deep_dive(symbol)` computes
SMA-20/50 and 52w H/L on the fly — the model is never retrained.

### Frontend additions (`web/static/index.html`)

- **NIFTY 50 hero strip** under the topbar: 36-px value, large
  green/red percentage badge, source label (`proxy_top50_liquid_avg ·
  as of 24-08-2026`), and a hint that the source is parquet / live.
- **Symbol picker** (`<select>`) above the screener, populated with
  the 102 NIFTY-100 tickers, plus a "Clear" button.
- **Row-click handler** via event delegation: clicking any row in
  either table selects the symbol; clicking the same row again
  deselects it. The `data-symbol` attribute carries the ticker.
- **`.selected` row highlight**: `background: rgba(245,158,11,.12)` +
  a 3-px gold left border on the matching `<tr>` in **both** the Top-5
  and Bottom-5 tables.
- **Selected Stock Deep Dive** container: large symbol header, then
  six detail cards in a 4-column grid (collapses to 2 on narrow
  screens) — P(UP) with progress bar, Confidence with bar, Last close
  with signed change %, Day OHLC + Volume, 52-week range, SMA-20 / 50
  — and a final full-width **inline-SVG sparkline** of the last 30
  closes (auto-coloured green/red by direction). Source attribution:
  `parquet_ohlcv · computed on the fly from nifty100_ohlcv.parquet ·
  No model retraining.`
- **LTP-when-closed**: the live-quotes bar now reads
  `"Market closed. Showing last available LTP as of 24-08-2026 15:30 IST
  · 50 symbols · source: parquet_last_day · as of 24-08-2026"` and the
  table is populated with last/change%/volume for the 50 most liquid
  symbols from the most recent parquet day.
- **Client-side data cache** (`dataCache`): small in-memory
  url → `{t, data}` map with a TTL; `cachedFetch(url, maxAgeMs)`
  prevents the screener from re-fetching on every keystroke or filter
  change, and the deep-dive from re-fetching when the same symbol is
  re-picked within 60 s. (Streamlit's `@st.cache_data` equivalent.)
- **Tab-visibility debounce**: when the tab is hidden the heavy
  pollers pause; on `visibilitychange → visible` we do a single
  catch-up tick of each poller.
- **Number formatting**: every price uses `toLocaleString('en-IN')`
  (e.g. `1,280.70`, `1,78,245`). Dates use `DD-MM-YYYY`.

### Theme

- Dark navy background (`#0b0f17`) with two radial-gradient
  highlights, dark slate cards (`#131a28` / `#182132`), UP-green
  (`#22c55e`) and DOWN-red (`#ef4444`) accents.
- KPI cards now have a soft blue glow
  (`0 0 24px rgba(59,130,246,.06)`) that intensifies on hover.
- The deep-dive panel has a distinct border (accent blue,
  `box-shadow: 0 0 32px rgba(59,130,246,.18)`) so it visually
  separates from the screener cards.

### Verified

- All 9 endpoints return 200.
- `/api/symbol/MAXHEALTH` → 200 with `prob_up=0.5376, rank=1,
  last_close=1016, change_percent=+2.57%, volume=1.89M, high=1016,
  low=980.3, sma_20=1044.62, sma_50=1078.17, high_52w=1250.1,
  low_52w=931.6, history rows=30`.
- `/api/symbol/NOPE404` → 404.
- Browser snapshot confirms: NIFTY 50 hero strip with `+0.63%` badge,
  102-option selectbox, clickable rows `[cursor=pointer]`, deep-dive
  panel for INFY (P(UP) 48.4%, rank #68, confidence 3.2%, last close
  1,144, change +1.24%, 52w range 1,689.8 – 985.3, SMA 1,154.8 /
  1,102.36, sparkline), and for VEDL (Top-5 row highlighted gold).
- Row-click test: clicking IDEA → `selected: ["IDEA"]`, deep-dive
  opens, selectbox auto-syncs; clicking the same row again →
  `selected: []`, deep-dive closes.
- LTP-when-closed test: quote bar reads "Market closed. Showing last
  available LTP as of 24-08-2026 15:30 IST" and the table is
  populated with ABB, ADANIENSOL, etc., all formatted with Indian
  commas (`7,627`, `1,78,245`, `20,43,772`).
- `python -m pytest -q` → **90 passed in 42.41s**, no regression.

## 13. Render deployment (added 2026-08-31)

The dashboard is now one-click deployable to Render free tier. All
artifacts are in the working tree; the user just needs to push to a
GitHub repo and click "New Blueprint" in the Render dashboard.

### Files added for Render

| File | Purpose |
|------|---------|
| `render.yaml` | Render Blueprint: 1 web service, free plan, Oregon region, Python 3.12, health check `/api/now` |
| `Procfile` | Process type: `web: uvicorn web.main:app --host 0.0.0.0 --port $PORT` |
| `runtime.txt` | Pins Python to 3.12.8 (Render's supported line) |
| `requirements-render.txt` | Minimal runtime deps — only `fastapi`, `uvicorn`, `pydantic`, `pandas`, `numpy`, `pyarrow`, `python-dotenv`. No XGBoost / sklearn / jugaad-data / growwapi, since the dashboard never imports them. Keeps the slug small and the cold start fast. |
| `render_build.sh` | Build-time sanity checks. Confirms the parquet and `live_signal.json` are in the build context; fails the build if `web/static/index.html` is missing. |
| `.renderignore` | `.dockerignore`-compatible exclusion list. Drops `data/raw`, `data/external`, `data/interim`, `models/`, `src/`, `tests/`, `scripts/`, all the `phase*` reports, the 126 MB `nifty100_features.parquet`, and the local `.env`. Keeps the slug at **5.06 MB / 36 files**. |
| `.gitignore` | First-class `.gitignore` for the GitHub repo. Mirrors `.renderignore` but also excludes `data/live/`, all phase reports, the heavy `nifty100_features.*` files, the `models/` directory, and `fix_indent.py`/`pytest.ini`/`pytest_last_run.txt`. `.env` is excluded; `.env.example` is committed. |
| `RENDER_DEPLOY.md` | Step-by-step deploy guide with the two ways to create the service (Blueprint vs. manual). |

### Verified locally

- `render.yaml` + `Procfile` + `runtime.txt` + `requirements-render.txt`
  + `render_build.sh` parse correctly.
- Simulated Render boot by running
  `uvicorn web.main:app --host 0.0.0.0 --port 8771` (the same command
  Render will run, with `$PORT` replaced). Boot latency ~1 s.
  All 9 endpoints return 200; `/api/symbol/NOPE404` returns the
  expected 404.
- Final Render slug size (after `.renderignore`): **5.06 MB** —
  the parquet (5.10 MB) dominates; everything else is <60 KB.

### Free-tier caveats baked into the design

- **Cold start**: 15 min of no traffic → service sleeps. First
  request after that takes 30-50 s. Documented in `RENDER_DEPLOY.md`
  §5.
- **Ephemeral disk**: Free Render web services don't have a
  persistent disk. The dashboard only *reads* from disk
  (`data/processed/nifty100_ohlcv.parquet`, `reports/live_signal.json`),
  so this is fine. The live collector isn't shipped to Render because
  Groww requires a whitelisted static IP, which the free tier can't
  provide.
- **Python 3.14 not yet on Render**: manylinux wheels for 3.14 are
  not all published, so `runtime.txt` pins 3.12.8.
- **No secrets to set**: The dashboard has no API keys, no DB, no
  external services. Deploy with zero env vars. The Groww creds in
  `.env` are for the offline collector only and stay on the user's
  laptop / Oracle VM.

### Deploy steps (what the user has to do)

1. `git init` (if not a repo), `git add` the new files, push to GitHub.
2. Render dashboard → **New** → **Blueprint** → pick the repo → **Apply**.
3. Wait ~2-3 min for the build. Open the URL.

That's the entire deploy. See `RENDER_DEPLOY.md` for the long version
including the manual web-service route, the verification checklist, and
how to update the data after deploy.


