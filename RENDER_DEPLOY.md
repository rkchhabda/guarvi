# Deploying Guarvi Signal Portal to Render

This is the read-only FastAPI dashboard at `web/main.py` + `web/static/index.html`.
Everything you need is in this repo. No API keys, no database, no worker.

## 1. Prerequisites

- A GitHub account.
- This project pushed to a GitHub repository (public or private).
- A free [Render](https://render.com) account.

## 2. One-time: create the GitHub repo

If you don't already have one:

```bash
cd "C:/Users/r_chh/OneDrive/Apps/Guarvi1"
git init
git add render.yaml Procfile runtime.txt requirements-render.txt \
        render_build.sh .renderignore .gitignore \
        web/ data/processed/nifty100_ohlcv.parquet reports/live_signal.json \
        RENDER_DEPLOY.md README.md
git commit -m "Guarvi: Render deploy artifacts + read-only dashboard"
gh repo create guarvi-portal --public --source=. --push
```

If you already have a GitHub repo for this project, just `git add` the new
files and push.

> **Why these files only?** `.gitignore` excludes heavy training artifacts
> (raw CSVs, model `.joblib`, phase reports). The dashboard needs only the
> 5 MB OHLCV parquet and the 15 KB `live_signal.json` to render.

## 3. Deploy on Render

You have two options. **Option A (Blueprint) is one-click.**

### Option A: Blueprint (recommended)

1. Sign in to https://render.com.
2. Dashboard -> **New** -> **Blueprint**.
3. Connect the GitHub repo you just pushed.
4. Render reads `render.yaml` and shows the plan: 1 web service, free tier,
   Python 3.12, Oregon region.
5. Click **Apply**. Render builds and deploys in ~2-3 min.
6. Your URL is `https://guarvi-portal.onrender.com`.

### Option B: Manual web service

1. Dashboard -> **New** -> **Web Service**.
2. Connect the same GitHub repo.
3. Fill in:
   - **Name**: `guarvi-portal`
   - **Region**: `Oregon (US West)`
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `bash render_build.sh && pip install --upgrade pip && pip install -r requirements-render.txt`
   - **Start Command**: `uvicorn web.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
4. **Advanced** -> **Health Check Path**: `/api/now`
5. **Advanced** -> **Environment Variables**:
   - `PYTHON_VERSION` = `3.12.8`
   - `PYTHONUNBUFFERED` = `1`
   - `WEB_CONCURRENCY` = `2`
6. Click **Create Web Service**.

## 4. Verify the deploy

After Render says "Live", open the URL. The dashboard should show:

- The dark-navy theme with the **NIFTY 50 hero strip** (+0.63% badge).
- The 4-column **KPI grid** (Direction, Breadth, Auc, etc.).
- The **Top 5 / Bottom 5** tables with 5 rows each.
- The **Live Quotes** bar reading
  *"Market closed. Showing last available LTP as of 24-08-2026 15:30 IST ..."*.

If you see "signal not built" or empty tables, the parquet/JSON didn't
get included. Check the build log: it prints the file sizes — if either
is missing, ensure `data/processed/nifty100_ohlcv.parquet` and
`reports/live_signal.json` are committed.

You can also hit the JSON endpoints directly:

```
curl https://guarvi-portal.onrender.com/api/now
curl https://guarvi-portal.onrender.com/api/signal
curl https://guarvi-portal.onrender.com/api/quotes
curl https://guarvi-portal.onrender.com/api/nifty50
curl https://guarvi-portal.onrender.com/api/symbol/INFY
```

## 5. Free-tier caveats

- **Cold start**: After 15 min of no traffic, the service sleeps. The
  first request after that takes ~30-50 s. Subsequent requests are fast.
- **Disk is ephemeral**: Anything written to disk is lost on restart.
  The dashboard only **reads** from disk, so this is fine. If you ever
  add a live collector to Render, mount a persistent disk (paid).
- **Outbound HTTP from the live collector won't work on free tier**.
  Render free web services can't reach most external APIs reliably, and
  Groww requires a whitelisted static IP. The offline-mode (parquet
  fallback) is the primary mode on Render.

## 6. Updating the data

When `reports/live_signal.json` or `data/processed/nifty100_ohlcv.parquet`
change locally:

```bash
git add reports/live_signal.json data/processed/nifty100_ohlcv.parquet
git commit -m "Refresh signal + OHLCV"
git push
```

Render auto-deploys (Blueprint option) or you click **Manual Deploy** ->
**Deploy latest commit**.

## 7. Files this deploy uses

| Path | Purpose | Size |
|------|---------|------|
| `web/main.py` | FastAPI app (9 endpoints) | ~12 KB |
| `web/static/index.html` | Dashboard frontend | ~39 KB |
| `data/processed/nifty100_ohlcv.parquet` | OHLCV for quotes + deep-dive | ~5 MB |
| `reports/live_signal.json` | Top-5/Bottom-5 + KPI metrics | ~15 KB |
| `render.yaml` | Render Blueprint | ~1 KB |
| `Procfile` | Process type | <100 B |
| `runtime.txt` | Python version pin | ~15 B |
| `requirements-render.txt` | Minimal runtime deps | ~500 B |
| `render_build.sh` | Build-time sanity checks | ~1 KB |
| `.renderignore` | Build context exclusions | ~3 KB |

That's it. The whole service is ~5.5 MB of source data plus ~1 MB of
code, well under Render's free-tier slug limit.
