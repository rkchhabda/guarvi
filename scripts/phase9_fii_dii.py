#!/usr/bin/env python3
"""Phase 9: FII/DII institutional flow features - pre-registered two-gate test (DATA-LIMITED PROBE).

PRE-REGISTRATION (declared before evaluation):
  New information class: institutional equity flows (FII = foreign, DII = domestic).
  Source: NSE/NSDL daily FII/DII cash turnover (via the reachable aggregator, since
  NSE/NSDL block automated access from this host). 30 derived features from
  compute_fii_dii_features(): net-flow MAs, z-scores, FII-DII spread, buy ratio,
  momentum, rolling correlation, consecutive buy/sell streaks.
  Lookahead rule: FII/DII net for session t is published after NSE close t, so it is
  merged on date t and used ONLY to predict target_1d (close t+1 > close t). No leakage.
  Features (30): 62 base + 30 = 92 total.
  Protocol: identical model to Phase 3/8 (XGB 150/depth4/lr0.05 hist, median imputer).

  *** DATA-LIMITED PROBE ***: only 142 FII/DII sessions (2026-01-14..2026-08-28) are
  reachable from this host, so the walk-forward is restricted to that window. The
  validated Phase-3 protocol uses a 500-day warmup and 2025-26 holdout; those are NOT
  possible here. This probe therefore uses a shrunk warmup (40 sessions) and treats the
  whole 2026 window as the test period. RESULTS ARE INDICATIVE ONLY, not a holdout pass.
  A definitive Phase 9 requires full FII/DII history (run the fetcher where NSE/NSDL are
  reachable, or supply data/external/fii_dii_flows.csv).

  Gate 1: extended AUC > base AUC on paired OOS predictions (McNemar).
  Sanity alarm: OOS accuracy > 0.60 or AUC > 0.65 => ABORT (leakage).
  Gate 2 (only if Gate 1 passes): G0 naive daily equal-weight vs G1 xgb>=0.5 daily vs
  G2 tercile tilt, evaluated on the 2026 window (labeled as in-probe, not a true holdout).
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

from market_ml.external_data import fetch_fii_dii_flows, compute_fii_dii_features

LOG = ROOT / "reports" / "phase9_run.log"
RETRAIN_EVERY = 7          # probe: retrain weekly within the short window
START_DAYS = 40            # probe: shrunk warmup (protocol uses 500)
MAX_TRAIN_ROWS = 200_000

BASE_EXCL = {"date", "symbol", "open", "high", "low", "close", "volume", "source",
             "target_direction", "next_ret",
             "next_ret_1d", "target_1d", "next_ret_5d", "target_5d", "sector"}
LEAKY = {"target_1d", "next_ret_1d", "next_ret", "target_direction",
         "target_5d", "next_ret_5d"}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_data():
    pq = ROOT / "data/processed/nifty100_features.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(ROOT / "data/processed/nifty100_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")["close"]
    df["next_ret_1d"] = g.shift(-1) / df["close"] - 1.0
    bad = df["next_ret_1d"].abs() > 0.5
    n_bad = int(bad.sum())
    df = df[~bad].dropna(subset=["next_ret_1d"])
    df["target_1d"] = (df["next_ret_1d"] > 0).astype(int)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    log(f"excluded {n_bad} rows with |1d ret|>0.5; rows={len(df):,} "
        f"base_rate={df['target_1d'].mean():.4f}")
    return df


def get_feats(df):
    return [c for c in df.columns
            if c not in BASE_EXCL and df[c].dtype in ("float64", "float32", "int64")]


def merge_fii_dii(df):
    raw = fetch_fii_dii_flows(start_date="2015-01-01", force_refresh=False)
    if raw.empty:
        raise RuntimeError("no FII/DII data fetched")
    feats = compute_fii_dii_features(raw)
    feats["date"] = pd.to_datetime(feats["date"]).astype("datetime64[ns]")
    out = df.merge(feats, on="date", how="left")
    # Restrict to the FII/DII coverage window so train/test both carry the feature.
    lo, hi = feats["date"].min(), feats["date"].max()
    out = out[(out["date"] >= lo) & (out["date"] <= hi)].reset_index(drop=True)
    log(f"FII/DII merged; window {lo.date()}..{hi.date()} rows={len(out):,} "
        f"coverage={out['fii_net_5d_ma'].notna().mean():.3f}")
    return out, [c for c in feats.columns if c != "date"]


def walk_forward(df, feats, out_csv):
    dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    dmap = {d: i for i, d in enumerate(dates)}
    date_idx = df["date"].map(dmap).values
    cols = ["date", "symbol", "actual", "ret_1d", "prob"]
    rows, n_blocks = [], 0
    t0 = time.time()
    for cut in range(START_DAYS, len(dates) - 1, RETRAIN_EVERY):
        trn_mask = date_idx <= cut
        tst_mask = (date_idx > cut) & (date_idx <= min(cut + RETRAIN_EVERY, len(dates) - 1))
        trn, tst = df[trn_mask], df[tst_mask]
        if len(trn) < 500 or len(tst) == 0:
            continue
        if len(trn) > MAX_TRAIN_ROWS:
            trn = trn.iloc[-MAX_TRAIN_ROWS:]
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(trn[feats].values)
        m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              tree_method="hist", n_jobs=-1, random_state=42,
                              verbosity=0).fit(X, trn["target_1d"].values)
        p = m.predict_proba(imp.transform(tst[feats].values))[:, 1]
        for d, s, a, r, pp in zip(tst["date"].values, tst["symbol"].values,
                                  tst["target_1d"].values, tst["next_ret_1d"].values, p):
            rows.append((d, s, int(a), float(r), float(pp)))
        n_blocks += 1
        pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
        log(f"block {n_blocks}: cutoff={dates[cut].date()} train={len(trn):,} "
            f"preds={len(tst):,} elapsed={time.time()-t0:.0f}s")
    return pd.DataFrame(rows, columns=cols), n_blocks


def auc_mw(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def backtest(P, W, cost_rate=0.001):
    assert not P.duplicated(["date", "symbol"]).any(), "duplicate date-symbol predictions"
    all_dates = np.sort(P["date"].unique())
    W = W.reindex(all_dates).fillna(0.0)
    R = P.pivot_table(index="date", columns="symbol", values="ret_1d", aggfunc="mean").reindex(all_dates)
    gross = (W * R.fillna(0.0)).sum(axis=1)
    turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - cost_rate * turnover
    assert (P["ret_1d"].abs() <= 0.5 + 1e-12).all(), "per-symbol |1d ret| > 0.5"
    assert np.isfinite(net).all() and (net > -1).all(), "bad daily net returns"
    assert (net.abs() <= 0.15 + 1e-12).all(), "absurd daily portfolio return"
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all(), "daily exposure exceeds 1.0"
    eq = (1.0 + net).cumprod()
    assert np.isfinite(eq).all() and (eq > 0).all(), "equity not finite/positive"
    mdd = float((eq / eq.cummax() - 1.0).min())
    assert -1.0 <= mdd <= 0.0, "MDD outside [-1, 0]"
    sd = net.std(ddof=0)
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    return {"cumulative_return_net": float(eq.iloc[-1] - 1.0),
            "cumulative_return_gross": float((1.0 + gross).prod() - 1.0),
            "max_drawdown": mdd, "sharpe_net": sharpe,
            "total_turnover": float(turnover.sum()),
            "n_active_mean": float((W > 0).sum(axis=1).mean()),
            "n_days": int(len(all_dates))}, eq, net


def norm(raw):
    W = raw.div(raw.sum(axis=1), axis=0)
    assert (W.sum(axis=1).fillna(0.0) <= 1.0 + 1e-9).all(), "exposure > 1"
    return W.fillna(0.0)


def main():
    t0 = time.time()
    log("PHASE 9: FII/DII FLOW FEATURES - PRE-REGISTERED TWO-GATE TEST (DATA-LIMITED PROBE)")
    df = load_data()
    df["date"] = df["date"].astype("datetime64[ns]")
    df, fii_feats = merge_fii_dii(df)
    base_feats = [c for c in get_feats(df) if c not in fii_feats]
    assert len(base_feats) == 62, f"base feature count {len(base_feats)} != 62 (protocol)"
    ext_feats = base_feats + fii_feats
    assert len(ext_feats) == 62 + len(fii_feats)
    assert not (set(base_feats) & LEAKY), f"LEAKAGE in base: {set(base_feats) & LEAKY}"
    assert not (set(ext_feats) & LEAKY), f"LEAKAGE in ext: {set(ext_feats) & LEAKY}"
    log(f"rows={len(df):,} base_feats={len(base_feats)} ext_feats={len(ext_feats)} "
        f"(anti-leakage assertions OK). WARMUP={START_DAYS} RETRAIN_EVERY={RETRAIN_EVERY} (probe).")

    base_csv = ROOT / "reports/phase9_predictions_base.csv"
    ext_csv = ROOT / "reports/phase9_predictions_ext.csv"
    if base_csv.exists():
        base = pd.read_csv(base_csv, parse_dates=["date"])
        log(f"resumed cached base predictions: {len(base):,} rows")
    else:
        base, nb = walk_forward(df, base_feats, base_csv)
        log(f"base walk-forward done: blocks={nb} oos={len(base):,}")
    if ext_csv.exists():
        ext = pd.read_csv(ext_csv, parse_dates=["date"])
        log(f"resumed cached ext predictions: {len(ext):,} rows")
    else:
        ext, nb = walk_forward(df, ext_feats, ext_csv)
        log(f"ext walk-forward done: blocks={nb} oos={len(ext):,}")

    m = ext.merge(base, on=["date", "symbol"], how="inner", suffixes=("", "_base"))
    assert not m[["prob_base", "prob"]].isna().any().any(), "NaN probabilities after merge"
    log(f"paired n={len(m):,}")
    y = m["actual"].values.astype(int)

    auc_b, auc_e = auc_mw(y, m["prob_base"]), auc_mw(y, m["prob"])
    pb = (m["prob_base"] >= 0.5).astype(int).values
    pe = (m["prob"] >= 0.5).astype(int).values
    if float(np.mean(pe == y)) > 0.60 or auc_e > 0.65:
        log("SANITY ALARM: extended OOS acc>0.60 or AUC>0.65 - leakage suspected; ABORT")
        raise SystemExit(1)
    acc_b, acc_e = float(np.mean(pb == y)), float(np.mean(pe == y))
    br = float(y.mean())
    z_b = (acc_b - br) / np.sqrt(br * (1 - br) / len(m))
    z_e = (acc_e - br) / np.sqrt(br * (1 - br) / len(m))
    b = int(((pb == y) & (pe != y)).sum())
    c = int(((pe == y) & (pb != y)).sum())
    z_mc = (b - c) / np.sqrt(b + c) if (b + c) > 0 else float("nan")
    log(f"base: auc={auc_b:.4f} acc={acc_b:.4f} z={z_b:+.1f} | "
        f"ext: auc={auc_e:.4f} acc={acc_e:.4f} z={z_e:+.1f} | dAUC={auc_e-auc_b:+.4f} | "
        f"McNemar b={b} c={c} z={z_mc:+.2f}")
    gate1 = bool(auc_e > auc_b)
    log(f"GATE 1: {'PASS' if gate1 else 'FAIL'} (probe on 2026 window)")

    results, verdicts, gate2_pass = {}, {}, False
    if gate1:
        P = m[["date", "symbol", "actual", "ret_1d", "prob"]]
        T = P.assign(t=1.0).pivot_table(index="date", columns="symbol", values="t", aggfunc="max")
        probs = P.pivot_table(index="date", columns="symbol", values="prob", aggfunc="mean")
        rk = probs.rank(axis=1, pct=True)
        tw = pd.DataFrame(1.0, index=probs.index, columns=probs.columns)
        tw[rk >= 2.0 / 3.0] = 1.5
        tw[rk <= 1.0 / 3.0] = 0.5
        designs = {
            "G0_naive_daily": norm(T),
            "G1_xgb_binary_daily": norm((probs >= 0.5).astype(float).where(T.notna())),
            "G2_xgb_tercile_daily": norm(tw.where(T.notna())),
        }
        for name, W in designs.items():
            full, _, net = backtest(P, W)
            hc = {}
            for bps in (0.001, 0.0025):
                r, _, _ = backtest(P, W, cost_rate=bps)
                hc[f"{bps}"] = {"net": r["cumulative_return_net"], "sharpe": r["sharpe_net"]}
            results[name] = {"full_window": full, "holdout_costs_probe": hc}
            log(f"{name:22s} net={full['cumulative_return_net']:+8.3f} sh={full['sharpe_net']:5.2f} "
                f"to={full['total_turnover']:7.1f} | @25 net={hc['0.0025']['net']:+7.3f} "
                f"sh={hc['0.0025']['sharpe']:5.2f}")
        ref = results["G0_naive_daily"]["holdout_costs_probe"]
        for name in ("G1_xgb_binary_daily", "G2_xgb_tercile_daily"):
            cde = results[name]["holdout_costs_probe"]
            ok = all(cde[k]["net"] > ref[k]["net"] and cde[k]["sharpe"] > ref[k]["sharpe"] for k in cde)
            verdicts[name] = {"pass": bool(ok), "design": cde, "reference": ref}
            log(f"VERDICT {name}: {'PASS' if ok else 'FAIL'} vs G0 "
                f"(probe window only - NOT a true holdout)")
        gate2_pass = any(v["pass"] for v in verdicts.values())

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "data_limited_probe": True,
           "fii_dii_window": [str(df["date"].min().date()), str(df["date"].max().date())],
           "protocol_deviation": "warmup=40 (protocol 500), no 2025-26 holdout (only reachable 2026 FII/DII)",
           "n_predictions_ext": int(len(ext)), "n_paired": int(len(m)),
           "base_feats": len(base_feats), "ext_feats": len(ext_feats),
           "fii_dii_features": fii_feats,
           "auc_base": auc_b, "auc_ext": auc_e, "acc_base": acc_b, "acc_ext": acc_e,
           "z_base": float(z_b), "z_ext": float(z_e),
           "mcnemar_b": b, "mcnemar_c": c, "mcnemar_z": float(z_mc),
           "gate1_pass": gate1, "gate2_pass": bool(gate2_pass),
           "results": results, "verdicts": verdicts,
           "pre_registration": "Gate1: ext AUC>base AUC (paired, McNemar). Sanity acc>0.60/AUC>0.65 aborts. "
                               "Gate2 (if G1): G1/G2 beat G0 on the 2026 probe window net AND Sharpe at 10/25 bps."}
    (ROOT / "reports/phase9_results.json").write_text(json.dumps(out, indent=2))
    log(f"DONE in {time.time()-t0:.0f}s -> reports/phase9_results.json | "
        f"gate1={'PASS' if gate1 else 'FAIL'} gate2={'PASS' if gate2_pass else 'FAIL/NA'}")


if __name__ == "__main__":
    main()
