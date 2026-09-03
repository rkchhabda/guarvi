"""Quick smoke test for baselines module."""
import sys, warnings, time
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from market_ml.baselines import (
    load_features, prepare_xy, create_walk_forward_folds,
    train_and_evaluate_model, train_previous_direction, train_always_up,
    compute_trading_metrics, aggregate_overall
)

t0 = time.time()
df = load_features("data/processed/nifty100_features.parquet")
X, y, meta = prepare_xy(df)
holdout_mask, folds = create_walk_forward_folds(meta["date"])
print(f"Data loaded: {len(df)} rows, {len(folds)} folds, {holdout_mask.sum()} holdout ({time.time()-t0:.1f}s)")

# Test each model on first5 folds
for model_name in ["always_up", "previous_direction", "logistic_regression", "random_forest", "xgboost"]:
    t1 = time.time()
    if model_name == "always_up":
        fp, fm, hp = train_always_up(df, meta, folds[:5], holdout_mask)
    elif model_name == "previous_direction":
        fp, fm, hp = train_previous_direction(df, meta, folds[:5], holdout_mask)
    else:
        fp, fm, hp = train_and_evaluate_model(df, X, y, meta, model_name, folds[:5], holdout_mask)
    overall = aggregate_overall(fp)
    trading = compute_trading_metrics(fp, threshold=0.55)
    elapsed = time.time()-t1
    print(f"  {model_name}: acc={overall['accuracy']:.4f}, sharpe={trading.sharpe_ratio:.4f}, "
          f"n_fold={len(fp)}, n_hold={len(hp)} ({elapsed:.1f}s)")

print(f"\nTotal: {time.time()-t0:.1f}s")
print("ALL MODELS OK")
