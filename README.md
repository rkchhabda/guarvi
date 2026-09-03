# Indian Stock Market Movement Prediction

Predicts next-day direction (up/down) for an Indian stock/index using OHLCV
data, technical indicators, and walk-forward validation.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (synthetic demo data)

```bash
python scripts/build_dataset.py --synthetic
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/predict_latest.py --synthetic
```

## Using real data

Place a CSV at `data/raw/<TICKER>.csv` with columns
`date, open, high, low, close, volume`, then:

```bash
python scripts/build_dataset.py --ticker TCS
python scripts/train_model.py --ticker TCS
python scripts/evaluate_model.py --ticker TCS
python scripts/predict_latest.py --ticker TCS
```

No data source credentials are bundled. If you want automated downloads,
ask before adding any API keys.

## Layout

- `src/market_ml/` — library code (`data`, `features`, `labels`, `splits`, `model`)
- `scripts/` — CLI entry points for the pipeline stages
- `data/raw/` — input OHLCV CSVs; `data/processed/` — engineered datasets
- `models/` — trained artifacts + metadata JSON
- `reports/` — evaluation reports (markdown + JSON)
- `tests/` — pytest suite

## Methodology

- **Labels**: 1 if tomorrow's close > today's close, else 0.
- **Features**: returns, log returns, SMA/EMA ratios, RSI, MACD, Bollinger
  Bands, ATR, rolling volatility, and volume features — all deterministic.
- **Validation**: expanding-window walk-forward splits only. Random
  train/test splits are never used on market data.
- **Models**: Logistic Regression, Random Forest, XGBoost.

## Important disclaimer

Directional accuracy reported here is a historical backtest estimate.
It is **not** a guarantee of future performance, and this project is not
investment advice.
