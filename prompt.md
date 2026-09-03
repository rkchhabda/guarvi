\# AGENTS.md



\## Project overview

This repository builds an Indian stock market movement prediction system in Python.

The first version predicts next-day direction using OHLCV data, technical indicators, and walk-forward validation.



\## Tech stack

\- Python 3.11+

\- pandas, numpy, scikit-learn

\- xgboost or lightgbm

\- ta or pandas\_ta\_classic

\- pytest for tests



\## Core commands

\- Install dependencies: `pip install -r requirements.txt`

\- Build dataset: `python scripts/build\_dataset.py`

\- Train model: `python scripts/train\_model.py`

\- Evaluate model: `python scripts/evaluate\_model.py`

\- Run latest prediction: `python scripts/predict\_latest.py`

\- Run tests: `pytest -q`



\## Repository rules

\- Use time-series split only.

\- Do not use random train-test split for market data.

\- Keep modules small and focused.

\- Save processed data in `data/processed/`.

\- Save model artifacts in `models/`.

\- Save reports and figures in `reports/`.



\## Modeling rules

\- Start with Logistic Regression, Random Forest, and XGBoost or LightGBM.

\- Use directional accuracy as the primary benchmark.

\- Do not claim 60% accuracy as guaranteed.

\- Treat 60% as a target to evaluate, not a promise.



\## Feature rules

\- Include returns, log returns, SMA/EMA, RSI, MACD, Bollinger Bands, ATR, volume features, and rolling volatility.

\- Keep feature generation deterministic.

\- Drop or handle NaNs after indicator creation.



\## Testing rules

\- Add tests for feature creation.

\- Add tests for label generation.

\- Add tests for walk-forward split logic.

\- Keep tests fast and deterministic.



\## Safety and boundaries

\- Never commit secrets, API keys, or credentials.

\- If data source access is unclear, ask before adding credentials.

\- Avoid unrelated refactors unless required by the task.



\## Working style

\- Inspect existing files first.

\- Write a short implementation plan before changes.

\- Implement file by file.

\- Verify each major step with tests or scripts.

\- Keep changes minimal and reviewable.

