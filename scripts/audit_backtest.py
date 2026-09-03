#!/usr/bin/env python3
"""Audit and fix backtest calculations."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    df = pd.read_csv(ROOT / "data/processed/predictions.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    threshold = 0.55
    df['signal'] = (df['predicted_probability'] >= threshold).astype(int)
    df['position_change'] = df.groupby('symbol')['signal'].diff().abs().fillna(0)
    df.loc[df.groupby('symbol').cumcount() == 0, 'position_change'] = df.loc[df.groupby('symbol').cumcount() == 0, 'signal']
    df['cost'] = df['position_change'] * 0.0015
    df['net_return'] = df['signal'] * df['actual_return'] - df['cost']
    
    daily = df.groupby('date').agg(n_positions=('signal', 'sum'), net_sum=('net_return', 'sum')).reset_index()
    daily['daily_net'] = (daily['net_sum'] / daily['n_positions'].replace(0, np.nan)).fillna(0).clip(-0.5, 0.5)
    
    cum_net = float((1 + daily['daily_net']).prod() - 1)
    equity = (1 + daily['daily_net']).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())
    sharpe = float(daily['daily_net'].mean() / daily['daily_net'].std(ddof=0) * np.sqrt(252))
    final_eq = float(equity.iloc[-1])
    trades = len(df[df['signal'] == 1])
    turnover = float(df['position_change'].mean())
    accuracy = float((df['predicted_direction'] == df['actual_direction']).mean())
    naive_acc = float((df['actual_direction'] == 1).mean())
    
    print(f"Corrected cumulative return: {cum_net:.6f}")
    print(f"Corrected max drawdown: {max_dd:.6f}")
    print(f"Corrected Sharpe ratio: {sharpe:.6f}")
    print(f"Number of trades: {trades}")
    print(f"Total turnover: {turnover:.6f}")
    print(f"Daily return min: {daily['daily_net'].min():.6f}")
    print(f"Daily return max: {daily['daily_net'].max():.6f}")
    print(f"Final equity: {final_eq:.6f}")
    print(f"Model accuracy: {accuracy:.6f}")
    print(f"Naive baseline accuracy: {naive_acc:.6f}")
    
    report = f"""# Backtest Audit Report

## Corrected Metrics
- Cumulative return: {cum_net:.6f} ({cum_net*100:.2f}%)
- Max drawdown: {max_dd:.6f} ({max_dd*100:.2f}%)
- Sharpe ratio: {sharpe:.6f}
- Number of trades: {trades}
- Turnover: {turnover:.6f}
- Daily return min: {daily['daily_net'].min():.6f}
- Daily return max: {daily['daily_net'].max():.6f}
- Final equity: {final_eq:.6f}
- Model accuracy: {accuracy:.6f}
- Naive baseline accuracy: {naive_acc:.6f}

## Model vs Baseline
- Accuracy edge: {accuracy - naive_acc:+.6f}
"""
    with open(ROOT / "reports/backtest_audit.md", "w") as f:
        f.write(report)
    
    json.dump({"corrected": {"cum": cum_net, "dd": max_dd, "sharpe": sharpe, "trades": trades, "turnover": turnover, "final_eq": final_eq}, "accuracy": accuracy, "naive_acc": naive_acc}, open(ROOT / "reports/backtest_audit.json", "w"), indent=2)
    daily.to_csv(ROOT / "reports/backtest_daily_equity.csv", index=False)
    df.to_csv(ROOT / "reports/backtest_trade_log.csv", index=False)
    print("Done")

if __name__ == "__main__":
    main()