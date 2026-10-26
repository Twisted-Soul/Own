
from pathlib import Path
import numpy as np
import pandas as pd

file = Path("data/btc/BTCUSDT_1m_event_tp_sl_matrix.csv")
df = pd.read_csv(file)

df = df[
    np.isclose(df["tp_pct"], 0.30)
    & np.isclose(df["sl_pct"], 0.30)
].copy()

returns = df["net_return_pct"].dropna()
gains = returns[returns > 0].sum()
losses = -returns[returns < 0].sum()

print("=" * 65)
print("AUDIT: ORIGINAL 1m EVENT TRADE OUTCOMES")
print("=" * 65)

print(f"Trades: {len(df)}")
print(f"Positive net returns: {(returns > 0).sum()}")
print(f"Negative net returns: {(returns < 0).sum()}")
print(f"Zero net returns: {(returns == 0).sum()}")
print(f"Exit reasons:\n{df['exit_reason'].value_counts(dropna=False)}")
print(f"Gross sum of positive net returns: {gains:.6f}")
print(f"Absolute sum of negative net returns: {losses:.6f}")
print(f"Profit factor: {gains / losses if losses else np.inf:.6f}")
print(f"Average net return: {returns.mean():.6f}%")
print(f"Total net return: {returns.sum():.6f}%")

print("\nOUTCOMES BY DIRECTION")
print(
    df.groupby("direction", dropna=False).agg(
        trades=("net_return_pct", "count"),
        wins=("net_return_pct", lambda x: (x > 0).sum()),
        losses=("net_return_pct", lambda x: (x < 0).sum()),
        average_return=("net_return_pct", "mean"),
        total_return=("net_return_pct", "sum"),
    ).to_string()
)

print("\nINDIVIDUAL TRADE OUTCOMES")
columns = [
    "event_timestamp", "event_type", "direction",
    "entry_timestamp", "exit_timestamp", "exit_reason",
    "gross_return_pct", "net_return_pct",
]
print(df[columns].sort_values("entry_timestamp").to_string(index=False))