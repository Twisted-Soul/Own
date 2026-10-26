import pandas as pd
from pathlib import Path

SOURCE = Path("data/btc/BTCUSDT_chronological_trades.csv")

df = pd.read_csv(SOURCE)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")

df = df[(df["tp_pct"] == 0.20) & (df["sl_pct"] == 0.20)].copy()

df["holding_minutes"] = (
    df["exit_time"] - df["entry_time"]
).dt.total_seconds() / 60

columns = [
    "split", "signal_time", "entry_time", "exit_time",
    "signal", "exit_reason", "holding_minutes", "gross_return_pct"
]

print("=== INVALID OR SUSPICIOUS TRADES ===")

invalid = df[
    df["entry_time"].isna()
    | df["exit_time"].isna()
    | (df["holding_minutes"] < 0)
]

print("\nInvalid timestamps or negative durations:", len(invalid))
if not invalid.empty:
    print(invalid[columns].head(20).to_string(index=False))

long_trades = df[df["holding_minutes"] > 60]
print("\nTrades held longer than 60 minutes:", len(long_trades))
if not long_trades.empty:
    print(
        long_trades.sort_values("holding_minutes", ascending=False)
        [columns].head(30).to_string(index=False)
    )

zero_trades = df[df["holding_minutes"] == 0]
print("\nZero-minute trades:", len(zero_trades))
if not zero_trades.empty:
    print(zero_trades[columns].head(20).to_string(index=False))

print("\nExit reasons for trades over 60 minutes:")
if not long_trades.empty:
    print(long_trades["exit_reason"].value_counts().to_string())

print("\nAudit complete. No files modified.")
