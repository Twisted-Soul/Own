import pandas as pd
from pathlib import Path

SOURCE = Path("data/btc/BTCUSDT_chronological_trades.csv")

df = pd.read_csv(SOURCE)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

# Audit the selected TP/SL configuration.
df = df[(df["tp_pct"] == 0.20) & (df["sl_pct"] == 0.20)].copy()

print("=== TRADE INTEGRITY AUDIT ===")
print("Total selected trades:", len(df))

for split, group in df.groupby("split"):
    group = group.sort_values("entry_time").reset_index(drop=True)

    print(f"\n--- {split} ---")
    print("Trades:", len(group))

    if group.empty:
        continue

    duration = (
        group["exit_time"] - group["entry_time"]
    ).dt.total_seconds() / 60

    print("Minimum holding minutes:", duration.min())
    print("Maximum holding minutes:", duration.max())
    print("Median holding minutes:", duration.median())

    # A new entry must not occur before the previous position exits.
    previous_exit = group["exit_time"].shift(1)
    overlap = group["entry_time"] < previous_exit

    print("Overlapping trades:", int(overlap.sum()))

    # Count short intervals between consecutive entries.
    entry_gap = (
        group["entry_time"].diff().dt.total_seconds() / 60
    )

    print("Entries less than 5 minutes apart:", int((entry_gap < 5).sum()))
    print("Entries less than 60 minutes apart:", int((entry_gap < 60).sum()))

    print("\nFirst 10 trades:")
    print(
        group[
            ["entry_time", "exit_time", "signal",
             "exit_reason", "gross_return_pct"]
        ].head(10).to_string(index=False)
    )

print("\nAudit complete. No files modified.")
