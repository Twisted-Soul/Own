import pandas as pd
from pathlib import Path

SOURCE = Path("data/btc/BTCUSDT_5m_full_signals_experiment.csv")

df = pd.read_csv(SOURCE)
time_col = "timestamps"

if time_col not in df.columns:
    raise ValueError(f"Missing {time_col!r}. Columns: {list(df.columns)}")

df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

invalid = df[time_col].isna().sum()
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

duplicates = df[time_col].duplicated().sum()
df["gap_minutes"] = df[time_col].diff().dt.total_seconds() / 60

print("=== DATASET GAP AUDIT ===")
print("File:", SOURCE)
print("Valid candles:", len(df))
print("Invalid timestamps removed for audit:", invalid)
print("Duplicate timestamps:", duplicates)
print("First candle:", df[time_col].iloc[0])
print("Last candle:", df[time_col].iloc[-1])

print("\n=== GAP COUNTS ===")
for threshold in [5, 10, 30, 60, 360, 1440]:
    count = (df["gap_minutes"] > threshold).sum()
    print(f"Gaps greater than {threshold:>4} minutes: {count}")

print("\n=== LARGEST TIMESTAMP GAPS ===")
gaps = df[df["gap_minutes"] > 5].sort_values(
    "gap_minutes", ascending=False
)

if gaps.empty:
    print("No gaps greater than 5 minutes.")
else:
    for idx in gaps.head(20).index:
        print(
            f"Previous: {df.loc[idx - 1, time_col]} | "
            f"Next: {df.loc[idx, time_col]} | "
            f"Gap: {df.loc[idx, 'gap_minutes']:.1f} minutes"
        )

print("\n=== CANDLES AROUND SUSPICIOUS TRADE ===")
start = pd.Timestamp("2025-09-30 20:00:00", tz="UTC")
end = pd.Timestamp("2025-10-03 01:00:00", tz="UTC")

nearby = df[df[time_col].between(start, end)].copy()

cols = [time_col, "gap_minutes"]
for col in ["open", "high", "low", "close", "signal"]:
    if col in nearby.columns:
        cols.append(col)

print(nearby[cols].to_string(index=False))
print("\nAudit complete. No files modified.")
