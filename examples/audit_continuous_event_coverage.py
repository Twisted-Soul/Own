
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/btc")
HISTORY = BASE / "BTCUSDT_1m_event_history.csv"
CONTINUOUS = BASE / "BTCUSDT_1m_continuous_2020_2026.parquet"
OUTPUT = BASE / "BTCUSDT_continuous_event_coverage_audit.csv"

print("Loading event history...")
events_df = pd.read_csv(HISTORY)
events_df["timestamp"] = pd.to_datetime(
    events_df["timestamp"], utc=True
)
events_df["event_timestamp"] = pd.to_datetime(
    events_df["event_timestamp"], utc=True
)

events = (
    events_df[["event_timestamp", "event_type"]]
    .drop_duplicates()
    .sort_values("event_timestamp")
    .reset_index(drop=True)
)

print("Loading continuous candles...")
candles = pd.read_parquet(
    CONTINUOUS,
    columns=["timestamp_utc", "close"]
)
candles["timestamp_utc"] = pd.to_datetime(
    candles["timestamp_utc"], utc=True
)
candles = candles.sort_values("timestamp_utc")

if candles["timestamp_utc"].duplicated().any():
    raise ValueError("Duplicate timestamps in continuous dataset.")

start = candles["timestamp_utc"].min()
end = candles["timestamp_utc"].max()

# Compare the original event-history closes with continuous closes
# at identical candle timestamps.
print("Comparing prices at matching timestamps...")
comparison = events_df[
    ["timestamp", "close"]
].drop_duplicates(subset=["timestamp"]).merge(
    candles,
    left_on="timestamp",
    right_on="timestamp_utc",
    how="inner",
    validate="one_to_one",
    suffixes=("_history", "_continuous"),
)

if len(comparison):
    comparison["abs_diff_pct"] = (
        (comparison["close_continuous"] /
         comparison["close_history"] - 1).abs() * 100
    )
    comparison["signed_diff_pct"] = (
        (comparison["close_continuous"] /
         comparison["close_history"] - 1) * 100
    )

# The original strategy needs a window around each release.
# Its historical window starts about 89 minutes before release
# and ends about 179 minutes after release.
records = []

for _, event in events.iterrows():
    t = event["event_timestamp"]
    window_start = t - pd.Timedelta(minutes=89)
    window_end = t + pd.Timedelta(minutes=179)

    records.append({
        "event_timestamp": t,
        "event_type": event["event_type"],
        "full_window_available": (
            window_start >= start and window_end <= end
        ),
        "event_time_in_dataset": start <= t <= end,
        "window_start": window_start,
        "window_end": window_end,
    })

audit = pd.DataFrame(records)

# Count actual candles in each event window.
times = candles["timestamp_utc"]
window_counts = []
for _, row in audit.iterrows():
    count = int(
        ((times >= row["window_start"]) &
         (times <= row["window_end"])).sum()
    )
    window_counts.append(count)

audit["candles_in_window"] = window_counts
audit["expected_window_candles"] = 269
audit["window_complete_269_candles"] = (
    audit["candles_in_window"] == 269
)

audit.to_csv(OUTPUT, index=False)

print("\n" + "=" * 72)
print("CONTINUOUS DATA COVERAGE")
print("=" * 72)
print(f"Dataset starts: {start}")
print(f"Dataset ends:   {end}")
print(f"Unique events in history: {len(events)}")
print(
    "Events with full window by date bounds: "
    f"{int(audit['full_window_available'].sum())}/{len(audit)}"
)
print(
    "Events with exactly 269 candles: "
    f"{int(audit['window_complete_269_candles'].sum())}/{len(audit)}"
)
print(
    "Events inside dataset date range: "
    f"{int(audit['event_time_in_dataset'].sum())}/{len(audit)}"
)

print("\n" + "=" * 72)
print("PRICE ALIGNMENT AT MATCHING TIMESTAMPS")
print("=" * 72)
print(f"Matched unique candle timestamps: {len(comparison):,}")

if len(comparison):
    print(
        "Median absolute close difference: "
        f"{comparison['abs_diff_pct'].median():.6f}%"
    )
    print(
        "95th percentile absolute difference: "
        f"{comparison['abs_diff_pct'].quantile(0.95):.6f}%"
    )
    print(
        "Maximum absolute close difference: "
        f"{comparison['abs_diff_pct'].max():.6f}%"
    )
    print(
        "Median signed close difference: "
        f"{comparison['signed_diff_pct'].median():.6f}%"
    )
    print("\nLargest differences:")
    print(
        comparison.nlargest(5, "abs_diff_pct")[
            ["timestamp", "close_history", "close_continuous",
             "signed_diff_pct"]
        ].to_string(index=False)
    )
else:
    print("No matching timestamps were found.")

print("\nEvent coverage details:")
print(
    audit[
        ["event_timestamp", "event_type",
         "candles_in_window", "full_window_available"]
    ].to_string(index=False)
)

print(f"\nSaved audit: {OUTPUT}")