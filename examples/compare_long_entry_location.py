from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/btc/BTCUSDT_event_long_forward_outcomes_30m_60m.csv"
)

if not INPUT.exists():
    raise FileNotFoundError(f"Input not found: {INPUT}")

df = pd.read_csv(INPUT)
df = df[df["status"] == "OK"].copy()

required = [
    "prior_60m_range_position_pct",
    "return_30m_net_est_pct",
    "return_60m_net_est_pct",
    "mfe_30m_pct",
    "mae_30m_pct",
    "mfe_60m_pct",
    "mae_60m_pct",
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["prior_60m_range_position_pct"] = pd.to_numeric(
    df["prior_60m_range_position_pct"], errors="coerce"
)

df = df.dropna(subset=["prior_60m_range_position_pct"]).copy()

df["entry_location"] = pd.cut(
    df["prior_60m_range_position_pct"],
    bins=[float("-inf"), 25, 50, 75, float("inf")],
    labels=[
        "At/below 25%",
        "25–50%",
        "50–75%",
        "Above 75%",
    ],
    right=True,
    include_lowest=True,
    ordered=True,
)

print("=== ENTRY LOCATION AUDIT ===")
print(f"Valid LONG setups: {len(df)}")
print("Location uses only the 60 minutes BEFORE entry.")
print("Net estimates subtract the existing 0.14% cost assumption.")
print("No entry thresholds or strategy parameters were changed.")

for horizon in (30, 60):
    net_col = f"return_{horizon}m_net_est_pct"
    mfe_col = f"mfe_{horizon}m_pct"
    mae_col = f"mae_{horizon}m_pct"

    grouped = df.groupby(
        "entry_location", observed=False
    ).agg(
        setups=(net_col, "count"),
        positive_net_rate_pct=(
            net_col, lambda s: (s > 0).mean() * 100
        ),
        mean_gross_mfe_pct=(mfe_col, "mean"),
        mean_adverse_mae_pct=(mae_col, "mean"),
        mean_net_est_pct=(net_col, "mean"),
        median_net_est_pct=(net_col, "median"),
    )

    print(f"\n=== {horizon}-MINUTE OUTCOMES ===")
    print(grouped.round(4).to_string())

print("\n=== INDIVIDUAL SETUPS ===")
columns = [
    "event_timestamp",
    "event_type",
    "entry_timestamp",
    "prior_60m_range_position_pct",
    "distance_from_prior_60m_low_pct",
    "return_30m_net_est_pct",
    "return_60m_net_est_pct",
    "mfe_60m_pct",
    "mae_60m_pct",
]
columns = [c for c in columns if c in df.columns]
print(df[columns].sort_values(
    "prior_60m_range_position_pct"
).round(4).to_string(index=False))

print("\nRead-only audit complete. No CSV files were written.")
