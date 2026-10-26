
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/btc/BTCUSDT_1m_event_tp_sl_matrix.csv")
SPLIT_OUTPUT = Path("data/btc/BTCUSDT_event_oos_split.csv")
SUMMARY_OUTPUT = Path("data/btc/BTCUSDT_event_oos_summary.csv")

TP = 0.30
SL = 0.30
TRAIN_FRACTION = 0.70


def calculate_metrics(group):
    group = group.sort_values("entry_timestamp")
    returns = group["net_return_pct"].dropna()

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    positive_sum = wins.sum()
    negative_sum = -losses.sum()

    if negative_sum > 0:
        pf = positive_sum / negative_sum
    else:
        pf = np.inf if positive_sum > 0 else np.nan

    # Additive cumulative net-return drawdown, in percentage points.
    equity = returns.cumsum()
    equity_with_start = pd.concat(
        [pd.Series([0.0]), equity.reset_index(drop=True)],
        ignore_index=True,
    )
    running_peak = equity_with_start.cummax()
    drawdown = equity_with_start - running_peak
    max_drawdown = abs(drawdown.min())

    return {
        "trades": len(returns),
        "wins": int((returns > 0).sum()),
        "losses": int((returns < 0).sum()),
        "zero_returns": int((returns == 0).sum()),
        "win_rate_pct": (
            100 * (returns > 0).mean() if len(returns) else np.nan
        ),
        "profit_factor": pf,
        "avg_net_return_pct": (
            returns.mean() if len(returns) else np.nan
        ),
        "total_net_return_pct": returns.sum(),
        "max_drawdown_pct": max_drawdown,
        "tp_exits": int((group["exit_reason"] == "TP").sum()),
        "sl_exits": int((group["exit_reason"] == "SL").sum()),
        "time_exits": int((group["exit_reason"] == "TIME").sum()),
    }


def add_summary(rows, period, group, dimension, value):
    row = {
        "period": period,
        "dimension": dimension,
        "category": value,
    }
    row.update(calculate_metrics(group))
    rows.append(row)


# ---------------------------------------------------------
# 1. Load and filter the fixed TP/SL configuration
# ---------------------------------------------------------
df = pd.read_csv(INPUT)

required = [
    "event_timestamp",
    "event_type",
    "direction",
    "entry_timestamp",
    "exit_reason",
    "net_return_pct",
    "tp_pct",
    "sl_pct",
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[
    np.isclose(df["tp_pct"], TP)
    & np.isclose(df["sl_pct"], SL)
].copy()

if df.empty:
    raise ValueError("No rows found for the selected TP/SL settings.")

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"], utc=True, errors="coerce"
)
df["entry_timestamp"] = pd.to_datetime(
    df["entry_timestamp"], utc=True, errors="coerce"
)
df["net_return_pct"] = pd.to_numeric(
    df["net_return_pct"], errors="coerce"
)

df = df.dropna(
    subset=["event_timestamp", "entry_timestamp", "net_return_pct"]
)

df = df.sort_values("event_timestamp").reset_index(drop=True)

# The split must happen at the EVENT level, not the trade-row level.
event_dates = (
    df[["event_timestamp"]]
    .drop_duplicates()
    .sort_values("event_timestamp")
    .reset_index(drop=True)
)

n_events = len(event_dates)
if n_events < 10:
    raise ValueError(
        f"Only {n_events} unique events; insufficient for this split."
    )

split_index = int(np.floor(n_events * TRAIN_FRACTION))
split_index = max(1, min(split_index, n_events - 1))

cutoff = event_dates.loc[split_index, "event_timestamp"]

df["period"] = np.where(
    df["event_timestamp"] < cutoff,
    "TRAIN_RESEARCH",
    "HOLDOUT",
)

train_events = df.loc[
    df["period"] == "TRAIN_RESEARCH", "event_timestamp"
].nunique()
holdout_events = df.loc[
    df["period"] == "HOLDOUT", "event_timestamp"
].nunique()

# Every event must belong to exactly one period.
assert train_events + holdout_events == n_events

# ---------------------------------------------------------
# 2. Save the split so dates can be inspected
# ---------------------------------------------------------
df.to_csv(SPLIT_OUTPUT, index=False)

# ---------------------------------------------------------
# 3. Summarize overall, by direction, and by event type
# ---------------------------------------------------------
summary_rows = []

for period in ["TRAIN_RESEARCH", "HOLDOUT"]:
    period_df = df[df["period"] == period]

    add_summary(
        summary_rows, period, period_df, "OVERALL", "ALL"
    )

    for direction, group in period_df.groupby("direction"):
        add_summary(
            summary_rows, period, group,
            "DIRECTION", direction
        )

    for event_type, group in period_df.groupby("event_type"):
        add_summary(
            summary_rows, period, group,
            "EVENT_TYPE", event_type
        )

    for (event_type, direction), group in period_df.groupby(
        ["event_type", "direction"]
    ):
        add_summary(
            summary_rows, period, group,
            "EVENT_TYPE_AND_DIRECTION",
            f"{event_type} | {direction}",
        )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUTPUT, index=False)

# ---------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------
print("=" * 78)
print("CHRONOLOGICAL EVENT STRATEGY VALIDATION")
print("=" * 78)
print(f"Fixed TP: {TP}% | Fixed SL: {SL}%")
print(f"Unique events: {n_events}")
print(f"Research-period events: {train_events}")
print(f"Holdout events: {holdout_events}")
print(f"Holdout begins: {cutoff}")
print(f"Training range: {df.loc[df.period == 'TRAIN_RESEARCH', 'event_timestamp'].min()}")
print(f"Holdout range: {df.loc[df.period == 'HOLDOUT', 'event_timestamp'].min()} "
      f"to {df.loc[df.period == 'HOLDOUT', 'event_timestamp'].max()}")

columns = [
    "period", "dimension", "category", "trades",
    "wins", "losses", "win_rate_pct", "profit_factor",
    "avg_net_return_pct", "total_net_return_pct",
    "max_drawdown_pct", "tp_exits", "sl_exits", "time_exits",
]

print("\nOVERALL RESULTS")
print(
    summary[summary["dimension"] == "OVERALL"][columns]
    .to_string(index=False)
)

print("\nRESULTS BY DIRECTION")
print(
    summary[summary["dimension"] == "DIRECTION"][columns]
    .to_string(index=False)
)

print("\nRESULTS BY EVENT TYPE AND DIRECTION")
print(
    summary[
        summary["dimension"] == "EVENT_TYPE_AND_DIRECTION"
    ][columns].to_string(index=False)
)

print("\nSaved files:")
print(f"  {SPLIT_OUTPUT}")
print(f"  {SUMMARY_OUTPUT}")

if holdout_events < 10:
    print(
        "\nWARNING: Fewer than 10 holdout events. "
        "Treat these results as highly uncertain."
    )

print(
    "\nIMPORTANT: This is a preliminary chronological holdout, "
    "not proof of a tradable edge."
)