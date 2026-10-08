from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

EVENT_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_event_reactions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_event_confirmation.csv"


print("=" * 80)
print("KRONOS EVENT CONFIRMATION ANALYSIS")
print("=" * 80)

df = pd.read_csv(EVENT_FILE)

print(f"Loaded event reactions: {len(df)}")


# ---------------------------------------------------------------------
# Normalize timestamp
# ---------------------------------------------------------------------

df["release_timestamp"] = pd.to_datetime(
    df["release_timestamp"],
    utc=True,
    errors="coerce"
)


# ---------------------------------------------------------------------
# Event type
# ---------------------------------------------------------------------

EVENT_COL = "event_type"

if EVENT_COL not in df.columns:
    raise ValueError(
        f"Missing {EVENT_COL}. Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Remove duplicate release observations
#
# Some BLS releases contain multiple rows for the same actual release
# timestamp. We only want one BTC reaction per release.
# ---------------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=["release_timestamp"]
).copy()

df = df.drop_duplicates(
    subset=[EVENT_COL, "release_timestamp"]
).copy()

print(f"Removed duplicate event rows: {before - len(df)}")
print(f"Unique events: {len(df)}")


# ---------------------------------------------------------------------
# Identify future-return columns
# ---------------------------------------------------------------------

return_columns = {}

for horizon in ["5m", "15m", "30m", "1h", "2h", "4h"]:

    column = f"future_return_{horizon}"

    if column in df.columns:
        return_columns[horizon] = column


if not return_columns:
    raise ValueError(
        "No future return columns found. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Convert returns to numeric
# ---------------------------------------------------------------------

for column in return_columns.values():
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------------------
# Direction helper
# ---------------------------------------------------------------------

def direction(value):

    if pd.isna(value):
        return np.nan

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


# ---------------------------------------------------------------------
# Overall reaction persistence
#
# Example:
#
# 5m -> 30m
#
# If BTC moved +0.20% after 5 minutes and +0.35% after 30 minutes,
# the reaction is confirmed.
#
# If BTC moved +0.20% after 5 minutes but -0.30% after 30 minutes,
# the reaction failed.
# ---------------------------------------------------------------------

pairs = [
    ("5m", "15m"),
    ("5m", "30m"),
    ("5m", "1h"),
    ("15m", "30m"),
    ("15m", "1h"),
    ("15m", "2h"),
    ("30m", "1h"),
    ("30m", "2h"),
    ("30m", "4h"),
]


print()
print("=" * 80)
print("REACTION PERSISTENCE")
print("=" * 80)

overall_results = []


for early, late in pairs:

    early_col = return_columns.get(early)
    late_col = return_columns.get(late)

    if early_col is None or late_col is None:
        continue

    temp = df[
        [EVENT_COL, "release_timestamp", early_col, late_col]
    ].copy()

    temp["early_direction"] = temp[early_col].apply(direction)
    temp["late_direction"] = temp[late_col].apply(direction)

    temp = temp.dropna(
        subset=[
            "early_direction",
            "late_direction"
        ]
    )

    temp = temp[
        (temp["early_direction"] != 0)
        &
        (temp["late_direction"] != 0)
    ]

    if len(temp) == 0:
        continue

    temp["confirmed"] = (
        temp["early_direction"]
        ==
        temp["late_direction"]
    )

    confirmation_rate = (
        temp["confirmed"].mean() * 100
    )

    # Align the later return with the direction of the
    # initial reaction.
    temp["directional_late_return"] = (
        temp[late_col]
        *
        temp["early_direction"]
    )

    avg_directional_return = (
        temp["directional_late_return"].mean()
    )

    median_directional_return = (
        temp["directional_late_return"].median()
    )

    overall_results.append(
        {
            "analysis": "overall_persistence",
            "early_horizon": early,
            "late_horizon": late,
            "events": len(temp),
            "confirmation_pct": confirmation_rate,
            "avg_directional_late_return_pct":
                avg_directional_return * 100,
            "median_directional_late_return_pct":
                median_directional_return * 100,
        }
    )

    print()
    print(f"{early} -> {late}")
    print("-" * 60)
    print(f"Events:                  {len(temp)}")
    print(
        f"Direction confirmed:     "
        f"{confirmation_rate:.2f}%"
    )
    print(
        f"Avg directional return:  "
        f"{avg_directional_return * 100:+.4f}%"
    )
    print(
        f"Median directional return:"
        f" {median_directional_return * 100:+.4f}%"
    )


# ---------------------------------------------------------------------
# Event type analysis
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("EVENT-TYPE CONFIRMATION")
print("=" * 80)

event_type_results = []


for event_type, group in df.groupby(EVENT_COL):

    for early, late in pairs:

        early_col = return_columns.get(early)
        late_col = return_columns.get(late)

        if early_col is None or late_col is None:
            continue

        temp = group[
            [early_col, late_col]
        ].copy()

        temp["early_direction"] = temp[early_col].apply(direction)
        temp["late_direction"] = temp[late_col].apply(direction)

        temp = temp.dropna(
            subset=[
                "early_direction",
                "late_direction"
            ]
        )

        temp = temp[
            (temp["early_direction"] != 0)
            &
            (temp["late_direction"] != 0)
        ]

        if len(temp) == 0:
            continue

        temp["confirmed"] = (
            temp["early_direction"]
            ==
            temp["late_direction"]
        )

        confirmation = (
            temp["confirmed"].mean() * 100
        )

        temp["directional_return"] = (
            temp[late_col]
            *
            temp["early_direction"]
        )

        avg_return = (
            temp["directional_return"].mean()
            * 100
        )

        event_type_results.append(
            {
                "analysis": "event_type",
                "event_type": event_type,
                "early_horizon": early,
                "late_horizon": late,
                "events": len(temp),
                "confirmation_pct": confirmation,
                "avg_directional_late_return_pct":
                    avg_return,
            }
        )

        print(
            f"{event_type:12s} | "
            f"{early:>3s}->{late:<3s} | "
            f"n={len(temp):2d} | "
            f"confirm={confirmation:6.2f}% | "
            f"directional={avg_return:+.4f}%"
        )


# ---------------------------------------------------------------------
# Strong initial reaction test
#
# We deliberately use fixed thresholds rather than optimizing them.
#
# 0.10%
# 0.20%
# 0.30%
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("STRONG INITIAL REACTION TEST")
print("=" * 80)

strong_results = []


for threshold in [0.001, 0.002, 0.003]:

    for early, late in pairs:

        early_col = return_columns.get(early)
        late_col = return_columns.get(late)

        if early_col is None or late_col is None:
            continue

        temp = df[
            [early_col, late_col]
        ].copy()

        temp = temp.dropna()

        temp = temp[
            temp[early_col].abs() >= threshold
        ]

        if len(temp) == 0:
            continue

        temp["early_direction"] = (
            temp[early_col].apply(direction)
        )

        temp["late_direction"] = (
            temp[late_col].apply(direction)
        )

        temp = temp[
            (temp["early_direction"] != 0)
            &
            (temp["late_direction"] != 0)
        ]

        if len(temp) == 0:
            continue

        temp["confirmed"] = (
            temp["early_direction"]
            ==
            temp["late_direction"]
        )

        confirmation = (
            temp["confirmed"].mean() * 100
        )

        temp["directional_return"] = (
            temp[late_col]
            *
            temp["early_direction"]
        )

        avg_return = (
            temp["directional_return"].mean()
            * 100
        )

        strong_results.append(
            {
                "analysis": "strong_initial_reaction",
                "threshold_pct": threshold * 100,
                "early_horizon": early,
                "late_horizon": late,
                "events": len(temp),
                "confirmation_pct": confirmation,
                "avg_directional_return_pct":
                    avg_return,
            }
        )

        print(
            f"| initial >= {threshold * 100:.2f}% | "
            f"{early:>3s}->{late:<3s} | "
            f"n={len(temp):2d} | "
            f"confirm={confirmation:6.2f}% | "
            f"directional={avg_return:+.4f}%"
        )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

all_results = (
    overall_results
    +
    event_type_results
    +
    strong_results
)

output = pd.DataFrame(all_results)

output.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 80)
print("OUTPUT")
print("=" * 80)
print(f"Saved: {OUTPUT_FILE}")
print()
print("Analysis complete.")