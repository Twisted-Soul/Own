from pathlib import Path

import pandas as pd


RAW_FILE = Path(
    "data/btc/BTCUSDT_5m_full_raw.csv"
)

EVENT_FILE = Path(
    "data/btc/BTCUSDT_1m_event_reactions.csv"
)


print("=" * 70)
print("CHECK 5m DATA GAP AGAINST EVENT HISTORY")
print("=" * 70)


# ============================================================
# LOAD 5m DATA
# ============================================================

df = pd.read_csv(RAW_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
    errors="coerce",
)

df = df.sort_values("timestamps").reset_index(drop=True)


# ============================================================
# FIND GAPS
# ============================================================

diff = df["timestamps"].diff()

expected = pd.Timedelta(minutes=5)

gaps = df.loc[
    diff > expected,
    [
        "timestamps",
    ],
].copy()

gaps["previous_timestamp"] = (
    df["timestamps"].shift(1)
)

gaps["gap"] = (
    gaps["timestamps"]
    - gaps["previous_timestamp"]
)

print()
print("5m GAPS")
print("-" * 70)

if gaps.empty:

    print("No gaps found.")

else:

    for _, row in gaps.iterrows():

        print(
            f"{row['previous_timestamp']} "
            f"→ "
            f"{row['timestamps']} "
            f"= "
            f"{row['gap']}"
        )


# ============================================================
# LOAD EVENTS
# ============================================================

events = pd.read_csv(
    EVENT_FILE
)

events["event_timestamp"] = pd.to_datetime(
    events["event_timestamp"],
    utc=True,
    errors="coerce",
)

events = (
    events[
        [
            "event_timestamp",
            "event_type",
            "importance",
        ]
    ]
    .drop_duplicates()
    .sort_values("event_timestamp")
    .reset_index(drop=True)
)


# ============================================================
# CHECK EVENTS INSIDE GAPS
# ============================================================

print()
print("=" * 70)
print("EVENTS INSIDE / NEAR GAPS")
print("=" * 70)

for _, gap in gaps.iterrows():

    start = gap["previous_timestamp"]
    end = gap["timestamps"]

    # Include events during the missing interval.
    affected = events[
        (
            events["event_timestamp"] > start
        )
        &
        (
            events["event_timestamp"] < end
        )
    ]

    print()
    print(
        f"Gap: {start} → {end}"
    )

    if affected.empty:

        print("No event occurs inside this gap.")

    else:

        print(
            f"Affected events: "
            f"{len(affected)}"
        )

        print(
            affected.to_string(
                index=False
            )
        )


# ============================================================
# FINAL EVENT COVERAGE
# ============================================================

print()
print("=" * 70)
print("EVENT COVERAGE")
print("=" * 70)

data_start = df["timestamps"].min()
data_end = df["timestamps"].max()

covered = events[
    (
        events["event_timestamp"] >= data_start
    )
    &
    (
        events["event_timestamp"] <= data_end
    )
]

print(
    f"Total events: {len(events)}"
)

print(
    f"Events inside overall data range: "
    f"{len(covered)}"
)