from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

EVENT_HISTORY_FILE = (
    BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_history.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_reactions.csv"
)

PRE_WINDOWS = [15, 30]
CHECKPOINTS = [1, 5, 10, 15, 20, 30, 45, 60]


# =============================================================================
# HELPERS
# =============================================================================

def pct_return(start_price, end_price):
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return np.nan

    return (end_price / start_price - 1.0) * 100.0


def direction_from_return(value):
    if pd.isna(value):
        return "UNKNOWN"

    if value > 0:
        return "UP"

    if value < 0:
        return "DOWN"

    return "FLAT"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("KRONOS 1M EVENT REACTION DATASET")
print("=" * 80)

if not EVENT_HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Event history file not found:\n{EVENT_HISTORY_FILE}\n\n"
        "Run collect_1m_event_history.py first."
    )

df = pd.read_csv(EVENT_HISTORY_FILE)

required_columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "event_timestamp",
    "event_type",
    "importance",
    "minutes_from_release",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"],
    utc=True,
)

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "minutes_from_release",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = (
    df.dropna(
        subset=[
            "timestamp",
            "event_timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )
    .sort_values(
        [
            "event_timestamp",
            "timestamp",
        ]
    )
    .reset_index(drop=True)
)

# Make sure each event timestamp is represented once.
events = (
    df[
        [
            "event_timestamp",
            "event_type",
            "importance",
        ]
    ]
    .drop_duplicates(subset=["event_timestamp"])
    .sort_values("event_timestamp")
    .reset_index(drop=True)
)

print(f"Events available: {len(events)}")
print(f"Candles available: {len(df)}")
print()


# =============================================================================
# BUILD EVENT DATASET
# =============================================================================

records = []

for event_index, event in events.iterrows():

    event_timestamp = event["event_timestamp"]
    event_type = event["event_type"]
    importance = event["importance"]

    # -------------------------------------------------------------------------
    # Event-specific candle data
    # -------------------------------------------------------------------------

    event_df = df[
        df["event_timestamp"] == event_timestamp
    ].copy()

    if event_df.empty:
        print(
            f"[{event_index + 1}/{len(events)}] "
            f"{event_timestamp} | NO DATA"
        )
        continue

    event_df = event_df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Release timestamp is the reference point.
    #
    # Do NOT use an undefined "release_timestamp" variable.
    # We explicitly use event_timestamp everywhere.
    # -------------------------------------------------------------------------

    pre_start = event_timestamp - pd.Timedelta(
        value=30,
        unit="m",
    )

    pre_df = event_df[
        (event_df["timestamp"] < event_timestamp)
        & (event_df["timestamp"] >= pre_start)
    ].copy()

    post_df = event_df[
        event_df["timestamp"] >= event_timestamp
    ].copy()

    if post_df.empty:
        print(
            f"[{event_index + 1}/{len(events)}] "
            f"{event_timestamp} | {event_type} | "
            f"NO POST-RELEASE DATA"
        )
        continue

    # -------------------------------------------------------------------------
    # Release price
    # -------------------------------------------------------------------------

    release_candles = post_df[
        post_df["timestamp"] >= event_timestamp
    ]

    if release_candles.empty:
        continue

    release_candle = release_candles.iloc[0]

    release_price = float(release_candle["open"])
    release_candle_timestamp = release_candle["timestamp"]

    # -------------------------------------------------------------------------
    # Base record
    # -------------------------------------------------------------------------

    record = {
        "event_timestamp": event_timestamp,
        "event_type": event_type,
        "importance": importance,
        "release_candle_timestamp": release_candle_timestamp,
        "release_price": release_price,
    }

    # -------------------------------------------------------------------------
    # Pre-release movement
    # -------------------------------------------------------------------------

    for minutes in PRE_WINDOWS:

        target_timestamp = event_timestamp - pd.Timedelta(
            value=minutes,
            unit="m",
        )

        previous_candidates = pre_df[
            pre_df["timestamp"] <= target_timestamp
        ]

        if previous_candidates.empty:
            record[f"pre_return_{minutes}m_pct"] = np.nan
            record[f"pre_price_{minutes}m"] = np.nan
            continue

        previous_candle = previous_candidates.iloc[-1]

        previous_price = float(previous_candle["close"])

        record[f"pre_price_{minutes}m"] = previous_price

        record[f"pre_return_{minutes}m_pct"] = pct_return(
            previous_price,
            release_price,
        )

    # -------------------------------------------------------------------------
    # Fixed reaction checkpoints
    # -------------------------------------------------------------------------

    checkpoint_rows = {}

    for minutes in CHECKPOINTS:

        target_timestamp = event_timestamp + pd.Timedelta(
            value=minutes,
            unit="m",
        )

        candidates = post_df[
            post_df["timestamp"] >= target_timestamp
        ]

        if candidates.empty:
            checkpoint_rows[minutes] = None
            continue

        checkpoint = candidates.iloc[0]

        checkpoint_rows[minutes] = checkpoint

        checkpoint_price = float(checkpoint["close"])

        future_return = pct_return(
            release_price,
            checkpoint_price,
        )

        record[f"price_{minutes}m"] = checkpoint_price

        record[f"return_{minutes}m_pct"] = future_return

        record[f"direction_{minutes}m"] = (
            direction_from_return(future_return)
        )

        record[f"timestamp_{minutes}m"] = checkpoint["timestamp"]

    # -------------------------------------------------------------------------
    # Maximum favorable / adverse movement during first 60 minutes
    # -------------------------------------------------------------------------

    sixty_minute_end = event_timestamp + pd.Timedelta(
        value=60,
        unit="m",
    )

    first_hour = post_df[
        post_df["timestamp"] <= sixty_minute_end
    ].copy()

    if not first_hour.empty:

        max_high = first_hour["high"].max()
        min_low = first_hour["low"].min()

        record["max_favorable_up_60m_pct"] = pct_return(
            release_price,
            max_high,
        )

        record["max_adverse_down_60m_pct"] = pct_return(
            release_price,
            min_low,
        )

        record["max_abs_move_60m_pct"] = max(
            abs(record["max_favorable_up_60m_pct"]),
            abs(record["max_adverse_down_60m_pct"]),
        )

    else:
        record["max_favorable_up_60m_pct"] = np.nan
        record["max_adverse_down_60m_pct"] = np.nan
        record["max_abs_move_60m_pct"] = np.nan

    # -------------------------------------------------------------------------
    # Recovery measurements
    #
    # These are especially important for the Kronos event-reaction idea.
    # -------------------------------------------------------------------------

    def get_checkpoint_price(minutes):
        row = checkpoint_rows.get(minutes)

        if row is None:
            return np.nan

        return float(row["close"])

    price_5 = get_checkpoint_price(5)
    price_15 = get_checkpoint_price(15)
    price_20 = get_checkpoint_price(20)
    price_30 = get_checkpoint_price(30)
    price_45 = get_checkpoint_price(45)

    # 5m -> 15m
    if not pd.isna(price_5) and not pd.isna(price_15):
        record["recovery_5m_to_15m_pct"] = pct_return(
            price_5,
            price_15,
        )
    else:
        record["recovery_5m_to_15m_pct"] = np.nan

    # 15m -> 30m
    if not pd.isna(price_15) and not pd.isna(price_30):
        record["recovery_15m_to_30m_pct"] = pct_return(
            price_15,
            price_30,
        )
    else:
        record["recovery_15m_to_30m_pct"] = np.nan

    # 20m -> 45m
    if not pd.isna(price_20) and not pd.isna(price_45):
        record["recovery_20m_to_45m_pct"] = pct_return(
            price_20,
            price_45,
        )
    else:
        record["recovery_20m_to_45m_pct"] = np.nan

    # -------------------------------------------------------------------------
    # Recovery relative to release price
    # -------------------------------------------------------------------------

    if not pd.isna(price_15):
        record["release_to_15m_pct"] = pct_return(
            release_price,
            price_15,
        )
    else:
        record["release_to_15m_pct"] = np.nan

    if not pd.isna(price_30):
        record["release_to_30m_pct"] = pct_return(
            release_price,
            price_30,
        )
    else:
        record["release_to_30m_pct"] = np.nan

    if not pd.isna(price_45):
        record["release_to_45m_pct"] = pct_return(
            release_price,
            price_45,
        )
    else:
        record["release_to_45m_pct"] = np.nan

    # -------------------------------------------------------------------------
    # MYT timestamp
    # -------------------------------------------------------------------------

    record["event_timestamp_myt"] = (
        event_timestamp
        .tz_convert("Asia/Kuala_Lumpur")
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    )

    records.append(record)

    print(
        f"[{event_index + 1}/{len(events)}] "
        f"{event_timestamp} | "
        f"{event_type} | "
        f"release={release_price:.2f}"
    )


# =============================================================================
# SAVE
# =============================================================================

if not records:
    raise RuntimeError(
        "No event reaction records were created."
    )

result = pd.DataFrame(records)

result = result.sort_values(
    "event_timestamp"
).reset_index(drop=True)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 80)
print("DATASET COMPLETE")
print("=" * 80)
print(f"Events processed: {len(result)}")
print(f"Columns: {len(result.columns)}")
print(f"Output: {OUTPUT_FILE}")
print()

# -------------------------------------------------------------------------
# Basic coverage summary
# -------------------------------------------------------------------------

for minutes in CHECKPOINTS:

    column = f"return_{minutes}m_pct"

    if column not in result.columns:
        continue

    valid = result[column].dropna()

    if valid.empty:
        continue

    print(
        f"{minutes:>2}m | "
        f"n={len(valid):>2} | "
        f"avg={valid.mean():>8.4f}% | "
        f"median={valid.median():>8.4f}% | "
        f"UP={(valid > 0).mean() * 100:>6.2f}%"
    )

print()