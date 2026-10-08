import time
from pathlib import Path

import pandas as pd
import requests


# =============================================================================
# KRONOS 1M ECONOMIC EVENT HISTORY COLLECTOR
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RELEASE_FILE = BASE_DIR / "data" / "bls_economic_releases.csv"
OUTPUT_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_history.csv"

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"

INST_ID = "BTC-USDT"
BAR = "1m"

BEFORE_MINUTES = 90
AFTER_MINUTES = 180

BATCH_SIZE = 100
REQUEST_TIMEOUT = 20
REQUEST_DELAY = 0.25


# =============================================================================
# OKX API
# =============================================================================

def fetch_okx_history(after_ms=None, before_ms=None, limit=BATCH_SIZE):

    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": str(limit),
    }

    if after_ms is not None:
        params["after"] = str(int(after_ms))

    if before_ms is not None:
        params["before"] = str(int(before_ms))

    response = requests.get(
        OKX_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {payload}"
        )

    return payload.get("data", [])


# =============================================================================
# CANDLE PARSER
# =============================================================================

def candles_to_dataframe(rows):

    if not rows:
        return pd.DataFrame()

    # OKX market candles currently return:
    #
    # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    #
    # That is 9 fields.
    #
    # We only need the first 6 for the event study, plus the remaining
    # volume fields for completeness.

    parsed_rows = []

    for row in rows:

        if len(row) < 6:
            continue

        parsed_rows.append(
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "volume_currency": row[6] if len(row) > 6 else None,
                "volume_quote": row[7] if len(row) > 7 else None,
                "confirm": row[8] if len(row) > 8 else None,
            }
        )

    if not parsed_rows:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_rows)

    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(
            df["timestamp"],
            errors="coerce",
        ),
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_currency",
        "volume_quote",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["confirm"] = pd.to_numeric(
        df["confirm"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    return (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# =============================================================================
# FETCH ONE EVENT WINDOW
# =============================================================================

def fetch_event_window(event_time):

    start_time = event_time - pd.Timedelta(
        value=int(BEFORE_MINUTES),
        unit="m",
    )

    end_time = event_time + pd.Timedelta(
        value=int(AFTER_MINUTES),
        unit="m",
    )

    start_ms = int(
        start_time.timestamp() * 1000
    )

    end_ms = int(
        end_time.timestamp() * 1000
    )

    all_pages = []

    # OKX history-candles pagination:
    #
    # after  = older boundary
    # before = newer boundary
    #
    # We start at the newest end and move backwards.

    cursor_after = end_ms

    safety_counter = 0

    while cursor_after > start_ms:

        safety_counter += 1

        if safety_counter > 10:
            break

        rows = fetch_okx_history(
            after_ms=cursor_after,
            before_ms=start_ms,
            limit=BATCH_SIZE,
        )

        if not rows:
            break

        page = candles_to_dataframe(rows)

        if page.empty:
            break

        all_pages.append(page)

        oldest_timestamp = page["timestamp"].min()

        oldest_ms = int(
            oldest_timestamp.timestamp() * 1000
        )

        new_cursor = oldest_ms

        if new_cursor >= cursor_after:
            break

        cursor_after = new_cursor

        if oldest_ms <= start_ms:
            break

        time.sleep(REQUEST_DELAY)

    if not all_pages:
        return pd.DataFrame()

    df = pd.concat(
        all_pages,
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df[
        (df["timestamp"] >= start_time)
        & (df["timestamp"] <= end_time)
    ]

    return (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# =============================================================================
# LOAD RELEASE DATA
# =============================================================================

print("=" * 80)
print("KRONOS 1M ECONOMIC EVENT HISTORY COLLECTOR")
print("=" * 80)

if not RELEASE_FILE.exists():
    raise FileNotFoundError(
        f"Release file not found:\n{RELEASE_FILE}"
    )

releases = pd.read_csv(
    RELEASE_FILE
)

required_columns = {
    "event",
    "release_time",
    "importance",
}

missing = (
    required_columns
    - set(releases.columns)
)

if missing:
    raise ValueError(
        f"Missing required release columns: "
        f"{sorted(missing)}\n"
        f"Available columns: "
        f"{list(releases.columns)}"
    )


# =============================================================================
# NORMALIZE RELEASE TIMES
# =============================================================================

releases["release_time"] = pd.to_datetime(
    releases["release_time"],
    utc=True,
    errors="coerce",
)

releases = releases.dropna(
    subset=["release_time"]
).copy()

releases = (
    releases
    .sort_values("release_time")
    .drop_duplicates(
        subset=["release_time"],
        keep="first",
    )
    .reset_index(drop=True)
)


print(
    f"Unique release windows: "
    f"{len(releases)}"
)

print()


# =============================================================================
# COLLECT
# =============================================================================

all_event_data = []

for index, row in releases.iterrows():

    event_time = row["release_time"]
    event_name = str(row["event"])
    importance = str(row["importance"])

    print(
        f"[{index + 1}/{len(releases)}] "
        f"{event_time} | "
        f"{event_name} | "
        f"importance={importance}"
    )

    try:

        df = fetch_event_window(
            event_time
        )

        if df.empty:

            print(
                "  No candles returned."
            )

            print()

            continue

        # Event metadata
        df["event_timestamp"] = event_time
        df["event_type"] = event_name
        df["importance"] = importance

        # Minutes relative to economic release
        df["minutes_from_release"] = (
            (
                df["timestamp"]
                - event_time
            )
            .dt.total_seconds()
            / 60.0
        )

        # Malaysia Time
        df["timestamp_myt"] = (
            df["timestamp"]
            .dt.tz_convert(
                "Asia/Kuala_Lumpur"
            )
        )

        df["event_timestamp_myt"] = (
            event_time
            .tz_convert(
                "Asia/Kuala_Lumpur"
            )
        )

        all_event_data.append(df)

        print(
            f"  Collected {len(df)} candles"
        )

        print(
            f"  Window: "
            f"{df['timestamp'].min()} "
            f"-> "
            f"{df['timestamp'].max()}"
        )

    except Exception as exc:

        print(
            f"  ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

    print()

    time.sleep(REQUEST_DELAY)


# =============================================================================
# SAVE
# =============================================================================

if not all_event_data:

    raise RuntimeError(
        "No event data was collected.\n"
        "The OKX historical-candle request returned "
        "no usable candles."
    )


result = pd.concat(
    all_event_data,
    ignore_index=True,
)

result = result.drop_duplicates(
    subset=[
        "event_timestamp",
        "timestamp",
    ]
)

result = (
    result
    .sort_values(
        [
            "event_timestamp",
            "timestamp",
        ]
    )
    .reset_index(drop=True)
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("COLLECTION COMPLETE")
print("=" * 80)

print(
    f"Events with data: "
    f"{result['event_timestamp'].nunique()}"
)

print(
    f"Total candles: "
    f"{len(result)}"
)

print(
    f"Output: "
    f"{OUTPUT_FILE}"
)

print()

coverage = (
    result
    .groupby("event_timestamp")
    .agg(
        candles=("timestamp", "count"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
)

print("Coverage summary:")
print(
    coverage.to_string()
)

print()

print("=" * 80)
print("DONE")
print("=" * 80)