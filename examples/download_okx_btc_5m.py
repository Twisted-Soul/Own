import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://www.okx.com/api/v5/market/history-candles"

OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_1y.csv")

INST_ID = "BTC-USDT"
BAR = "5m"

# Approximately 1 year of data.
DAYS = 365

# OKX allows up to 300 on this endpoint.
LIMIT = 300

REQUEST_DELAY = 0.15


def utc_ms(dt):
    return int(dt.timestamp() * 1000)


def fetch_batch(after=None):
    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": str(LIMIT),
    }

    if after is not None:
        params["after"] = str(after)

    response = requests.get(
        API_URL,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {payload.get('msg')}"
        )

    return payload.get("data", [])


def main():
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    start_ms = utc_ms(start_time)
    end_ms = utc_ms(end_time)

    print("OKX BTC 5m historical downloader")
    print("--------------------------------")
    print(f"Instrument : {INST_ID}")
    print(f"Interval   : {BAR}")
    print(f"Start      : {start_time.isoformat()}")
    print(f"End        : {end_time.isoformat()}")
    print()

    all_rows = []
    after = None
    batch_number = 0

    while True:
        batch_number += 1

        print(
            f"Request {batch_number}: "
            f"downloaded {len(all_rows):,} candles so far..."
        )

        rows = fetch_batch(after)

        if not rows:
            print("No more data returned.")
            break

        all_rows.extend(rows)

        timestamps = [int(row[0]) for row in rows]

        oldest = min(timestamps)
        newest = max(timestamps)

        print(
            f"  batch={len(rows)} "
            f"oldest={datetime.fromtimestamp(oldest / 1000, timezone.utc)} "
            f"newest={datetime.fromtimestamp(newest / 1000, timezone.utc)}"
        )

        # We have reached the requested historical range.
        if oldest <= start_ms:
            break

        # OKX's `after` requests data older than the supplied timestamp.
        next_after = oldest

        if after is not None and next_after >= after:
            raise RuntimeError(
                "Pagination stopped making progress."
            )

        after = next_after

        time.sleep(REQUEST_DELAY)

    print()
    print("Processing data...")

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_currency",
        "volume_quote",
        "confirm",
    ]

    df = pd.DataFrame(all_rows, columns=columns)

    # Convert timestamp.
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"]),
        unit="ms",
        utc=True,
    )

    # Numeric columns.
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

    # Keep only the requested time range.
    df = df[
        (df["timestamp"] >= pd.Timestamp(start_time))
        & (df["timestamp"] <= pd.Timestamp(end_time))
    ].copy()

    # Historical candles should be completed.
    df = df[df["confirm"] == "1"].copy()

    # Remove duplicate timestamps.
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    # Sort oldest -> newest.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rename timestamp to match Kronos' existing dataset.
    df = df.rename(
        columns={
            "timestamp": "timestamps",
            "volume_currency": "amount",
        }
    )

    # Keep the same core schema as the existing Kronos BTC data.
    df = df[
        [
            "timestamps",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
    ]

    # Remove invalid rows.
    df = df.dropna()

    # Check for duplicate timestamps.
    duplicate_count = df["timestamps"].duplicated().sum()

    if duplicate_count:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    # Check chronological order.
    if not df["timestamps"].is_monotonic_increasing:
        raise RuntimeError(
            "Timestamps are not sorted chronologically."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Download complete.")
    print(f"Candles : {len(df):,}")
    print(f"Start   : {df['timestamps'].iloc[0]}")
    print(f"End     : {df['timestamps'].iloc[-1]}")
    print(f"Output  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()