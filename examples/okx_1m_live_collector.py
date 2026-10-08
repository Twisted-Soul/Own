from pathlib import Path
import time
import requests
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live.csv"
)

API_URL = "https://www.okx.com/api/v5/market/candles"

INST_ID = "BTC-USDT"
BAR = "1m"

POLL_SECONDS = 15
MAX_ROWS = 10000


# ---------------------------------------------------------------------
# Fetch latest candles
# ---------------------------------------------------------------------

def fetch_candles(limit=100):

    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": str(limit),
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {payload}"
        )

    return payload["data"]


# ---------------------------------------------------------------------
# Convert OKX response
# ---------------------------------------------------------------------

def candles_to_dataframe(data):

    rows = []

    for row in data:

        if len(row) < 7:
            continue

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "amount": float(row[6]),
                "confirm": row[8] if len(row) > 8 else None,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.sort_values("timestamp")

    return df


# ---------------------------------------------------------------------
# Load existing file
# ---------------------------------------------------------------------

def load_existing():

    if not OUTPUT_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "confirm",
            ]
        )

    df = pd.read_csv(OUTPUT_FILE)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df


# ---------------------------------------------------------------------
# Keep completed candles only
# ---------------------------------------------------------------------

def completed_only(df):

    if df.empty:
        return df

    now = pd.Timestamp.now(tz="UTC")

    # A 1-minute candle is complete once its timestamp is older
    # than the current minute.
    current_minute = now.floor("min")

    df = df[
        df["timestamp"] < current_minute
    ].copy()

    return df


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save(df):

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .tail(MAX_ROWS)
        .copy()
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

print("=" * 80)
print("KRONOS 1M LIVE BTC COLLECTOR")
print("=" * 80)

print(f"Instrument: {INST_ID}")
print(f"Interval:   {BAR}")
print(f"Output:     {OUTPUT_FILE}")
print()

existing = load_existing()

print(
    f"Existing candles: {len(existing)}"
)

# Initial fetch
try:

    raw = fetch_candles(100)

    latest = candles_to_dataframe(raw)

    latest = completed_only(latest)

    combined = pd.concat(
        [existing, latest],
        ignore_index=True,
    )

    combined = save(combined)

    print(
        f"Initial update: {len(combined)} candles"
    )

except Exception as exc:

    print(
        f"Initial fetch failed: {exc}"
    )


last_timestamp = None

if not existing.empty:

    last_timestamp = existing["timestamp"].max()


print()
print("Collector running...")
print("Press Ctrl+C to stop.")
print()


while True:

    try:

        raw = fetch_candles(10)

        latest = candles_to_dataframe(raw)

        latest = completed_only(latest)

        if latest.empty:

            time.sleep(POLL_SECONDS)
            continue

        existing = load_existing()

        previous_count = len(existing)

        combined = pd.concat(
            [existing, latest],
            ignore_index=True,
        )

        combined = save(combined)

        new_rows = combined[
            combined["timestamp"]
            > (
                last_timestamp
                if last_timestamp is not None
                else pd.Timestamp.min.tz_localize("UTC")
            )
        ]

        if not new_rows.empty:

            for _, row in new_rows.iterrows():

                print(
                    f"{row['timestamp']} | "
                    f"NEW 1m candle | "
                    f"O={row['open']:.2f} "
                    f"H={row['high']:.2f} "
                    f"L={row['low']:.2f} "
                    f"C={row['close']:.2f} "
                    f"V={row['volume']:.4f}"
                )

            last_timestamp = new_rows[
                "timestamp"
            ].max()

        elif len(combined) != previous_count:

            print(
                f"Updated dataset: "
                f"{len(combined)} candles"
            )

        time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:

        print()
        print("Collector stopped.")
        break

    except Exception as exc:

        print(
            f"Collector error: {exc}"
        )

        time.sleep(POLL_SECONDS)