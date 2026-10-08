from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = (
    "https://www.okx.com/api/v5/market/candles"
)

INST_ID = "BTC-USDT"
BAR = "5m"

LIMIT = 100

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_live.csv"
)

POLL_SECONDS = 30

# Keep a reasonable rolling local history.
MAX_CANDLES = 5000


# ============================================================
# FETCH OKX DATA
# ============================================================

def fetch_candles():

    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": str(LIMIT),
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

    rows = payload.get(
        "data",
        [],
    )

    if not rows:

        raise RuntimeError(
            "OKX returned no candle data."
        )

    records = []

    for row in rows:

        records.append(
            {
                "timestamps": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "amount": float(row[7]),
                "confirm": int(row[8]),
            }
        )

    df = pd.DataFrame(
        records
    )

    df = (
        df.sort_values(
            "timestamps"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD EXISTING DATA
# ============================================================

def load_existing():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame(
            columns=[
                "timestamps",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "confirm",
            ]
        )

    df = pd.read_csv(
        OUTPUT_FILE
    )

    df["timestamps"] = pd.to_datetime(
        df["timestamps"],
        utc=True,
    )

    return df


# ============================================================
# MERGE NEW DATA
# ============================================================

def merge_data(
    existing,
    incoming,
):

    # --------------------------------------------------------
    # If no existing data, use incoming directly.
    # This avoids pandas concatenation warnings.
    # --------------------------------------------------------

    if existing.empty:

        combined = incoming.copy()

    else:

        combined = pd.concat(
            [
                existing,
                incoming,
            ],
            ignore_index=True,
        )

    # --------------------------------------------------------
    # Deduplicate timestamps.
    #
    # Incoming data is kept because it may contain a
    # more recent version of the candle.
    # --------------------------------------------------------

    combined = (
        combined
        .drop_duplicates(
            subset=["timestamps"],
            keep="last",
        )
        .sort_values(
            "timestamps"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Keep only the latest MAX_CANDLES.
    # --------------------------------------------------------

    if len(combined) > MAX_CANDLES:

        combined = (
            combined
            .iloc[-MAX_CANDLES:]
            .reset_index(drop=True)
        )

    return combined


# ============================================================
# SAVE
# ============================================================

def save_data(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KRONOS OKX LIVE 5-MINUTE COLLECTOR")
    print("=" * 70)

    print(
        f"Instrument: {INST_ID}"
    )

    print(
        f"Timeframe: {BAR}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Polling every {POLL_SECONDS} seconds"
    )

    print(
        f"Maximum stored candles: "
        f"{MAX_CANDLES:,}"
    )

    print()

    existing = load_existing()

    print(
        f"Existing candles: "
        f"{len(existing):,}"
    )

    if not existing.empty:

        print(
            f"Existing range: "
            f"{existing['timestamps'].iloc[0]} "
            f"→ "
            f"{existing['timestamps'].iloc[-1]}"
        )

    print()

    while True:

        try:

            incoming = fetch_candles()

            # ------------------------------------------------
            # Completed candles only
            # ------------------------------------------------

            completed = incoming[
                incoming["confirm"] == 1
            ].copy()

            if completed.empty:

                print(
                    "No completed candles returned."
                )

            else:

                latest_before = (
                    existing["timestamps"].max()
                    if not existing.empty
                    else None
                )

                existing = merge_data(
                    existing,
                    completed,
                )

                latest_after = (
                    existing["timestamps"].max()
                )

                # ------------------------------------------------
                # Report whether new candles arrived.
                # ------------------------------------------------

                if (
                    latest_before is None
                    or latest_after > latest_before
                ):

                    new_count = (
                        existing["timestamps"]
                        .gt(latest_before)
                        .sum()
                        if latest_before is not None
                        else len(completed)
                    )

                    latest = (
                        existing.iloc[-1]
                    )

                    print(
                        f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC] "
                        f"NEW candle(s): {new_count}"
                    )

                    print(
                        f"Latest completed: "
                        f"{latest['timestamps']}"
                    )

                    print(
                        f"Close: "
                        f"{latest['close']:.2f}"
                    )

                    print(
                        f"Stored candles: "
                        f"{len(existing):,}"
                    )

                    save_data(
                        existing
                    )

                else:

                    print(
                        f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC] "
                        f"No new completed candle."
                    )

                    print(
                        f"Latest stored: "
                        f"{latest_after}"
                    )

            print(
                f"Sleeping {POLL_SECONDS}s..."
            )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print(
                "Collector stopped by user."
            )

            break

        except Exception as exc:

            print()
            print(
                f"ERROR: {exc}"
            )

            print(
                "Retrying..."
            )

            time.sleep(
                POLL_SECONDS
            )


if __name__ == "__main__":
    main()