from pathlib import Path

import pandas as pd
import requests
import time


# ============================================================
# CONFIG
# ============================================================

API_URL = (
    "https://www.okx.com/api/v5/market/history-candles"
)

INST_ID = "BTC-USDT"
BAR = "5m"

TOTAL_CANDLES = 1000
BATCH_SIZE = 100

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_live.csv"
)


# ============================================================
# FETCH HISTORICAL BATCH
# ============================================================

def fetch_batch(after=None):

    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": str(BATCH_SIZE),
    }

    if after is not None:
        params["after"] = str(after)

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

    rows = payload.get("data", [])

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

    return records


# ============================================================
# FETCH HISTORY
# ============================================================

def fetch_history():

    all_records = []

    after = None

    while len(all_records) < TOTAL_CANDLES:

        batch = fetch_batch(
            after=after
        )

        if not batch:
            print(
                "No more historical data returned."
            )
            break

        all_records.extend(
            batch
        )

        oldest_timestamp = min(
            record["timestamps"]
            for record in batch
        )

        after = int(
            oldest_timestamp.timestamp()
            * 1000
        )

        print(
            f"Fetched "
            f"{len(all_records):,} candles "
            f"(oldest: {oldest_timestamp})"
        )

        if len(batch) < BATCH_SIZE:
            break

        # Avoid hammering the API.
        time.sleep(0.2)

    return pd.DataFrame(
        all_records
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KRONOS OKX HISTORICAL BOOTSTRAP")
    print("=" * 70)

    print(
        f"Instrument: {INST_ID}"
    )

    print(
        f"Timeframe: {BAR}"
    )

    print(
        f"Target candles: "
        f"{TOTAL_CANDLES:,}"
    )

    print()

    df = fetch_history()

    if df.empty:
        raise RuntimeError(
            "No data returned from OKX."
        )

    # --------------------------------------------------------
    # Completed candles only
    # --------------------------------------------------------

    df = df[
        df["confirm"] == 1
    ].copy()

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    df = (
        df.drop_duplicates(
            subset=["timestamps"],
            keep="last",
        )
        .sort_values("timestamps")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Keep requested amount
    # --------------------------------------------------------

    if len(df) > TOTAL_CANDLES:

        df = df.iloc[
            -TOTAL_CANDLES:
        ].copy()

        df = df.reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("BOOTSTRAP COMPLETE")
    print("=" * 70)

    print(
        f"Saved candles: "
        f"{len(df):,}"
    )

    print(
        f"Start: "
        f"{df['timestamps'].iloc[0]}"
    )

    print(
        f"End:   "
        f"{df['timestamps'].iloc[-1]}"
    )

    print(
        f"Latest close: "
        f"{df['close'].iloc[-1]:.2f}"
    )

    print()
    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()