import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/btc/5m_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

MONTHS = [
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
]


# Binance kline columns
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def convert_binance_timestamp(series):
    """
    Automatically detect Binance timestamp unit.

    Typical values:

    milliseconds:
        ~1.7e12

    microseconds:
        ~1.7e15
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        raise ValueError("No valid timestamps found.")

    median_value = valid.median()

    print(f"Timestamp sample : {median_value:.0f}")

    if median_value >= 1e14:
        unit = "us"
    elif median_value >= 1e11:
        unit = "ms"
    elif median_value >= 1e9:
        unit = "s"
    else:
        raise ValueError(
            f"Unknown Binance timestamp format: {median_value}"
        )

    print(f"Detected timestamp unit: {unit}")

    return pd.to_datetime(
        numeric,
        unit=unit,
        utc=True,
    )


def download_month(month):
    filename = f"{SYMBOL}-{INTERVAL}-{month}.zip"

    url = (
        "https://data.binance.vision/data/spot/monthly/"
        f"klines/{SYMBOL}/{INTERVAL}/{filename}"
    )

    print()
    print("=" * 70)
    print(f"Downloading: {filename}")
    print(url)

    try:
        response = requests.get(
            url,
            timeout=60,
        )

        response.raise_for_status()

    except Exception as e:
        print(f"DOWNLOAD FAILED: {e}")
        return None

    print(
        f"Downloaded: "
        f"{len(response.content) / 1024 / 1024:.2f} MB"
    )

    try:
        with zipfile.ZipFile(
            io.BytesIO(response.content)
        ) as z:

            names = z.namelist()

            csv_names = [
                name
                for name in names
                if name.lower().endswith(".csv")
            ]

            if not csv_names:
                print("ERROR: No CSV found inside ZIP.")
                return None

            csv_name = csv_names[0]

            print(f"CSV inside ZIP: {csv_name}")

            with z.open(csv_name) as f:

                df = pd.read_csv(
                    f,
                    header=None,
                    names=KLINE_COLUMNS,
                )

    except Exception as e:
        print(f"ZIP/CSV ERROR: {e}")
        return None

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    try:
        df["timestamp"] = convert_binance_timestamp(
            df["open_time"]
        )

    except Exception as e:
        print(f"TIMESTAMP ERROR: {e}")
        return None

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
    ].copy()

    # Match existing feature dataset naming.
    df = df.rename(
        columns={
            "quote_volume": "amount",
        }
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove invalid rows.
    df = df.dropna()

    # Sort and remove duplicates.
    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if df.empty:
        print("ERROR: No valid rows remain.")
        return None

    print(
        f"Range: "
        f"{df['timestamp'].iloc[0]} → "
        f"{df['timestamp'].iloc[-1]}"
    )

    # Make sure the month actually looks correct.
    expected_prefix = month

    actual_prefix = (
        df["timestamp"]
        .iloc[0]
        .strftime("%Y-%m")
    )

    if actual_prefix != expected_prefix:
        print(
            "WARNING: Timestamp does not match requested month!"
        )
        print(
            f"Expected: {expected_prefix}"
        )
        print(
            f"Actual:   {actual_prefix}"
        )

    # --------------------------------------------------------
    # Save monthly file
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"{SYMBOL}_5m_{month}.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved: {output_file}")

    return df


def main():

    print("=" * 70)
    print("BTCUSDT 5m Historical Data Downloader")
    print("=" * 70)

    all_data = []

    for month in MONTHS:

        output_file = (
            OUTPUT_DIR
            / f"{SYMBOL}_5m_{month}.csv"
        )

        # ----------------------------------------------------
        # Existing file
        # ----------------------------------------------------

        if output_file.exists():

            print()
            print(
                f"Already exists, skipping: "
                f"{output_file}"
            )

            try:
                df = pd.read_csv(
                    output_file
                )

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"],
                    utc=True,
                )

                all_data.append(df)

            except Exception as e:
                print(
                    f"Could not read existing file: {e}"
                )

            continue

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        df = download_month(month)

        if df is not None:
            all_data.append(df)

    # --------------------------------------------------------
    # Combine all months
    # --------------------------------------------------------

    if not all_data:

        print()
        print("No data downloaded.")
        return

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    combined_file = (
        OUTPUT_DIR
        / "BTCUSDT_5m_missing_2025-02_to_2025-09.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Rows:  {len(combined):,}"
    )

    print(
        f"Start: {combined['timestamp'].iloc[0]}"
    )

    print(
        f"End:   {combined['timestamp'].iloc[-1]}"
    )

    print(
        f"Output: {combined_file}"
    )

    print()
    print("Columns:")
    print(
        list(combined.columns)
    )

    print()
    print("Missing values:")
    print(
        combined.isna().sum()
    )


if __name__ == "__main__":
    main()