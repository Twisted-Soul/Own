import requests
import pandas as pd


# Bybit public market-data endpoint
URL = "https://api.bybit.com/v5/market/kline"

PARAMS = {
    "category": "linear",
    "symbol": "BTCUSDT",
    "interval": "5",
    "limit": 400
}


def fetch_btc_data():
    response = requests.get(URL, params=PARAMS, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data["retCode"] != 0:
        raise RuntimeError(f"Bybit API error: {data}")

    rows = data["result"]["list"]

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Bybit returns newest candles first.
    # Kronos expects chronological order.
    df = df.iloc[::-1].reset_index(drop=True)

    # Convert numerical columns
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Convert timestamp to UTC
    df["timestamps"] = pd.to_datetime(
        pd.to_numeric(df["open_time"]),
        unit="ms",
        utc=True
    )

    # Kronos-compatible amount
    df["amount"] = df["quote_volume"]

    # Keep only what Kronos needs
    df = df[
        [
            "timestamps",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount"
        ]
    ]

    return df


if __name__ == "__main__":
    df = fetch_btc_data()

    print("\nBTC/USDT data successfully retrieved!")
    print(f"Rows: {len(df)}")
    print("Timeframe: 5 minutes")
    print(f"First candle: {df['timestamps'].iloc[0]}")
    print(f"Last candle:  {df['timestamps'].iloc[-1]}")

    print("\nLatest 5 candles:")
    print(df.tail().to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nData types:")
    print(df.dtypes)