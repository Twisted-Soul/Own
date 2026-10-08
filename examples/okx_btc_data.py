import requests
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

API_URL = "https://www.okx.com/api/v5/market/candles"

INST_ID = "BTC-USDT"
BAR = "5m"
LIMIT = 100


# ============================================================
# FETCH DATA
# ============================================================

params = {
    "instId": INST_ID,
    "bar": BAR,
    "limit": LIMIT
}

print("Fetching BTC-USDT data from OKX...")

response = requests.get(
    API_URL,
    params=params,
    timeout=10
)

response.raise_for_status()

data = response.json()


# ============================================================
# CHECK API RESPONSE
# ============================================================

if data.get("code") != "0":
    raise RuntimeError(
        f"OKX API error: {data}"
    )

rows = data["data"]

print(f"Received {len(rows)} candles.")


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

columns = [
    "timestamp_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "quote_volume_currency",
    "confirmed"
]

df = pd.DataFrame(
    rows,
    columns=columns
)


# ============================================================
# CLEAN DATA
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp_ms"].astype("int64"),
    unit="ms",
    utc=True
)

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "quote_volume_currency"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["confirmed"] = df["confirmed"].astype(int)


# ============================================================
# KEEP ONLY COMPLETED CANDLES
# ============================================================

df = df[
    df["confirmed"] == 1
].copy()


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# FORMAT FOR KRONOS
# ============================================================

kronos_df = df[
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]
].copy()

kronos_df = kronos_df.rename(
    columns={
        "quote_volume": "amount"
    }
)


# ============================================================
# DISPLAY
# ============================================================

print("\nLatest completed candles:")
print(
    kronos_df.tail(10).to_string(
        index=False
    )
)

print("\nData shape:")
print(kronos_df.shape)

print("\nLatest completed candle:")
print(kronos_df.iloc[-1]["timestamp"])

print("\nLatest BTC price:")
print(kronos_df.iloc[-1]["close"])