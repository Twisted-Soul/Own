import pandas as pd
from pathlib import Path


# Location of the downloaded BTCUSDT 5-minute files
DATA_DIR = Path("./data/btc")

files = sorted(DATA_DIR.glob("2026-05-*.parquet"))

if not files:
    raise FileNotFoundError("No BTCUSDT parquet files found.")


print(f"Found {len(files)} files:")

for file in files:
    print(f"  - {file.name}")


# Read all daily files
frames = []

for file in files:
    df = pd.read_parquet(file)
    frames.append(df)


# Combine all days
df = pd.concat(frames, ignore_index=True)


# Convert timestamp from milliseconds to UTC datetime
df["timestamps"] = pd.to_datetime(
    df["open_time_ms"],
    unit="ms",
    utc=True
)


# Kronos uses 'amount'
df["amount"] = df["quote_volume"]


# Keep only the fields Kronos needs
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


# Make sure data is chronological
df = df.sort_values("timestamps").reset_index(drop=True)


print("\nCombined dataset")
print("----------------")
print(f"Rows: {len(df)}")
print(f"First candle: {df['timestamps'].iloc[0]}")
print(f"Last candle:  {df['timestamps'].iloc[-1]}")


# Check missing values
print("\nMissing values")
print("--------------")
print(df.isna().sum())


# Check candle intervals
intervals = df["timestamps"].diff().dropna()

print("\nCandle intervals")
print("----------------")
print(intervals.value_counts().head())


# Check duplicate timestamps
duplicates = df["timestamps"].duplicated().sum()

print(f"\nDuplicate timestamps: {duplicates}")


# Save clean dataset
output_file = DATA_DIR / "BTCUSDT_5m_clean.csv"

df.to_csv(output_file, index=False)


print(f"\nSaved clean dataset to:")
print(output_file)