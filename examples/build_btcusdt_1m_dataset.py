
from pathlib import Path
import pandas as pd

SOURCE = Path("data/btc/smilyface_1m/raw/klines_1m/BTCUSDT")
OUTPUT = Path("data/btc/BTCUSDT_1m_continuous_2020_2026.parquet")

files = sorted(SOURCE.glob("*.parquet"))
if not files:
    raise SystemExit(f"No source files found: {SOURCE.resolve()}")

frames = []
expected = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "trade_count",
    "taker_buy_volume", "taker_buy_quote_volume",
]

print(f"Reading {len(files):,} daily files...")

for i, file in enumerate(files, 1):
    df = pd.read_parquet(file)

    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"{file.name} missing columns: {missing}")

    df = df[expected].copy()
    frames.append(df)

    if i % 200 == 0 or i == len(files):
        print(f"Loaded {i:,}/{len(files):,} files")

data = pd.concat(frames, ignore_index=True)
del frames

data["open_time_ms"] = pd.to_numeric(
    data["open_time_ms"], errors="raise"
).astype("int64")

data = data.sort_values("open_time_ms").reset_index(drop=True)

if data["open_time_ms"].duplicated().any():
    raise ValueError("Duplicate candle opening timestamps found.")

if len(data) > 1:
    differences = data["open_time_ms"].diff().iloc[1:]
    if not differences.eq(60_000).all():
        raise ValueError("Unexpected timestamp interval detected.")

numeric_cols = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trade_count",
    "taker_buy_volume", "taker_buy_quote_volume",
]
data[numeric_cols] = data[numeric_cols].apply(
    pd.to_numeric, errors="raise"
)

data["timestamp_utc"] = pd.to_datetime(
    data["open_time_ms"], unit="ms", utc=True
)
data["close_time_utc"] = pd.to_datetime(
    data["close_time_ms"], unit="ms", utc=True
)

# Keep timestamps and original exchange fields for traceability.
data = data[
    [
        "timestamp_utc", "close_time_utc", "open_time_ms",
        "open", "high", "low", "close", "volume",
        "quote_volume", "trade_count",
        "taker_buy_volume", "taker_buy_quote_volume",
        "close_time_ms",
    ]
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
data.to_parquet(OUTPUT, index=False, compression="snappy")

print("\nDataset built successfully.")
print(f"Rows: {len(data):,}")
print(f"First candle: {data['timestamp_utc'].iloc[0]}")
print(f"Last candle:  {data['timestamp_utc'].iloc[-1]}")
print(f"Output: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / (1024**2):.1f} MiB")