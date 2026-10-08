import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "btc" / "BTCUSDT_5m_clean.csv"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "btc" / "BTCUSDT_5m_features.csv"


# ============================================================
# Technical indicator functions
# ============================================================

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    previous_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_previous = (df["high"] - previous_close).abs()
    low_previous = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [high_low, high_previous, low_previous],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("BTC/USDT Technical Feature Engine")
print("=" * 60)

print(f"\nLoading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Loaded {len(df):,} candles")


# ============================================================
# Moving averages
# ============================================================

print("\nCalculating moving averages...")

df["ema_20"] = df["close"].ewm(
    span=20,
    adjust=False
).mean()

df["ema_50"] = df["close"].ewm(
    span=50,
    adjust=False
).mean()

df["ema_200"] = df["close"].ewm(
    span=200,
    adjust=False
).mean()


# ============================================================
# RSI
# ============================================================

print("Calculating RSI...")

df["rsi_14"] = calculate_rsi(df["close"], 14)


# ============================================================
# MACD
# ============================================================

print("Calculating MACD...")

ema_12 = df["close"].ewm(
    span=12,
    adjust=False
).mean()

ema_26 = df["close"].ewm(
    span=26,
    adjust=False
).mean()

df["macd"] = ema_12 - ema_26

df["macd_signal"] = df["macd"].ewm(
    span=9,
    adjust=False
).mean()

df["macd_histogram"] = (
    df["macd"] - df["macd_signal"]
)


# ============================================================
# ATR
# ============================================================

print("Calculating ATR...")

df["atr_14"] = calculate_atr(df, 14)


# ============================================================
# Volume features
# ============================================================

print("Calculating volume features...")

df["volume_ma_20"] = df["volume"].rolling(20).mean()

df["relative_volume"] = (
    df["volume"] / df["volume_ma_20"]
)


# ============================================================
# Recent market structure
# ============================================================

print("Calculating recent highs/lows...")

# Shift by one candle so the current candle itself
# cannot leak into these historical reference levels.
df["high_20"] = (
    df["high"]
    .shift(1)
    .rolling(20)
    .max()
)

df["low_20"] = (
    df["low"]
    .shift(1)
    .rolling(20)
    .min()
)

df["high_50"] = (
    df["high"]
    .shift(1)
    .rolling(50)
    .max()
)

df["low_50"] = (
    df["low"]
    .shift(1)
    .rolling(50)
    .min()
)


# ============================================================
# Price returns
# ============================================================

print("Calculating returns...")

# 5 minutes
df["return_5m"] = df["close"].pct_change(1)

# 15 minutes
df["return_15m"] = df["close"].pct_change(3)

# 1 hour
df["return_1h"] = df["close"].pct_change(12)


# ============================================================
# Rolling volatility
# ============================================================

print("Calculating volatility...")

df["volatility_20"] = (
    df["return_5m"]
    .rolling(20)
    .std()
)


# ============================================================
# Distance from EMAs
# ============================================================

print("Calculating EMA distances...")

df["distance_ema_20"] = (
    df["close"] - df["ema_20"]
) / df["ema_20"]

df["distance_ema_50"] = (
    df["close"] - df["ema_50"]
) / df["ema_50"]

df["distance_ema_200"] = (
    df["close"] - df["ema_200"]
) / df["ema_200"]


# ============================================================
# Basic candle information
# ============================================================

df["candle_range"] = (
    df["high"] - df["low"]
)

df["candle_body"] = (
    df["close"] - df["open"]
)

df["candle_body_pct"] = (
    df["candle_body"] / df["open"]
)


# ============================================================
# Remove rows where indicators aren't available yet
# ============================================================

feature_columns = [
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_14",
    "volume_ma_20",
    "relative_volume",
    "high_20",
    "low_20",
    "high_50",
    "low_50",
    "return_5m",
    "return_15m",
    "return_1h",
    "volatility_20",
    "distance_ema_20",
    "distance_ema_50",
    "distance_ema_200",
    "candle_range",
    "candle_body",
    "candle_body_pct",
]

before = len(df)

df = df.dropna(
    subset=feature_columns
).reset_index(drop=True)

removed = before - len(df)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINE COMPLETE")
print("=" * 60)

print(f"\nOriginal candles : {before:,}")
print(f"Removed rows     : {removed:,}")
print(f"Final candles    : {len(df):,}")

print(f"\nOutput:")
print(OUTPUT_FILE)

print("\nLatest feature snapshot:")

display_columns = [
    "timestamps",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
    "relative_volume",
    "return_5m",
    "return_15m",
    "return_1h",
    "volatility_20",
]

print(
    df[display_columns]
    .tail(10)
    .to_string(index=False)
)

print("\nFeature columns:")

for column in feature_columns:
    print(f"  - {column}")

print("\nDone.")