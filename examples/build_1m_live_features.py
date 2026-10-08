from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live_features.csv"
)


print("=" * 80)
print("KRONOS 1M LIVE FEATURES")
print("=" * 80)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("1m live data file is empty.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce",
)

df = df.dropna(
    subset=["timestamp"]
).copy()

df = df.sort_values("timestamp")

df = df.drop_duplicates(
    subset=["timestamp"],
    keep="last",
).reset_index(drop=True)


print(f"Loaded candles: {len(df)}")


# ---------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------

close = df["close"]
high = df["high"]
low = df["low"]
volume = df["volume"]


# EMA
df["ema_5"] = close.ewm(
    span=5,
    adjust=False,
).mean()

df["ema_20"] = close.ewm(
    span=20,
    adjust=False,
).mean()

df["ema_50"] = close.ewm(
    span=50,
    adjust=False,
).mean()

df["ema_200"] = close.ewm(
    span=200,
    adjust=False,
).mean()


# ---------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(
    14
).mean()

avg_loss = loss.rolling(
    14
).mean()

rs = avg_gain / avg_loss.replace(
    0,
    np.nan,
)

df["rsi_14"] = (
    100
    -
    (
        100
        /
        (1 + rs)
    )
)


# ---------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------

ema_12 = close.ewm(
    span=12,
    adjust=False,
).mean()

ema_26 = close.ewm(
    span=26,
    adjust=False,
).mean()

df["macd"] = (
    ema_12
    -
    ema_26
)

df["macd_signal"] = df["macd"].ewm(
    span=9,
    adjust=False,
).mean()

df["macd_histogram"] = (
    df["macd"]
    -
    df["macd_signal"]
)


# ---------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------

previous_close = close.shift(1)

true_range = pd.concat(
    [
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ],
    axis=1,
).max(axis=1)

df["atr_14"] = true_range.rolling(
    14
).mean()


# ---------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------

df["volume_ma_20"] = volume.rolling(
    20
).mean()

df["relative_volume"] = (
    volume
    /
    df["volume_ma_20"]
)


# ---------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------

df["return_1m"] = (
    close
    /
    close.shift(1)
    - 1
)

df["return_5m"] = (
    close
    /
    close.shift(5)
    - 1
)

df["return_15m"] = (
    close
    /
    close.shift(15)
    - 1
)

df["return_30m"] = (
    close
    /
    close.shift(30)
    - 1
)

df["return_1h"] = (
    close
    /
    close.shift(60)
    - 1
)


# ---------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------

df["volatility_5"] = (
    df["return_1m"]
    .rolling(5)
    .std()
)

df["volatility_15"] = (
    df["return_1m"]
    .rolling(15)
    .std()
)

df["volatility_30"] = (
    df["return_1m"]
    .rolling(30)
    .std()
)

df["volatility_60"] = (
    df["return_1m"]
    .rolling(60)
    .std()
)


# ---------------------------------------------------------------------
# Price ranges
# ---------------------------------------------------------------------

df["high_15"] = high.rolling(
    15
).max()

df["low_15"] = low.rolling(
    15
).min()

df["high_30"] = high.rolling(
    30
).max()

df["low_30"] = low.rolling(
    30
).min()

df["high_60"] = high.rolling(
    60
).max()

df["low_60"] = low.rolling(
    60
).min()


# ---------------------------------------------------------------------
# Candle structure
# ---------------------------------------------------------------------

df["candle_range"] = (
    high - low
)

df["candle_range_pct"] = (
    df["candle_range"]
    /
    close
)

df["candle_body"] = (
    close
    -
    df["open"]
)

df["candle_body_pct"] = (
    df["candle_body"]
    /
    df["open"]
)

df["upper_wick"] = (
    high
    -
    df[["open", "close"]].max(axis=1)
)

df["lower_wick"] = (
    df[["open", "close"]].min(axis=1)
    -
    low
)


# ---------------------------------------------------------------------
# Distance from moving averages
# ---------------------------------------------------------------------

df["distance_ema_20"] = (
    close
    /
    df["ema_20"]
    - 1
)

df["distance_ema_50"] = (
    close
    /
    df["ema_50"]
    - 1
)

df["distance_ema_200"] = (
    close
    /
    df["ema_200"]
    - 1
)


# ---------------------------------------------------------------------
# Shock / recovery features
#
# These are specifically for the future event-reaction engine.
# ---------------------------------------------------------------------

df["shock_1m"] = df["return_1m"]

df["shock_5m"] = df["return_5m"]

df["shock_15m"] = df["return_15m"]

df["range_expansion"] = (
    df["candle_range_pct"]
    /
    df["candle_range_pct"]
    .rolling(30)
    .mean()
)

df["volatility_expansion"] = (
    df["volatility_5"]
    /
    df["volatility_30"]
)

df["volume_expansion"] = (
    df["relative_volume"]
)


# ---------------------------------------------------------------------
# Recovery from recent low/high
#
# Positive recovery_from_15m_low:
# price has recovered upward from the recent low.
#
# Negative recovery_from_15m_high:
# price has fallen from the recent high.
# ---------------------------------------------------------------------

df["recovery_from_15m_low"] = (
    close
    /
    df["low_15"]
    - 1
)

df["recovery_from_30m_low"] = (
    close
    /
    df["low_30"]
    - 1
)

df["recovery_from_60m_low"] = (
    close
    /
    df["low_60"]
    - 1
)

df["distance_from_15m_high"] = (
    close
    /
    df["high_15"]
    - 1
)

df["distance_from_30m_high"] = (
    close
    /
    df["high_30"]
    - 1
)

df["distance_from_60m_high"] = (
    close
    /
    df["high_60"]
    - 1
)


# ---------------------------------------------------------------------
# Remove warmup rows
# ---------------------------------------------------------------------

feature_columns = [
    "ema_5",
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
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_1h",
    "volatility_5",
    "volatility_15",
    "volatility_30",
    "volatility_60",
    "high_15",
    "low_15",
    "high_30",
    "low_30",
    "high_60",
    "low_60",
    "candle_range",
    "candle_range_pct",
    "candle_body",
    "candle_body_pct",
    "upper_wick",
    "lower_wick",
    "distance_ema_20",
    "distance_ema_50",
    "distance_ema_200",
    "shock_1m",
    "shock_5m",
    "shock_15m",
    "range_expansion",
    "volatility_expansion",
    "volume_expansion",
    "recovery_from_15m_low",
    "recovery_from_30m_low",
    "recovery_from_60m_low",
    "distance_from_15m_high",
    "distance_from_30m_high",
    "distance_from_60m_high",
]

before = len(df)

df = df.dropna(
    subset=feature_columns
).reset_index(drop=True)

print(
    f"Removed warmup/invalid rows: "
    f"{before - len(df)}"
)

print(
    f"Final feature rows: {len(df)}"
)


# ---------------------------------------------------------------------
# MYT display column
#
# Keep timestamp UTC for machine processing.
# Add a separate Malaysia-time timestamp for human use.
# ---------------------------------------------------------------------

df["timestamp_myt"] = (
    df["timestamp"]
    .dt.tz_convert("Asia/Kuala_Lumpur")
)


# Put MYT immediately beside UTC timestamp.
columns = list(df.columns)

columns.remove("timestamp_myt")

timestamp_index = columns.index(
    "timestamp"
)

columns.insert(
    timestamp_index + 1,
    "timestamp_myt"
)

df = df[columns]


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------------------
# Latest observation
# ---------------------------------------------------------------------

latest = df.iloc[-1]

print()
print("=" * 80)
print("LATEST 1M STATE")
print("=" * 80)

print(
    f"UTC:  {latest['timestamp']}"
)

print(
    f"MYT:  {latest['timestamp_myt']}"
)

print(
    f"Close: {latest['close']:.2f}"
)

print(
    f"RSI: {latest['rsi_14']:.2f}"
)

print(
    f"EMA20: {latest['ema_20']:.2f}"
)

print(
    f"EMA50: {latest['ema_50']:.2f}"
)

print(
    f"EMA200: {latest['ema_200']:.2f}"
)

print(
    f"1m return: "
    f"{latest['return_1m'] * 100:+.4f}%"
)

print(
    f"5m return: "
    f"{latest['return_5m'] * 100:+.4f}%"
)

print(
    f"15m return: "
    f"{latest['return_15m'] * 100:+.4f}%"
)

print(
    f"1h return: "
    f"{latest['return_1h'] * 100:+.4f}%"
)

print(
    f"Relative volume: "
    f"{latest['relative_volume']:.2f}"
)

print(
    f"Volatility expansion: "
    f"{latest['volatility_expansion']:.2f}"
)

print(
    f"Recovery 15m low: "
    f"{latest['recovery_from_15m_low'] * 100:+.4f}%"
)

print(
    f"Recovery 30m low: "
    f"{latest['recovery_from_30m_low'] * 100:+.4f}%"
)

print()
print(f"Saved: {OUTPUT_FILE}")
print()
print("Analysis complete.")