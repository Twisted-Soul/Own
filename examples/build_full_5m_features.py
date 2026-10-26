from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_full_raw.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_full_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILD FULL BTCUSDT 5m FEATURE DATASET")
print("=" * 70)

print()
print("Loading raw data...")

df = pd.read_csv(
    INPUT_FILE
)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
    errors="coerce",
)

df = (
    df
    .sort_values("timestamps")
    .drop_duplicates("timestamps")
    .reset_index(drop=True)
)


print(
    f"Raw rows: {len(df):,}"
)

print(
    f"Range: "
    f"{df['timestamps'].min()} → "
    f"{df['timestamps'].max()}"
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

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


df = df.dropna(
    subset=numeric_columns
).reset_index(drop=True)


# ============================================================
# BASIC PRICE FEATURES
# ============================================================

print()
print("Calculating technical indicators...")


# ------------------------------------------------------------
# EMA
# ------------------------------------------------------------

df["ema_20"] = (
    df["close"]
    .ewm(
        span=20,
        adjust=False,
    )
    .mean()
)

df["ema_50"] = (
    df["close"]
    .ewm(
        span=50,
        adjust=False,
    )
    .mean()
)

df["ema_200"] = (
    df["close"]
    .ewm(
        span=200,
        adjust=False,
    )
    .mean()
)


# ============================================================
# RSI 14
# ============================================================

delta = df["close"].diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)

avg_gain = (
    gain
    .ewm(
        alpha=1 / 14,
        adjust=False,
    )
    .mean()
)

avg_loss = (
    loss
    .ewm(
        alpha=1 / 14,
        adjust=False,
    )
    .mean()
)

rs = avg_gain / avg_loss.replace(
    0,
    np.nan,
)

df["rsi_14"] = (
    100
    - (
        100
        / (
            1
            + rs
        )
    )
)

# Handle extreme cases.
df.loc[
    (avg_loss == 0)
    &
    (avg_gain > 0),
    "rsi_14",
] = 100

df.loc[
    (avg_gain == 0)
    &
    (avg_loss > 0),
    "rsi_14",
] = 0


# ============================================================
# MACD
# ============================================================

ema_12 = (
    df["close"]
    .ewm(
        span=12,
        adjust=False,
    )
    .mean()
)

ema_26 = (
    df["close"]
    .ewm(
        span=26,
        adjust=False,
    )
    .mean()
)

df["macd"] = (
    ema_12
    - ema_26
)

df["macd_signal"] = (
    df["macd"]
    .ewm(
        span=9,
        adjust=False,
    )
    .mean()
)

df["macd_histogram"] = (
    df["macd"]
    - df["macd_signal"]
)


# ============================================================
# ATR 14
# ============================================================

previous_close = (
    df["close"].shift(1)
)

true_range = pd.concat(
    [
        df["high"] - df["low"],

        (
            df["high"]
            - previous_close
        ).abs(),

        (
            df["low"]
            - previous_close
        ).abs(),
    ],
    axis=1,
).max(axis=1)

df["atr_14"] = (
    true_range
    .ewm(
        alpha=1 / 14,
        adjust=False,
    )
    .mean()
)


# ============================================================
# VOLUME
# ============================================================

df["volume_ma_20"] = (
    df["volume"]
    .rolling(
        window=20,
        min_periods=20,
    )
    .mean()
)

df["relative_volume"] = (
    df["volume"]
    / df["volume_ma_20"]
)


# ============================================================
# PRICE CHANNELS
# ============================================================

df["high_20"] = (
    df["high"]
    .rolling(
        window=20,
        min_periods=20,
    )
    .max()
)

df["low_20"] = (
    df["low"]
    .rolling(
        window=20,
        min_periods=20,
    )
    .min()
)

df["high_50"] = (
    df["high"]
    .rolling(
        window=50,
        min_periods=50,
    )
    .max()
)

df["low_50"] = (
    df["low"]
    .rolling(
        window=50,
        min_periods=50,
    )
    .min()
)


# ============================================================
# RETURNS
# ============================================================

# 5m
df["return_5m"] = (
    df["close"]
    / df["close"].shift(1)
    - 1
)

# 15m = 3 x 5m candles
df["return_15m"] = (
    df["close"]
    / df["close"].shift(3)
    - 1
)

# 1h = 12 x 5m candles
df["return_1h"] = (
    df["close"]
    / df["close"].shift(12)
    - 1
)


# ============================================================
# VOLATILITY
# ============================================================

df["volatility_20"] = (
    df["return_5m"]
    .rolling(
        window=20,
        min_periods=20,
    )
    .std()
)


# ============================================================
# EMA DISTANCES
# ============================================================

df["distance_ema_20"] = (
    df["close"]
    / df["ema_20"]
    - 1
)

df["distance_ema_50"] = (
    df["close"]
    / df["ema_50"]
    - 1
)

df["distance_ema_200"] = (
    df["close"]
    / df["ema_200"]
    - 1
)


# ============================================================
# CANDLE STRUCTURE
# ============================================================

df["candle_range"] = (
    df["high"]
    - df["low"]
)

df["candle_body"] = (
    df["close"]
    - df["open"]
)

df["candle_body_pct"] = (
    df["candle_body"]
    / df["open"]
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

columns = [
    "timestamps",

    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",

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


df = df[columns]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("FEATURE DATASET VALIDATION")
print("=" * 70)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Start: {df['timestamps'].min()}"
)

print(
    f"End:   {df['timestamps'].max()}"
)

print()
print("NaN counts:")

nan_counts = df.isna().sum()

print(
    nan_counts[
        nan_counts > 0
    ]
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)