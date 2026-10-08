from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_live.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_live_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading live BTC data...")

df = pd.read_csv(
    INPUT_FILE
)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
)

df = (
    df.sort_values("timestamps")
    .drop_duplicates(
        subset=["timestamps"],
        keep="last",
    )
    .reset_index(drop=True)
)

print(
    f"Loaded {len(df):,} candles."
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "timestamps",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )


# ============================================================
# TECHNICAL FEATURES
# ============================================================

print(
    "Calculating technical features..."
)


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


# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------

delta = (
    df["close"]
    .diff()
)

gain = (
    delta.clip(lower=0)
)

loss = (
    -delta.clip(upper=0)
)

avg_gain = (
    gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    )
    .mean()
)

avg_loss = (
    loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    )
    .mean()
)

rs = (
    avg_gain
    / avg_loss.replace(
        0,
        np.nan,
    )
)

df["rsi_14"] = (
    100
    - (
        100
        / (1 + rs)
    )
)


# ------------------------------------------------------------
# MACD
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------

previous_close = (
    df["close"].shift(1)
)

true_range = pd.concat(
    [
        df["high"]
        - df["low"],

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
    .rolling(14)
    .mean()
)


# ------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------

df["volume_ma_20"] = (
    df["volume"]
    .rolling(20)
    .mean()
)

df["relative_volume"] = (
    df["volume"]
    / df["volume_ma_20"]
)


# ------------------------------------------------------------
# PRICE CHANNELS
# ------------------------------------------------------------

df["high_20"] = (
    df["high"]
    .rolling(20)
    .max()
)

df["low_20"] = (
    df["low"]
    .rolling(20)
    .min()
)

df["high_50"] = (
    df["high"]
    .rolling(50)
    .max()
)

df["low_50"] = (
    df["low"]
    .rolling(50)
    .min()
)


# ------------------------------------------------------------
# RETURNS
# ------------------------------------------------------------

df["return_5m"] = (
    df["close"]
    / df["close"].shift(1)
    - 1
)

df["return_15m"] = (
    df["close"]
    / df["close"].shift(3)
    - 1
)

df["return_1h"] = (
    df["close"]
    / df["close"].shift(12)
    - 1
)


# ------------------------------------------------------------
# VOLATILITY
# ------------------------------------------------------------

df["volatility_20"] = (
    df["return_5m"]
    .rolling(20)
    .std()
)


# ------------------------------------------------------------
# DISTANCE FROM EMAs
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# CANDLE STRUCTURE
# ------------------------------------------------------------

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
# REMOVE WARMUP ROWS
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

removed = (
    before - len(df)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("LIVE FEATURE BUILD COMPLETE")
print("=" * 70)

print(
    f"Input candles:    {before:,}"
)

print(
    f"Warmup removed:   {removed:,}"
)

print(
    f"Final candles:    {len(df):,}"
)

print(
    f"Output:           {OUTPUT_FILE}"
)

if not df.empty:

    latest = df.iloc[-1]

    print()
    print(
        f"Latest timestamp: "
        f"{latest['timestamps']}"
    )

    print(
        f"Close:            "
        f"{latest['close']:.2f}"
    )

    print(
        f"RSI(14):           "
        f"{latest['rsi_14']:.2f}"
    )

    print(
        f"EMA20:             "
        f"{latest['ema_20']:.2f}"
    )

    print(
        f"EMA50:             "
        f"{latest['ema_50']:.2f}"
    )

    print(
        f"EMA200:            "
        f"{latest['ema_200']:.2f}"
    )

    print(
        f"MACD:              "
        f"{latest['macd']:.2f}"
    )

    print(
        f"MACD signal:       "
        f"{latest['macd_signal']:.2f}"
    )

    print(
        f"Relative volume:   "
        f"{latest['relative_volume']:.2f}"
    )

    print(
        f"1h return:         "
        f"{latest['return_1h']:+.4%}"
    )

    print(
        f"Volatility(20):    "
        f"{latest['volatility_20']:.6f}"
    )