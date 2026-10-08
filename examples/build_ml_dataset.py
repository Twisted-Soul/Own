import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_1y_features.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_dataset.csv"
)

# Prediction horizon
HORIZON_BARS = 12   # 1 hour

# Minimum future return required to classify as UP/DOWN.
# Anything inside this band becomes NEUTRAL.
TARGET_THRESHOLD = 0.001  # 0.10%

# ============================================================
# LOAD
# ============================================================

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

df = (
    df
    .sort_values("timestamps")
    .reset_index(drop=True)
)

print(f"Loaded {len(df):,} candles")

# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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

# Verify columns exist.

missing = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing feature columns: "
        + ", ".join(missing)
    )

# ============================================================
# FUTURE RETURN
# ============================================================

print()
print("Building prediction target...")

df["future_return_1h"] = (
    df["close"].shift(-HORIZON_BARS)
    / df["close"]
    - 1
)

# ============================================================
# TARGET
# ============================================================

df["target"] = np.select(
    [
        df["future_return_1h"]
        >= TARGET_THRESHOLD,

        df["future_return_1h"]
        <= -TARGET_THRESHOLD,
    ],
    [
        1,
        -1,
    ],
    default=0,
)

# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before = len(df)

df = df.dropna(
    subset=FEATURE_COLUMNS
    + [
        "future_return_1h",
        "target",
    ]
).copy()

df = df.reset_index(drop=True)

print(
    f"Removed {before - len(df):,} invalid rows"
)

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

counts = (
    df["target"]
    .value_counts()
    .sort_index()
)

total = len(df)

for target, count in counts.items():

    if target == -1:
        name = "DOWN"

    elif target == 0:
        name = "NEUTRAL"

    else:
        name = "UP"

    print(
        f"{name:<10} "
        f"{count:>8,} "
        f"({count / total:>6.2%})"
    )

# ============================================================
# BASIC TARGET STATISTICS
# ============================================================

print()
print("Future 1h return statistics:")

print(
    df["future_return_1h"]
    .describe()
)

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

# 70% train
# 15% validation
# 15% test

n = len(df)

train_end = int(n * 0.70)

validation_end = int(n * 0.85)

df["dataset"] = "test"

df.loc[
    :train_end - 1,
    "dataset"
] = "train"

df.loc[
    train_end:validation_end - 1,
    "dataset"
] = "validation"

print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

for split in [
    "train",
    "validation",
    "test",
]:

    subset = df[
        df["dataset"] == split
    ]

    print()
    print(split.upper())

    print(
        f"Rows:   {len(subset):,}"
    )

    print(
        f"Start:  {subset['timestamps'].iloc[0]}"
    )

    print(
        f"End:    {subset['timestamps'].iloc[-1]}"
    )

# ============================================================
# SAVE
# ============================================================

OUTPUT_COLUMNS = (
    [
        "timestamps",
        "dataset",
        "future_return_1h",
        "target",
    ]
    + FEATURE_COLUMNS
)

output = df[
    OUTPUT_COLUMNS
].copy()

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("ML DATASET COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Rows:  {len(output):,}"
)

print(
    f"Features: {len(FEATURE_COLUMNS):,}"
)