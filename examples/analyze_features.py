import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "btc"
    / "BTCUSDT_5m_features.csv"
)


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("BTC/USDT Feature → Future Return Analysis")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"\nLoaded {len(df):,} candles")


# ============================================================
# Future returns
# ============================================================

# 1 candle = 5 minutes
df["future_5m"] = (
    df["close"].shift(-1) / df["close"] - 1
)

# 3 candles = 15 minutes
df["future_15m"] = (
    df["close"].shift(-3) / df["close"] - 1
)

# 12 candles = 1 hour
df["future_1h"] = (
    df["close"].shift(-12) / df["close"] - 1
)


# ============================================================
# Future direction
# ============================================================

df["direction_5m"] = np.where(
    df["future_5m"] > 0,
    "UP",
    "DOWN"
)

df["direction_15m"] = np.where(
    df["future_15m"] > 0,
    "UP",
    "DOWN"
)

df["direction_1h"] = np.where(
    df["future_1h"] > 0,
    "UP",
    "DOWN"
)


# ============================================================
# Remove rows without future data
# ============================================================

df = df.dropna(
    subset=[
        "future_5m",
        "future_15m",
        "future_1h"
    ]
).reset_index(drop=True)


# ============================================================
# Helper function
# ============================================================

def analyze_condition(name, condition):
    subset = df[condition].copy()

    if len(subset) == 0:
        print(f"\n{name}")
        print("  No matching candles.")
        return

    print(f"\n{name}")
    print("-" * 60)
    print(f"Occurrences : {len(subset):,}")

    print(
        f"5m  average : "
        f"{subset['future_5m'].mean() * 100:.4f}%"
    )

    print(
        f"15m average : "
        f"{subset['future_15m'].mean() * 100:.4f}%"
    )

    print(
        f"1h  average : "
        f"{subset['future_1h'].mean() * 100:.4f}%"
    )

    print(
        f"5m  UP rate : "
        f"{(subset['future_5m'] > 0).mean() * 100:.2f}%"
    )

    print(
        f"15m UP rate : "
        f"{(subset['future_15m'] > 0).mean() * 100:.2f}%"
    )

    print(
        f"1h  UP rate : "
        f"{(subset['future_1h'] > 0).mean() * 100:.2f}%"
    )


# ============================================================
# Baseline
# ============================================================

print("\n" + "=" * 60)
print("BASELINE")
print("=" * 60)

print(
    f"\nOverall 5m UP rate : "
    f"{(df['future_5m'] > 0).mean() * 100:.2f}%"
)

print(
    f"Overall 15m UP rate: "
    f"{(df['future_15m'] > 0).mean() * 100:.2f}%"
)

print(
    f"Overall 1h UP rate : "
    f"{(df['future_1h'] > 0).mean() * 100:.2f}%"
)


# ============================================================
# RSI
# ============================================================

analyze_condition(
    "RSI < 30 (oversold)",
    df["rsi_14"] < 30
)

analyze_condition(
    "RSI > 70 (overbought)",
    df["rsi_14"] > 70
)


# ============================================================
# EMA trend
# ============================================================

analyze_condition(
    "EMA20 > EMA50",
    df["ema_20"] > df["ema_50"]
)

analyze_condition(
    "EMA20 < EMA50",
    df["ema_20"] < df["ema_50"]
)

analyze_condition(
    "EMA20 > EMA50 > EMA200",
    (
        (df["ema_20"] > df["ema_50"])
        & (df["ema_50"] > df["ema_200"])
    )
)

analyze_condition(
    "EMA20 < EMA50 < EMA200",
    (
        (df["ema_20"] < df["ema_50"])
        & (df["ema_50"] < df["ema_200"])
    )
)


# ============================================================
# MACD
# ============================================================

analyze_condition(
    "MACD > Signal",
    df["macd"] > df["macd_signal"]
)

analyze_condition(
    "MACD < Signal",
    df["macd"] < df["macd_signal"]
)


# ============================================================
# Relative volume
# ============================================================

analyze_condition(
    "High relative volume (> 1.5x)",
    df["relative_volume"] > 1.5
)

analyze_condition(
    "Very high relative volume (> 2x)",
    df["relative_volume"] > 2.0
)


# ============================================================
# Momentum
# ============================================================

analyze_condition(
    "Positive 1h momentum",
    df["return_1h"] > 0
)

analyze_condition(
    "Negative 1h momentum",
    df["return_1h"] < 0
)


# ============================================================
# Volatility
# ============================================================

volatility_median = df["volatility_20"].median()

analyze_condition(
    "Higher-than-median volatility",
    df["volatility_20"] > volatility_median
)

analyze_condition(
    "Lower-than-median volatility",
    df["volatility_20"] <= volatility_median
)


# ============================================================
# Combined conditions
# ============================================================

analyze_condition(
    "Bullish EMA + MACD",
    (
        (df["ema_20"] > df["ema_50"])
        & (df["macd"] > df["macd_signal"])
    )
)

analyze_condition(
    "Bearish EMA + MACD",
    (
        (df["ema_20"] < df["ema_50"])
        & (df["macd"] < df["macd_signal"])
    )
)


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)