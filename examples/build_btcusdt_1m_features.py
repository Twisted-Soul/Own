
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("data/btc/BTCUSDT_1m_continuous_2020_2026.parquet")
OUTPUT = Path("data/btc/BTCUSDT_1m_continuous_features.parquet")

if not SOURCE.exists():
    raise SystemExit(f"Source not found: {SOURCE.resolve()}")

print("Loading continuous BTCUSDT 1-minute dataset...")
df = pd.read_parquet(SOURCE)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

if df["timestamp_utc"].duplicated().any():
    raise ValueError("Duplicate timestamps detected.")

close = df["close"]
high = df["high"]
low = df["low"]
volume = df["volume"]

print("Calculating returns and moving averages...")

# Past returns only; no future prices are used as features.
for minutes in [1, 5, 15, 60, 240]:
    df[f"return_{minutes}m_pct"] = (
        close.pct_change(minutes) * 100
    )

for span in [9, 20, 50, 200]:
    df[f"ema_{span}"] = close.ewm(
        span=span, adjust=False, min_periods=span
    ).mean()

df["ema20_50_spread_pct"] = (
    (df["ema_20"] / df["ema_50"]) - 1
) * 100

df["ema50_200_spread_pct"] = (
    (df["ema_50"] / df["ema_200"]) - 1
) * 100

print("Calculating RSI and MACD...")

delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14, adjust=False, min_periods=14
).mean()
avg_loss = loss.ewm(
    alpha=1/14, adjust=False, min_periods=14
).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi14"] = 100 - (100 / (1 + rs))
df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100
df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi14"] = 50

ema12 = close.ewm(
    span=12, adjust=False, min_periods=12
).mean()
ema26 = close.ewm(
    span=26, adjust=False, min_periods=26
).mean()

df["macd"] = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(
    span=9, adjust=False, min_periods=9
).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

print("Calculating volatility and volume features...")

previous_close = close.shift(1)
true_range = pd.concat(
    [
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ],
    axis=1,
).max(axis=1)

df["atr14"] = true_range.ewm(
    alpha=1/14, adjust=False, min_periods=14
).mean()
df["atr14_pct"] = df["atr14"] / close * 100

df["candle_range_pct"] = (high - low) / close * 100
df["candle_body_pct"] = (close - df["open"]) / df["open"] * 100

df["volume_sma20"] = volume.rolling(
    20, min_periods=20
).mean()
df["relative_volume20"] = (
    volume / df["volume_sma20"].replace(0, np.nan)
)

# Order-flow proxy available in this dataset.
if "taker_buy_volume" in df.columns:
    df["taker_buy_ratio"] = (
        df["taker_buy_volume"]
        / volume.replace(0, np.nan)
    )
    df["taker_buy_imbalance"] = (
        2 * df["taker_buy_volume"] - volume
    ) / volume.replace(0, np.nan)

# Explicitly preserve when each candle finishes.
# Features are computed from the current and previous candles only.
df["feature_available_at_utc"] = df["close_time_utc"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT, index=False, compression="snappy")

feature_cols = [
    c for c in df.columns
    if c.startswith((
        "return_", "ema_", "rsi", "macd",
        "atr", "candle_", "volume_sma",
        "relative_volume", "taker_buy_"
    ))
]

print("\nFeature build complete.")
print(f"Rows: {len(df):,}")
print(f"Feature columns added: {len(feature_cols)}")
print(f"Output: {OUTPUT}")
print(f"First timestamp: {df['timestamp_utc'].iloc[0]}")
print(f"Last timestamp:  {df['timestamp_utc'].iloc[-1]}")
print("\nMissing values in selected features:")
print(df[feature_cols].isna().sum().to_string())
print("\nNote: initial indicator warm-up NaNs are expected.")
print("No future-return labels were added.")