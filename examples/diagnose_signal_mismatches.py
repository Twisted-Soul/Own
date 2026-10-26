import pandas as pd
import numpy as np
from pathlib import Path

LONG = Path("data/btc/BTCUSDT_5m_full_features.csv")
SHORT = Path("data/btc/BTCUSDT_5m_features.csv")
SIGNALS = Path("data/btc/BTCUSDT_5m_signals.csv")

long_df = pd.read_csv(LONG)
short_df = pd.read_csv(SHORT)
signals = pd.read_csv(SIGNALS)

for frame in (long_df, short_df, signals):
    frame["timestamps"] = pd.to_datetime(
        frame["timestamps"], utc=True, errors="coerce"
    )

merged = long_df.merge(
    short_df,
    on="timestamps",
    how="inner",
    suffixes=("_long", "_short"),
)

print("Overlapping candles:", len(merged))

columns = [
    "ema_20", "ema_50", "ema_200", "rsi_14",
    "macd", "macd_signal", "return_1h", "relative_volume",
    "high_20", "low_20", "high_50", "low_50",
]

print("\n=== INDICATOR DIFFERENCES ===")
for col in columns:
    a = pd.to_numeric(merged[f"{col}_long"], errors="coerce")
    b = pd.to_numeric(merged[f"{col}_short"], errors="coerce")
    valid = a.notna() & b.notna()

    if valid.sum() == 0:
        print(f"{col}: no comparable values")
        continue

    diff = (a[valid] - b[valid]).abs()
    print(
        f"{col:18s} | "
        f"exact matches={(diff < 1e-8).mean() * 100:6.2f}% | "
        f"mean abs diff={diff.mean():.8f} | "
        f"max abs diff={diff.max():.8f}"
    )

# Reconstruct signals using the short dataset's own indicator values.
rsi = pd.to_numeric(short_df["rsi_14"], errors="coerce")
ema20 = pd.to_numeric(short_df["ema_20"], errors="coerce")
ema50 = pd.to_numeric(short_df["ema_50"], errors="coerce")
macd = pd.to_numeric(short_df["macd"], errors="coerce")
macd_sig = pd.to_numeric(short_df["macd_signal"], errors="coerce")
ret1h = pd.to_numeric(short_df["return_1h"], errors="coerce")
relvol = pd.to_numeric(short_df["relative_volume"], errors="coerce")

bull = (
    (rsi < 30).astype(int)
    + (ema20 > ema50).astype(int)
    + (macd > macd_sig).astype(int)
    + (ret1h > 0).astype(int)
    + ((relvol > 1.5) & (ret1h > 0)).astype(int)
)

bear = (
    (rsi > 70).astype(int)
    + (ema20 < ema50).astype(int)
    + (macd < macd_sig).astype(int)
    + (ret1h < 0).astype(int)
    + ((relvol > 1.5) & (ret1h < 0)).astype(int)
)

reconstructed = pd.Series("HOLD", index=short_df.index)
reconstructed.loc[(bull >= 3) & (bull > bear)] = "BUY"
reconstructed.loc[(bear >= 3) & (bear > bull)] = "SELL"

check = short_df[["timestamps"]].copy()
check["reconstructed"] = reconstructed
check = check.merge(
    signals[["timestamps", "signal"]],
    on="timestamps",
    how="inner",
)

print("\n=== RULE REPRODUCTION USING SHORT-DATA INDICATORS ===")
print("Compared candles:", len(check))
print(
    "Exact matches:",
    int((check["reconstructed"] == check["signal"]).sum()),
)
print(
    "Match rate (%):",
    round((check["reconstructed"] == check["signal"]).mean() * 100, 2),
)
print("\nMismatch examples:")
print(
    check.loc[check["reconstructed"] != check["signal"]]
         .head(10)
         .to_string(index=False)
)
