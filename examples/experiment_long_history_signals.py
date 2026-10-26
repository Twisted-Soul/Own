import pandas as pd
from pathlib import Path

SOURCE = Path("data/btc/BTCUSDT_5m_full_features.csv")
REFERENCE = Path("data/btc/BTCUSDT_5m_signals.csv")
OUTPUT = Path("data/btc/BTCUSDT_5m_full_signals_experiment.csv")

df = pd.read_csv(SOURCE)
reference = pd.read_csv(REFERENCE)

required = {
    "timestamps", "open", "high", "low", "close",
    "ema_20", "ema_50", "rsi_14", "macd",
    "macd_signal", "return_1h", "relative_volume",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["timestamps"] = pd.to_datetime(
    df["timestamps"], utc=True, errors="coerce"
)
reference["timestamps"] = pd.to_datetime(
    reference["timestamps"], utc=True, errors="coerce"
)

df = (
    df.dropna(subset=["timestamps", "open", "high", "low", "close"])
      .sort_values("timestamps")
      .drop_duplicates("timestamps")
      .reset_index(drop=True)
)

# Reproduce the existing five-vote signal rules.
rsi = df["rsi_14"]
ema20 = df["ema_20"]
ema50 = df["ema_50"]
macd = df["macd"]
macd_signal = df["macd_signal"]
ret1h = df["return_1h"]
relvol = df["relative_volume"]

bull = (
    (rsi < 30).astype(int)
    + (ema20 > ema50).astype(int)
    + (macd > macd_signal).astype(int)
    + (ret1h > 0).astype(int)
    + ((relvol > 1.5) & (ret1h > 0)).astype(int)
)

bear = (
    (rsi > 70).astype(int)
    + (ema20 < ema50).astype(int)
    + (macd < macd_signal).astype(int)
    + (ret1h < 0).astype(int)
    + ((relvol > 1.5) & (ret1h < 0)).astype(int)
)

df["bullish_votes"] = bull
df["bearish_votes"] = bear
df["signal"] = "HOLD"

df.loc[(bull >= 3) & (bull > bear), "signal"] = "BUY"
df.loc[(bear >= 3) & (bear > bull), "signal"] = "SELL"

# Compare reconstructed signals against the existing short-period file.
comparison = df[["timestamps", "signal"]].merge(
    reference[["timestamps", "signal"]],
    on="timestamps",
    how="inner",
    suffixes=("_new", "_reference"),
)

if len(comparison):
    comparison["match"] = (
        comparison["signal_new"] == comparison["signal_reference"]
    )
    print("\n=== REFERENCE SIGNAL CHECK ===")
    print("Overlapping candles:", len(comparison))
    print("Exact signal matches:", int(comparison["match"].sum()))
    print("Match rate (%):", round(comparison["match"].mean() * 100, 2))

    if not comparison["match"].all():
        print("\nFirst mismatches:")
        print(
            comparison.loc[
                ~comparison["match"],
                ["timestamps", "signal_new", "signal_reference"],
            ].head(10).to_string(index=False)
        )
else:
    print("WARNING: No overlapping timestamps found.")

df.to_csv(OUTPUT, index=False)

print("\n=== LONG HISTORY SIGNALS ===")
print("Rows:", len(df))
print("First candle:", df["timestamps"].min())
print("Last candle:", df["timestamps"].max())
print(df["signal"].value_counts().to_string())
print("\nSaved:", OUTPUT)
