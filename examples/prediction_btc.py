import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor


# =========================
# 1. Load BTCUSDT data
# =========================

df = pd.read_csv("./data/btc/BTCUSDT_5m_clean.csv")

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

print(f"Loaded {len(df)} BTCUSDT candles.")
print(f"First candle: {df['timestamps'].iloc[0]}")
print(f"Last candle:  {df['timestamps'].iloc[-1]}")


# =========================
# 2. Select prediction setup
# =========================

lookback = 400
pred_len = 120

# We need enough data for:
# 400 historical candles + 120 future candles
if len(df) < lookback + pred_len:
    raise ValueError(
        f"Not enough data. Need at least "
        f"{lookback + pred_len} candles."
    )


# Use the first 400 candles as context.
# The following 120 candles are kept as ground truth.
x_df = df.iloc[:lookback].copy()
actual_df = df.iloc[lookback:lookback + pred_len].copy()


x_timestamp = x_df["timestamps"]
y_timestamp = actual_df["timestamps"]


# =========================
# 3. Load Kronos
# =========================

print("\nLoading Kronos tokenizer...")

tokenizer = KronosTokenizer.from_pretrained(
    "NeoQuasar/Kronos-Tokenizer-base"
)

print("Loading Kronos-small...")

model = Kronos.from_pretrained(
    "NeoQuasar/Kronos-small"
)

predictor = KronosPredictor(
    model,
    tokenizer,
    max_context=512
)


# =========================
# 4. Generate prediction
# =========================

print("\nGenerating BTCUSDT forecast...")
print(f"Context: {lookback} candles")
print(f"Prediction: {pred_len} candles")

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)


# =========================
# 5. Display prediction
# =========================

print("\nPrediction generated successfully!")

print("\nPredicted candles:")
print(pred_df.head())

print("\nLast predicted candle:")
print(pred_df.tail(1))


# =========================
# 6. Compare with reality
# =========================

actual_close = actual_df["close"].values
predicted_close = pred_df["close"].values


mae = abs(predicted_close - actual_close).mean()

rmse = (
    ((predicted_close - actual_close) ** 2).mean()
) ** 0.5


print("\nPrediction accuracy")
print("-------------------")
print(f"Close MAE:  {mae:.4f} USDT")
print(f"Close RMSE: {rmse:.4f} USDT")


# =========================
# 7. Plot
# =========================

plt.figure(figsize=(14, 6))

plt.plot(
    actual_df["timestamps"],
    actual_close,
    label="Actual BTCUSDT"
)

plt.plot(
    pred_df.index,
    predicted_close,
    label="Kronos Prediction"
)

plt.title("Kronos BTCUSDT 5-Minute Forecast")

plt.xlabel("Time")
plt.ylabel("Price (USDT)")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()