import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/btc/BTCUSDT_5m_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_signals.csv")


def generate_signal(row):
    bullish = 0
    bearish = 0

    # RSI
    if row["rsi_14"] < 30:
        bullish += 1
    elif row["rsi_14"] > 70:
        bearish += 1

    # EMA trend
    if row["ema_20"] > row["ema_50"]:
        bullish += 1
    elif row["ema_20"] < row["ema_50"]:
        bearish += 1

    # MACD
    if row["macd"] > row["macd_signal"]:
        bullish += 1
    elif row["macd"] < row["macd_signal"]:
        bearish += 1

    # Momentum
    if row["return_1h"] > 0:
        bullish += 1
    elif row["return_1h"] < 0:
        bearish += 1

    # Relative volume
    if row["relative_volume"] > 1.5:
        if row["return_1h"] > 0:
            bullish += 1
        elif row["return_1h"] < 0:
            bearish += 1

    # Signal decision
    if bullish >= 3 and bullish > bearish:
        return "BUY"

    if bearish >= 3 and bearish > bullish:
        return "SELL"

    return "HOLD"


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_FILE)
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)

    print(f"Loaded {len(df)} candles.")

    print("Generating signals...")

    df["signal"] = df.apply(generate_signal, axis=1)

    print("\nSignal distribution:")
    print(df["signal"].value_counts())

    print("\nLatest signals:")
    print(
        df[
            [
                "timestamps",
                "close",
                "rsi_14",
                "ema_20",
                "ema_50",
                "macd",
                "macd_signal",
                "return_1h",
                "relative_volume",
                "signal",
            ]
        ].tail(20).to_string(index=False)
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved signals to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()