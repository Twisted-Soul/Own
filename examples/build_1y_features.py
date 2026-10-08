import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")


def main():
    print("Loading 1-year BTC dataset...")

    df = pd.read_csv(INPUT_FILE)
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)

    df = df.sort_values("timestamps").reset_index(drop=True)

    print(f"Loaded {len(df):,} candles.")

    # ---------------------------------------------------------
    # EMAs
    # ---------------------------------------------------------

    df["ema_20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema_50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema_200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # ---------------------------------------------------------
    # RSI
    # ---------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    ema_12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # ---------------------------------------------------------
    # ATR
    # ---------------------------------------------------------

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    df["volume_ma_20"] = df["volume"].rolling(20).mean()

    df["relative_volume"] = (
        df["volume"] / df["volume_ma_20"]
    )

    # ---------------------------------------------------------
    # Rolling highs / lows
    # ---------------------------------------------------------

    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()

    df["high_50"] = df["high"].rolling(50).max()
    df["low_50"] = df["low"].rolling(50).min()

    # ---------------------------------------------------------
    # Returns
    # ---------------------------------------------------------

    df["return_5m"] = df["close"].pct_change(1)

    df["return_15m"] = df["close"].pct_change(3)

    df["return_1h"] = df["close"].pct_change(12)

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    df["volatility_20"] = (
        df["return_5m"]
        .rolling(20)
        .std()
    )

    # ---------------------------------------------------------
    # Distance from EMAs
    # ---------------------------------------------------------

    df["distance_ema_20"] = (
        df["close"] / df["ema_20"] - 1
    )

    df["distance_ema_50"] = (
        df["close"] / df["ema_50"] - 1
    )

    df["distance_ema_200"] = (
        df["close"] / df["ema_200"] - 1
    )

    # ---------------------------------------------------------
    # Candle structure
    # ---------------------------------------------------------

    df["candle_range"] = (
        df["high"] - df["low"]
    )

    df["candle_body"] = (
        df["close"] - df["open"]
    )

    df["candle_body_pct"] = (
        df["candle_body"] / df["open"]
    )

    # ---------------------------------------------------------
    # Remove warm-up period
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna().reset_index(drop=True)

    removed = before - len(df)

    print(f"Removed {removed:,} warm-up/invalid rows.")
    print(f"Final dataset: {len(df):,} candles.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Feature generation complete.")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("Columns:")
    print(", ".join(df.columns))

    print()
    print("Latest snapshot:")
    print(
        df[
            [
                "timestamps",
                "close",
                "ema_20",
                "ema_50",
                "ema_200",
                "rsi_14",
                "macd",
                "macd_signal",
                "atr_14",
                "relative_volume",
                "return_1h",
                "volatility_20",
            ]
        ].tail(1).to_string(index=False)
    )


if __name__ == "__main__":
    main()