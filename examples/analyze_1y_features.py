from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")


def analyze_condition(df, name, condition):
    subset = df[condition].copy()

    print(f"\n{name}")
    print(f"Occurrences: {len(subset)}")

    if len(subset) == 0:
        return

    for horizon, bars in {
        "5m": 1,
        "15m": 3,
        "1h": 12,
    }.items():
        future_return = (
            df["close"].shift(-bars) / df["close"] - 1
        )

        values = future_return.loc[subset.index].dropna()

        if len(values) == 0:
            continue

        avg_return = values.mean() * 100
        up_rate = (values > 0).mean() * 100

        print(
            f"{horizon}: "
            f"avg future return {avg_return:.4f}%, "
            f"UP {up_rate:.2f}%"
        )


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamps"])

    print(f"Loaded {len(df):,} candles")
    print(f"Start: {df['timestamps'].min()}")
    print(f"End:   {df['timestamps'].max()}")

    print("\n=== BASELINE ===")

    for horizon, bars in {
        "5m": 1,
        "15m": 3,
        "1h": 12,
    }.items():
        future_return = (
            df["close"].shift(-bars) / df["close"] - 1
        )

        values = future_return.dropna()

        print(
            f"{horizon}: "
            f"UP {(values > 0).mean() * 100:.2f}%"
        )

    # RSI
    analyze_condition(
        df,
        "RSI < 30",
        df["rsi_14"] < 30
    )

    analyze_condition(
        df,
        "RSI > 70",
        df["rsi_14"] > 70
    )

    # EMA structure
    analyze_condition(
        df,
        "EMA20 > EMA50",
        df["ema_20"] > df["ema_50"]
    )

    analyze_condition(
        df,
        "EMA20 < EMA50",
        df["ema_20"] < df["ema_50"]
    )

    analyze_condition(
        df,
        "Bullish EMA alignment",
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    )

    analyze_condition(
        df,
        "Bearish EMA alignment",
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"])
    )

    # MACD
    analyze_condition(
        df,
        "MACD > signal",
        df["macd"] > df["macd_signal"]
    )

    analyze_condition(
        df,
        "MACD < signal",
        df["macd"] < df["macd_signal"]
    )

    # Volume
    analyze_condition(
        df,
        "Relative volume > 1.5",
        df["relative_volume"] > 1.5
    )

    analyze_condition(
        df,
        "Relative volume > 2.0",
        df["relative_volume"] > 2.0
    )

    # Momentum
    analyze_condition(
        df,
        "Positive 1h momentum",
        df["return_1h"] > 0
    )

    analyze_condition(
        df,
        "Negative 1h momentum",
        df["return_1h"] < 0
    )

    # Volatility
    volatility_median = df["volatility_20"].median()

    analyze_condition(
        df,
        "Higher-than-median volatility",
        df["volatility_20"] > volatility_median
    )

    analyze_condition(
        df,
        "Lower-than-median volatility",
        df["volatility_20"] <= volatility_median
    )

    # Combined EMA + MACD
    analyze_condition(
        df,
        "Bullish EMA + MACD",
        (df["ema_20"] > df["ema_50"]) &
        (df["macd"] > df["macd_signal"])
    )

    analyze_condition(
        df,
        "Bearish EMA + MACD",
        (df["ema_20"] < df["ema_50"]) &
        (df["macd"] < df["macd_signal"])
    )


if __name__ == "__main__":
    main()