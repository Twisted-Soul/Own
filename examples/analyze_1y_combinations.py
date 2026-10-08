from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")


def analyze_condition(df, name, condition):
    subset = df.loc[condition].copy()

    print(f"\n{name}")
    print(f"Occurrences: {len(subset):,}")

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

        print(
            f"{horizon}: "
            f"avg return {values.mean() * 100:.4f}%, "
            f"median {values.median() * 100:.4f}%, "
            f"UP {(values > 0).mean() * 100:.2f}%"
        )


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamps"]
    )

    print(f"Loaded {len(df):,} candles")
    print(f"Start: {df['timestamps'].min()}")
    print(f"End:   {df['timestamps'].max()}")

    # ---------------------------------------------------------
    # RSI + MOMENTUM
    # ---------------------------------------------------------

    analyze_condition(
        df,
        "RSI < 30 + negative 1h momentum",
        (df["rsi_14"] < 30) &
        (df["return_1h"] < 0)
    )

    analyze_condition(
        df,
        "RSI < 30 + positive 1h momentum",
        (df["rsi_14"] < 30) &
        (df["return_1h"] > 0)
    )

    analyze_condition(
        df,
        "RSI > 70 + positive 1h momentum",
        (df["rsi_14"] > 70) &
        (df["return_1h"] > 0)
    )

    analyze_condition(
        df,
        "RSI > 70 + negative 1h momentum",
        (df["rsi_14"] > 70) &
        (df["return_1h"] < 0)
    )

    # ---------------------------------------------------------
    # RSI + VOLUME
    # ---------------------------------------------------------

    analyze_condition(
        df,
        "RSI < 30 + high volume",
        (df["rsi_14"] < 30) &
        (df["relative_volume"] > 1.5)
    )

    analyze_condition(
        df,
        "RSI > 70 + high volume",
        (df["rsi_14"] > 70) &
        (df["relative_volume"] > 1.5)
    )

    # ---------------------------------------------------------
    # RSI + VOLATILITY
    # ---------------------------------------------------------

    volatility_median = df["volatility_20"].median()

    analyze_condition(
        df,
        "RSI < 30 + high volatility",
        (df["rsi_14"] < 30) &
        (df["volatility_20"] > volatility_median)
    )

    analyze_condition(
        df,
        "RSI > 70 + high volatility",
        (df["rsi_14"] > 70) &
        (df["volatility_20"] > volatility_median)
    )

    # ---------------------------------------------------------
    # RSI + EMA
    # ---------------------------------------------------------

    analyze_condition(
        df,
        "RSI < 30 + EMA20 < EMA50",
        (df["rsi_14"] < 30) &
        (df["ema_20"] < df["ema_50"])
    )

    analyze_condition(
        df,
        "RSI < 30 + EMA20 > EMA50",
        (df["rsi_14"] < 30) &
        (df["ema_20"] > df["ema_50"])
    )

    analyze_condition(
        df,
        "RSI > 70 + EMA20 > EMA50",
        (df["rsi_14"] > 70) &
        (df["ema_20"] > df["ema_50"])
    )

    analyze_condition(
        df,
        "RSI > 70 + EMA20 < EMA50",
        (df["rsi_14"] > 70) &
        (df["ema_20"] < df["ema_50"])
    )

    # ---------------------------------------------------------
    # EXTREME RSI + VOLUME + MOMENTUM
    # ---------------------------------------------------------

    analyze_condition(
        df,
        "RSI < 30 + high volume + negative momentum",
        (df["rsi_14"] < 30) &
        (df["relative_volume"] > 1.5) &
        (df["return_1h"] < 0)
    )

    analyze_condition(
        df,
        "RSI < 30 + high volume + positive momentum",
        (df["rsi_14"] < 30) &
        (df["relative_volume"] > 1.5) &
        (df["return_1h"] > 0)
    )

    analyze_condition(
        df,
        "RSI > 70 + high volume + positive momentum",
        (df["rsi_14"] > 70) &
        (df["relative_volume"] > 1.5) &
        (df["return_1h"] > 0)
    )

    analyze_condition(
        df,
        "RSI > 70 + high volume + negative momentum",
        (df["rsi_14"] > 70) &
        (df["relative_volume"] > 1.5) &
        (df["return_1h"] < 0)
    )

    # ---------------------------------------------------------
    # EMA + MACD + MOMENTUM
    # ---------------------------------------------------------

    analyze_condition(
        df,
        "Bullish EMA + MACD + positive momentum",
        (df["ema_20"] > df["ema_50"]) &
        (df["macd"] > df["macd_signal"]) &
        (df["return_1h"] > 0)
    )

    analyze_condition(
        df,
        "Bearish EMA + MACD + negative momentum",
        (df["ema_20"] < df["ema_50"]) &
        (df["macd"] < df["macd_signal"]) &
        (df["return_1h"] < 0)
    )

    # ---------------------------------------------------------
    # CONTRARIAN EXTREMES
    # ---------------------------------------------------------

    analyze_condition(
        df,
        "Oversold reversal setup",
        (df["rsi_14"] < 30) &
        (df["return_1h"] < 0) &
        (df["relative_volume"] > 1.5)
    )

    analyze_condition(
        df,
        "Overbought reversal setup",
        (df["rsi_14"] > 70) &
        (df["return_1h"] > 0) &
        (df["relative_volume"] > 1.5)
    )


if __name__ == "__main__":
    main()