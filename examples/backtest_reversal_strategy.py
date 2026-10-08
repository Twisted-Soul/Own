from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_reversal_backtest.csv")

# Trading assumptions
FEE_PER_SIDE = 0.0005       # 0.05%
SLIPPAGE_PER_SIDE = 0.0002  # 0.02%

# Holding periods
HORIZONS = {
    "5m": 1,
    "15m": 3,
    "1h": 12,
}


def run_strategy(df, strategy_name, long_condition, short_condition):
    results = []

    for horizon_name, bars in HORIZONS.items():

        future_price = df["close"].shift(-bars)

        long_entries = df.loc[long_condition].copy()
        short_entries = df.loc[short_condition].copy()

        long_returns = (
            future_price.loc[long_entries.index]
            / long_entries["close"]
            - 1
        )

        short_returns = (
            short_entries["close"]
            / future_price.loc[short_entries.index]
            - 1
        )

        # Remove rows where future price doesn't exist
        long_returns = long_returns.dropna()
        short_returns = short_returns.dropna()

        # Costs:
        # entry fee + exit fee + entry slippage + exit slippage
        total_cost = (
            FEE_PER_SIDE * 2
            + SLIPPAGE_PER_SIDE * 2
        )

        long_net = long_returns - total_cost
        short_net = short_returns - total_cost

        all_returns = pd.concat([long_net, short_net])

        if len(all_returns) == 0:
            continue

        results.append({
            "strategy": strategy_name,
            "horizon": horizon_name,
            "trades": len(all_returns),
            "long_trades": len(long_net),
            "short_trades": len(short_net),
            "win_rate": (all_returns > 0).mean() * 100,
            "avg_return": all_returns.mean() * 100,
            "median_return": all_returns.median() * 100,
            "total_return": all_returns.sum() * 100,
        })

    return results


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamps"]
    )

    print(f"Loaded {len(df):,} candles")
    print(f"Start: {df['timestamps'].min()}")
    print(f"End:   {df['timestamps'].max()}")

    # ---------------------------------------------------------
    # STRATEGY A
    # Oversold + high volume
    # ---------------------------------------------------------

    long_a = (
        (df["rsi_14"] < 30) &
        (df["relative_volume"] > 1.5)
    )

    short_a = (
        (df["rsi_14"] > 70) &
        (df["relative_volume"] > 1.5)
    )

    # ---------------------------------------------------------
    # STRATEGY B
    # More selective reversal
    # ---------------------------------------------------------

    long_b = (
        (df["rsi_14"] < 30) &
        (df["relative_volume"] > 1.5) &
        (df["volatility_20"] > df["volatility_20"].median())
    )

    short_b = (
        (df["rsi_14"] > 70) &
        (df["relative_volume"] > 1.5) &
        (df["volatility_20"] > df["volatility_20"].median())
    )

    # ---------------------------------------------------------
    # STRATEGY C
    # RSI extreme + EMA context
    # ---------------------------------------------------------

    long_c = (
        (df["rsi_14"] < 30) &
        (df["relative_volume"] > 1.5) &
        (df["ema_20"] < df["ema_50"])
    )

    short_c = (
        (df["rsi_14"] > 70) &
        (df["relative_volume"] > 1.5) &
        (df["ema_20"] > df["ema_50"])
    )

    all_results = []

    all_results.extend(
        run_strategy(
            df,
            "A_RSI_volume",
            long_a,
            short_a
        )
    )

    all_results.extend(
        run_strategy(
            df,
            "B_RSI_volume_volatility",
            long_b,
            short_b
        )
    )

    all_results.extend(
        run_strategy(
            df,
            "C_RSI_volume_EMA",
            long_c,
            short_c
        )
    )

    results_df = pd.DataFrame(all_results)

    print("\n=== STRATEGY RESULTS ===")

    for _, row in results_df.iterrows():

        print(
            f"\n{row['strategy']} | {row['horizon']}"
        )

        print(
            f"Trades:       {int(row['trades']):,}"
        )

        print(
            f"Long trades:  {int(row['long_trades']):,}"
        )

        print(
            f"Short trades: {int(row['short_trades']):,}"
        )

        print(
            f"Win rate:     {row['win_rate']:.2f}%"
        )

        print(
            f"Avg return:   {row['avg_return']:.4f}%"
        )

        print(
            f"Median:       {row['median_return']:.4f}%"
        )

        print(
            f"Total return: {row['total_return']:.2f}%"
        )

    results_df.to_csv(OUTPUT_FILE, index=False)

    print(
        f"\nSaved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()