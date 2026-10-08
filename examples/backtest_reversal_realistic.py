from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_reversal_realistic.csv")

STARTING_CAPITAL = 10_000.0

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

TOTAL_COST = (
    FEE_PER_SIDE * 2
    + SLIPPAGE_PER_SIDE * 2
)

# Longer holding periods
HORIZONS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
}


def run_backtest(df, horizon_name, bars):

    capital = STARTING_CAPITAL
    trades = []

    i = 0

    while i < len(df) - bars:

        row = df.iloc[i]

        # Long signal
        long_signal = (
            row["rsi_14"] < 30
            and row["relative_volume"] > 1.5
        )

        # Short signal
        short_signal = (
            row["rsi_14"] > 70
            and row["relative_volume"] > 1.5
        )

        if not long_signal and not short_signal:
            i += 1
            continue

        entry_price = row["close"]
        exit_row = df.iloc[i + bars]
        exit_price = exit_row["close"]

        if long_signal:
            gross_return = (
                exit_price / entry_price - 1
            )
            direction = "LONG"

        else:
            gross_return = (
                entry_price / exit_price - 1
            )
            direction = "SHORT"

        net_return = gross_return - TOTAL_COST

        capital_before = capital
        capital *= (1 + net_return)

        trades.append({
            "entry_time": row["timestamps"],
            "exit_time": exit_row["timestamps"],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": net_return,
            "capital_before": capital_before,
            "capital_after": capital,
        })

        # Important:
        # Move forward to the exit candle so trades cannot overlap.
        i += bars

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return None

    # Equity curve
    equity = trades_df["capital_after"]

    running_max = equity.cummax()
    drawdown = (
        equity / running_max - 1
    )

    max_drawdown = drawdown.min() * 100

    win_rate = (
        trades_df["net_return"] > 0
    ).mean() * 100

    avg_return = (
        trades_df["net_return"].mean() * 100
    )

    median_return = (
        trades_df["net_return"].median() * 100
    )

    final_capital = trades_df.iloc[-1]["capital_after"]

    total_return = (
        final_capital / STARTING_CAPITAL - 1
    ) * 100

    long_trades = trades_df[
        trades_df["direction"] == "LONG"
    ]

    short_trades = trades_df[
        trades_df["direction"] == "SHORT"
    ]

    return {
        "horizon": horizon_name,
        "trades": len(trades_df),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "final_capital": final_capital,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "trades_df": trades_df,
    }


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamps"]
    )

    print(f"Loaded {len(df):,} candles")
    print(f"Start: {df['timestamps'].min()}")
    print(f"End:   {df['timestamps'].max()}")

    print(f"\nStarting capital: ${STARTING_CAPITAL:,.2f}")
    print(f"Round-trip cost: {TOTAL_COST * 100:.2f}%")

    summary = []

    for horizon_name, bars in HORIZONS.items():

        result = run_backtest(
            df,
            horizon_name,
            bars
        )

        if result is None:
            continue

        summary.append({
            "horizon": result["horizon"],
            "trades": result["trades"],
            "long_trades": result["long_trades"],
            "short_trades": result["short_trades"],
            "win_rate": result["win_rate"],
            "avg_return": result["avg_return"],
            "median_return": result["median_return"],
            "final_capital": result["final_capital"],
            "total_return": result["total_return"],
            "max_drawdown": result["max_drawdown"],
        })

        print(
            f"\n{horizon_name}"
        )

        print(
            f"Trades:       {result['trades']:,}"
        )

        print(
            f"Long:         {result['long_trades']:,}"
        )

        print(
            f"Short:        {result['short_trades']:,}"
        )

        print(
            f"Win rate:     {result['win_rate']:.2f}%"
        )

        print(
            f"Avg trade:    {result['avg_return']:.4f}%"
        )

        print(
            f"Median trade: {result['median_return']:.4f}%"
        )

        print(
            f"Final capital:${result['final_capital']:,.2f}"
        )

        print(
            f"Total return: {result['total_return']:.2f}%"
        )

        print(
            f"Max drawdown: {result['max_drawdown']:.2f}%"
        )

        # Save individual trade history
        trade_file = Path(
            f"data/btc/"
            f"reversal_trades_{horizon_name}.csv"
        )

        result["trades_df"].to_csv(
            trade_file,
            index=False
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved summary to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()