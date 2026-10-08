import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/btc/BTCUSDT_5m_signals.csv")


def max_drawdown(equity):
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def evaluate_horizon(df, periods, label):
    future_return = df["close"].shift(-periods) / df["close"] - 1

    results = []

    for signal in ["BUY", "SELL"]:
        mask = df["signal"] == signal
        returns = future_return[mask].dropna()

        if len(returns) == 0:
            continue

        if signal == "BUY":
            correct = returns > 0
            strategy_returns = returns
        else:
            correct = returns < 0
            strategy_returns = -returns

        results.append({
            "Signal": signal,
            "Trades": len(returns),
            "Hit Rate": correct.mean(),
            "Avg Future Return": returns.mean(),
            "Avg Directional Return": strategy_returns.mean(),
        })

    print(f"\n{'=' * 60}")
    print(f"{label} FUTURE-MOVEMENT ANALYSIS")
    print(f"{'=' * 60}")

    if results:
        result_df = pd.DataFrame(results)

        result_df["Hit Rate"] *= 100
        result_df["Avg Future Return"] *= 100
        result_df["Avg Directional Return"] *= 100

        print(
            result_df.to_string(
                index=False,
                formatters={
                    "Hit Rate": "{:.2f}%".format,
                    "Avg Future Return": "{:.4f}%".format,
                    "Avg Directional Return": "{:.4f}%".format,
                },
            )
        )

    # Combined directional signals
    mask = df["signal"].isin(["BUY", "SELL"])

    directional_returns = future_return[mask].copy()
    signals = df.loc[mask, "signal"]

    aligned = pd.DataFrame({
        "signal": signals,
        "future_return": directional_returns,
    }).dropna()

    if len(aligned) > 0:
        aligned["strategy_return"] = np.where(
            aligned["signal"] == "BUY",
            aligned["future_return"],
            -aligned["future_return"],
        )

        print("\nCombined directional signals:")
        print(f"Trades: {len(aligned)}")
        print(
            f"Directional hit rate: "
            f"{(aligned['strategy_return'] > 0).mean() * 100:.2f}%"
        )
        print(
            f"Average strategy return: "
            f"{aligned['strategy_return'].mean() * 100:.4f}%"
        )


def run_equity_backtest(df):
    print(f"\n{'=' * 60}")
    print("SIMPLE SIGNAL EQUITY BACKTEST")
    print(f"{'=' * 60}")

    # Position generated from the previous candle's signal.
    # This avoids using the current candle's future return.
    position = df["signal"].map({
        "BUY": 1,
        "SELL": -1,
        "HOLD": 0,
    }).shift(1).fillna(0)

    market_return = df["close"].pct_change()

    strategy_return = position * market_return

    equity = (1 + strategy_return.fillna(0)).cumprod()

    buy_and_hold = (1 + market_return.fillna(0)).cumprod()

    total_strategy = equity.iloc[-1] - 1
    total_bh = buy_and_hold.iloc[-1] - 1

    drawdown = max_drawdown(equity)

    trades = ((position != 0) & (position.shift(1) != position)).sum()

    print(f"Strategy return:   {total_strategy * 100:.2f}%")
    print(f"Buy & hold return: {total_bh * 100:.2f}%")
    print(f"Max drawdown:      {drawdown * 100:.2f}%")
    print(f"Position changes:  {trades}")

    print("\nSignal counts:")
    print(df["signal"].value_counts())

    output = df.copy()
    output["position"] = position
    output["market_return"] = market_return
    output["strategy_return"] = strategy_return
    output["equity"] = equity
    output["buy_hold_equity"] = buy_and_hold

    output_file = Path("data/btc/BTCUSDT_5m_backtest.csv")
    output.to_csv(output_file, index=False)

    print(f"\nSaved backtest data to:")
    print(output_file)


def main():
    print("Loading signal dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["timestamps"] = pd.to_datetime(
        df["timestamps"],
        utc=True,
    )

    df = df.sort_values("timestamps").reset_index(drop=True)

    print(f"Loaded {len(df)} candles.")
    print(
        f"Period: {df['timestamps'].iloc[0]} "
        f"-> {df['timestamps'].iloc[-1]}"
    )

    # 1 candle = 5 minutes
    evaluate_horizon(df, 1, "5-MINUTE")
    evaluate_horizon(df, 3, "15-MINUTE")
    evaluate_horizon(df, 12, "1-HOUR")

    run_equity_backtest(df)


if __name__ == "__main__":
    main()