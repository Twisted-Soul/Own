import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
TRADES_OUT = Path("data/btc/BTCUSDT_trade_ledger_experiment.csv")
EQUITY_OUT = Path("data/btc/BTCUSDT_trade_ledger_equity.csv")

MAX_HOLD_BARS = 12
COST_SCENARIOS = [0.0, 0.0002, 0.0007, 0.0010]


def confirm_entries(raw):
    previous = raw.shift(1)
    return raw.where(
        raw.isin(["BUY", "SELL"]) & raw.eq(previous),
        "HOLD"
    )


def run_simulation(df):
    signals = confirm_entries(df["signal"])
    trades = []
    actions = []
    positions = []

    position = 0
    entry_i = None
    trade_id = 0

    for i in range(len(df)):
        sig = signals.iloc[i]
        price = float(df["close"].iloc[i])
        action = "HOLD"
        reason = ""

        if position == 0:
            if sig in ("BUY", "SELL"):
                position = 1 if sig == "BUY" else -1
                entry_i = i
                trade_id += 1
                action = sig

        else:
            held_bars = i - entry_i
            opposite = (
                (position == 1 and sig == "SELL")
                or (position == -1 and sig == "BUY")
            )

            if opposite or held_bars >= MAX_HOLD_BARS:
                reason = (
                    "CONFIRMED_OPPOSITE_SIGNAL"
                    if opposite else "MAX_HOLD_TIME"
                )
                entry_price = float(df["close"].iloc[entry_i])
                direction = "LONG" if position == 1 else "SHORT"
                gross_return = position * (price / entry_price - 1.0)

                trades.append({
                    "trade_id": trade_id,
                    "direction": direction,
                    "entry_time": df["timestamps"].iloc[entry_i],
                    "exit_time": df["timestamps"].iloc[i],
                    "entry_index": entry_i,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "holding_bars": held_bars,
                    "holding_minutes": held_bars * 5,
                    "exit_reason": reason,
                    "gross_return_pct": gross_return * 100,
                })

                position = 0
                entry_i = None
                action = "EXIT"

                # Deliberately do not reverse on the same candle.

        actions.append(action)
        positions.append(position)

    # If the dataset ends with an open trade, close it at the final
    # close for offline accounting and mark it as an end-of-data exit.
    if position != 0 and entry_i is not None:
        i = len(df) - 1
        entry_price = float(df["close"].iloc[entry_i])
        price = float(df["close"].iloc[i])
        gross_return = position * (price / entry_price - 1.0)

        trades.append({
            "trade_id": trade_id,
            "direction": "LONG" if position == 1 else "SHORT",
            "entry_time": df["timestamps"].iloc[entry_i],
            "exit_time": df["timestamps"].iloc[i],
            "entry_index": entry_i,
            "exit_index": i,
            "entry_price": entry_price,
            "exit_price": price,
            "holding_bars": i - entry_i,
            "holding_minutes": (i - entry_i) * 5,
            "exit_reason": "END_OF_DATA_MARK",
            "gross_return_pct": gross_return * 100,
        })

    out = df.copy()
    out["confirmed_entry_signal"] = signals
    out["action"] = actions
    out["position_at_close"] = positions

    return out, pd.DataFrame(trades)


def main():
    df = pd.read_csv(INPUT)
    required = {"timestamps", "close", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (
        df.dropna(subset=["timestamps", "close"])
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )
    df["signal"] = df["signal"].fillna("HOLD").astype(str).str.upper()

    invalid = ~df["signal"].isin(["BUY", "SELL", "HOLD"])
    if invalid.any():
        raise ValueError(
            f"Unexpected signals: {df.loc[invalid, 'signal'].unique()}"
        )

    out, trades = run_simulation(df)
    if trades.empty:
        raise ValueError("No completed trades found.")

    # Cost is applied at entry and exit: 2 * cost per round trip.
    summary = []
    equity_data = []

    for cost in COST_SCENARIOS:
        per_trade_cost = 2 * cost
        trade_returns = (
            trades["gross_return_pct"] / 100 - per_trade_cost
        )
        equity = (1 + trade_returns).cumprod()
        drawdown = equity / equity.cummax() - 1

        wins = trade_returns[trade_returns > 0].sum()
        losses = -trade_returns[trade_returns < 0].sum()
        pf = wins / losses if losses > 0 else np.nan

        summary.append({
            "cost_per_side_pct": cost * 100,
            "trades": len(trades),
            "gross_win_rate_pct": (
                trades["gross_return_pct"] > 0
            ).mean() * 100,
            "net_win_rate_pct": (trade_returns > 0).mean() * 100,
            "avg_gross_trade_pct": trades["gross_return_pct"].mean(),
            "avg_net_trade_pct": trade_returns.mean() * 100,
            "compounded_net_return_pct": (equity.iloc[-1] - 1) * 100,
            "profit_factor": pf,
            "max_trade_equity_drawdown_pct": drawdown.min() * 100,
        })

        for n, (ret, eq) in enumerate(zip(trade_returns, equity), start=1):
            equity_data.append({
                "cost_per_side_pct": cost * 100,
                "trade_number": n,
                "net_trade_return_pct": ret * 100,
                "equity_index": eq,
            })

    print("\n=== TRADE LEDGER SUMMARY ===")
    print("Dataset UTC:", df["timestamps"].min(), "to", df["timestamps"].max())
    print("Trades:", len(trades))
    print("\nExit reasons:")
    print(trades["exit_reason"].value_counts().to_string())
    print("\nHolding time (minutes):")
    print(trades["holding_minutes"].describe().round(2).to_string())
    print("\nCost sensitivity (cost per side; illustrative):")
    print(pd.DataFrame(summary).round(4).to_string(index=False))

    print("\nFirst 10 trades:")
    print(trades.head(10).to_string(index=False))

    out.to_csv("data/btc/BTCUSDT_position_manager_actions_v2.csv", index=False)
    trades.to_csv(TRADES_OUT, index=False)
    pd.DataFrame(equity_data).to_csv(EQUITY_OUT, index=False)

    print("\nSaved actions: data/btc/BTCUSDT_position_manager_actions_v2.csv")
    print("Saved trades:", TRADES_OUT)
    print("Saved equity scenarios:", EQUITY_OUT)
    print("Cost scenarios are assumptions, not verified venue fees.")


if __name__ == "__main__":
    main()
