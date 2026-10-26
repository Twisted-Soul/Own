import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
TRADES_OUT = Path("data/btc/BTCUSDT_next_open_trades.csv")
EQUITY_OUT = Path("data/btc/BTCUSDT_next_open_cost_sensitivity.csv")

MAX_HOLD_BARS = 12
COST_PER_SIDE = [0.0, 0.0002, 0.0007, 0.0010]


def confirm_entries(raw):
    previous = raw.shift(1)
    return raw.where(
        raw.isin(["BUY", "SELL"]) & raw.eq(previous),
        "HOLD"
    )


def simulate(df):
    signals = confirm_entries(df["signal"])
    position = 0
    entry_price = None
    entry_time = None
    entry_index = None
    trade_id = 0
    pending = None
    trades = []

    for i in range(len(df)):
        open_price = float(df["open"].iloc[i])
        close_price = float(df["close"].iloc[i])
        timestamp = df["timestamps"].iloc[i]
        exited_this_open = False

        # Execute decisions made at the PREVIOUS candle's close.
        if pending is not None:
            kind, direction, reason = pending
            pending = None

            if kind == "ENTRY" and position == 0:
                position = direction
                entry_price = open_price
                entry_time = timestamp
                entry_index = i
                trade_id += 1

            elif kind == "EXIT" and position != 0:
                gross = position * (open_price / entry_price - 1.0)
                trades.append({
                    "trade_id": trade_id,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": open_price,
                    "holding_bars": i - entry_index,
                    "holding_minutes": (i - entry_index) * 5,
                    "exit_reason": reason,
                    "gross_return_pct": gross * 100.0,
                })
                position = 0
                entry_price = None
                entry_time = None
                entry_index = None
                exited_this_open = True

        # Only decisions from completed candles can schedule
        # execution at the NEXT candle's open.
        if i < len(df) - 1:
            sig = signals.iloc[i]

            if position != 0:
                held_bars = i - entry_index
                opposite = (
                    (position == 1 and sig == "SELL")
                    or (position == -1 and sig == "BUY")
                )

                if opposite or held_bars >= MAX_HOLD_BARS - 1:
                    reason = (
                        "CONFIRMED_OPPOSITE_SIGNAL"
                        if opposite else "MAX_HOLD_TIME"
                    )
                    pending = ("EXIT", 0, reason)

            elif not exited_this_open and sig in ("BUY", "SELL"):
                direction = 1 if sig == "BUY" else -1
                pending = ("ENTRY", direction, "")

        # If the dataset ends with an open trade, mark it at the
        # final close. This is a valuation, not an executable fill.
        if i == len(df) - 1 and position != 0:
            gross = position * (close_price / entry_price - 1.0)
            trades.append({
                "trade_id": trade_id,
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_time": entry_time,
                "exit_time": timestamp,
                "entry_index": entry_index,
                "exit_index": i,
                "entry_price": entry_price,
                "exit_price": close_price,
                "holding_bars": i - entry_index,
                "holding_minutes": (i - entry_index) * 5,
                "exit_reason": "END_OF_DATA_MARK",
                "gross_return_pct": gross * 100.0,
            })

    return pd.DataFrame(trades)


def main():
    df = pd.read_csv(INPUT)
    required = {"timestamps", "open", "close", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )
    for col in ["open", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["timestamps", "open", "close"])
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )

    df["signal"] = df["signal"].fillna("HOLD").astype(str).str.upper()
    if not df["signal"].isin(["BUY", "SELL", "HOLD"]).all():
        raise ValueError("Unexpected signal values found.")

    trades = simulate(df)
    if trades.empty:
        raise ValueError("No trades generated.")

    print("\n=== NEXT-OPEN EXECUTION EXPERIMENT ===")
    print("Execution: next candle open after signal confirmation")
    print("Maximum holding period:", MAX_HOLD_BARS, "bars")
    print("Trades:", len(trades))
    print("\nExit reasons:")
    print(trades["exit_reason"].value_counts().to_string())

    results = []
    for cost in COST_PER_SIDE:
        returns = trades["gross_return_pct"] / 100 - 2 * cost
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1

        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()

        results.append({
            "cost_per_side_pct": cost * 100,
            "trades": len(trades),
            "gross_win_rate_pct": (
                trades["gross_return_pct"] > 0
            ).mean() * 100,
            "net_win_rate_pct": (returns > 0).mean() * 100,
            "avg_gross_trade_pct": trades["gross_return_pct"].mean(),
            "avg_net_trade_pct": returns.mean() * 100,
            "compounded_net_return_pct": (equity.iloc[-1] - 1) * 100,
            "profit_factor": gains / losses if losses > 0 else np.nan,
            "max_trade_equity_drawdown_pct": drawdown.min() * 100,
        })

    print("\n=== COST SENSITIVITY ===")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print("\n=== FIRST 10 TRADES ===")
    print(trades.head(10).to_string(index=False))

    trades.to_csv(TRADES_OUT, index=False)
    pd.DataFrame(results).to_csv(EQUITY_OUT, index=False)
    print("\nSaved trades:", TRADES_OUT)
    print("Saved cost sensitivity:", EQUITY_OUT)
    print("Costs remain illustrative, not venue-verified.")


if __name__ == "__main__":
    main()
