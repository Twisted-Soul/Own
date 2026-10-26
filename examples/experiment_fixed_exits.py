import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
TRADES_OUT = Path("data/btc/BTCUSDT_fixed_exit_trades.csv")
SUMMARY_OUT = Path("data/btc/BTCUSDT_fixed_exit_summary.csv")

TAKE_PROFITS = [0.10, 0.15, 0.20]  # percent
STOP_LOSSES = [0.10, 0.15, 0.20]   # percent
MAX_HOLD_BARS = 12

# Illustrative round-trip costs, expressed in percentage points.
ROUND_TRIP_COSTS = [0.0, 0.04, 0.14]


def load_data():
    df = pd.read_csv(INPUT)

    required = {"timestamps", "open", "high", "low", "close", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["signal"] = (
        df["signal"].fillna("HOLD").astype(str).str.upper().str.strip()
    )

    if not df["signal"].isin(["BUY", "SELL", "HOLD"]).all():
        raise ValueError("Unexpected signal values found.")

    df = (
        df.dropna(subset=[
            "timestamps", "open", "high", "low", "close"
        ])
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive OHLC prices found.")

    return df


def confirmed_entries(df):
    previous = df["signal"].shift(1)
    confirmed = df["signal"].where(
        df["signal"].isin(["BUY", "SELL"])
        & df["signal"].eq(previous),
        "HOLD",
    )

    prior_confirmed = confirmed.shift(1).fillna("HOLD")
    mask = (
        confirmed.isin(["BUY", "SELL"])
        & confirmed.ne(prior_confirmed)
    )

    entries = []
    for signal_index in df.index[mask]:
        entry_index = signal_index + 1
        if entry_index >= len(df):
            continue

        entries.append({
            "signal_index": signal_index,
            "entry_index": entry_index,
            "signal": confirmed.iloc[signal_index],
        })

    return entries


def simulate_trade(df, entry, tp_pct, sl_pct):
    entry_index = entry["entry_index"]
    direction = 1 if entry["signal"] == "BUY" else -1
    entry_price = float(df.at[entry_index, "open"])

    tp_fraction = tp_pct / 100
    sl_fraction = sl_pct / 100

    if direction == 1:
        target = entry_price * (1 + tp_fraction)
        stop = entry_price * (1 - sl_fraction)
    else:
        target = entry_price * (1 - tp_fraction)
        stop = entry_price * (1 + sl_fraction)

    # Inspect at most 12 candles, including the entry candle.
    last_bar = min(entry_index + MAX_HOLD_BARS - 1, len(df) - 1)

    exit_price = None
    exit_index = None
    reason = None

    for j in range(entry_index, last_bar + 1):
        o = float(df.at[j, "open"])
        h = float(df.at[j, "high"])
        l = float(df.at[j, "low"])

        # Handle gaps at the candle open before intrabar touches.
        if direction == 1:
            if o <= stop:
                exit_price, reason = o, "STOP_GAP"
            elif o >= target:
                exit_price, reason = o, "TARGET_GAP"
        else:
            if o >= stop:
                exit_price, reason = o, "STOP_GAP"
            elif o <= target:
                exit_price, reason = o, "TARGET_GAP"

        if reason is not None:
            exit_index = j
            break

        if direction == 1:
            stop_hit = l <= stop
            target_hit = h >= target
        else:
            stop_hit = h >= stop
            target_hit = l <= target

        # Conservative assumption when intrabar order is unknown.
        if stop_hit and target_hit:
            exit_price, reason = stop, "BOTH_HIT_STOP_FIRST"
        elif stop_hit:
            exit_price, reason = stop, "STOP_LOSS"
        elif target_hit:
            exit_price, reason = target, "TAKE_PROFIT"

        if reason is not None:
            exit_index = j
            break

    if exit_price is None:
        # Exit at the next open after the holding window when possible.
        scheduled_index = entry_index + MAX_HOLD_BARS

        if scheduled_index < len(df):
            exit_index = scheduled_index
            exit_price = float(df.at[exit_index, "open"])
            reason = "MAX_HOLD_TIME"
        else:
            # Not an executable fill: mark remaining position at final close.
            exit_index = len(df) - 1
            exit_price = float(df.at[exit_index, "close"])
            reason = "END_OF_DATA_MARK"

    gross_return_pct = direction * (
        exit_price / entry_price - 1
    ) * 100

    return {
        "signal_time": df.at[entry["signal_index"], "timestamps"],
        "entry_time": df.at[entry_index, "timestamps"],
        "exit_time": df.at[exit_index, "timestamps"],
        "signal": entry["signal"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
        "holding_bars": exit_index - entry_index,
        "exit_reason": reason,
        "gross_return_pct": gross_return_pct,
    }


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return gains / losses if losses > 0 else np.nan


def main():
    df = load_data()
    entries = confirmed_entries(df)

    if not entries:
        raise ValueError("No confirmed entry episodes found.")

    trades = []

    for tp in TAKE_PROFITS:
        for sl in STOP_LOSSES:
            for entry in entries:
                trade = simulate_trade(df, entry, tp, sl)
                trades.append(trade)

    ledger = pd.DataFrame(trades)

    # Cost sensitivity only uses completed trades.
    completed = ledger[
        ledger["exit_reason"] != "END_OF_DATA_MARK"
    ].copy()

    summaries = []

    for (tp, sl), group in completed.groupby(
        ["take_profit_pct", "stop_loss_pct"]
    ):
        for cost in ROUND_TRIP_COSTS:
            net = group["gross_return_pct"] - cost

            summaries.append({
                "take_profit_pct": tp,
                "stop_loss_pct": sl,
                "round_trip_cost_pct": cost,
                "trades": len(group),
                "gross_win_rate_pct": (
                    (group["gross_return_pct"] > 0).mean() * 100
                ),
                "net_win_rate_pct": (net > 0).mean() * 100,
                "mean_gross_return_pct": group["gross_return_pct"].mean(),
                "mean_net_return_pct": net.mean(),
                "net_profit_factor": profit_factor(net),
                "take_profit_exits": group["exit_reason"].isin([
                    "TAKE_PROFIT", "TARGET_GAP"
                ]).sum(),
                "stop_exits": group["exit_reason"].isin([
                    "STOP_LOSS", "STOP_GAP", "BOTH_HIT_STOP_FIRST"
                ]).sum(),
                "time_exits": (
                    group["exit_reason"] == "MAX_HOLD_TIME"
                ).sum(),
            })

    summary = pd.DataFrame(summaries)

    ledger.to_csv(TRADES_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n=== FIXED EXIT EXPERIMENT ===")
    print("Confirmed entry episodes:", len(entries))
    print("Completed trades:", len(completed))
    print("Maximum holding period:", MAX_HOLD_BARS * 5, "minutes")
    print("Same-candle TP/SL ambiguity: stop assumed first")
    print("Overlapping hypothetical trades are allowed.")
    print("Results are diagnostic, not a deployable backtest.")

    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    print("\nSaved ledger:", TRADES_OUT)
    print("Saved summary:", SUMMARY_OUT)


if __name__ == "__main__":
    main()
