import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
OUTPUT = Path("data/btc/BTCUSDT_5m_position_manager_experiment.csv")

# Illustrative assumptions only. Replace with verified venue costs.
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_HOLD_BARS = 12  # 12 x 5 minutes = 1 hour


def confirm_entries(raw):
    previous = raw.shift(1)
    return raw.where(
        raw.isin(["BUY", "SELL"]) & raw.eq(previous),
        "HOLD"
    )


def simulate_position_manager(df):
    raw = df["signal"].astype(str).str.upper()
    confirmed = confirm_entries(raw)

    position = 0
    entry_index = None
    entry_price = None

    actions = []
    positions = []
    exit_reasons = []
    entry_prices = []

    for i in range(len(df)):
        signal = confirmed.iloc[i]
        action = "HOLD"
        reason = ""
        current_close = float(df["close"].iloc[i])

        if position == 0:
            # Only enter when flat and a confirmed entry is available.
            if signal == "BUY":
                position = 1
                entry_index = i
                entry_price = current_close
                action = "BUY"
            elif signal == "SELL":
                position = -1
                entry_index = i
                entry_price = current_close
                action = "SELL"

        else:
            held_bars = i - entry_index
            opposite_signal = (
                (position == 1 and signal == "SELL")
                or (position == -1 and signal == "BUY")
            )

            # EXIT only; never reverse on the same candle.
            if opposite_signal:
                action = "EXIT"
                reason = "CONFIRMED_OPPOSITE_SIGNAL"
                position = 0
                entry_index = None
                entry_price = None

            elif held_bars >= MAX_HOLD_BARS:
                action = "EXIT"
                reason = "MAX_HOLD_TIME"
                position = 0
                entry_index = None
                entry_price = None

        actions.append(action)
        positions.append(position)
        exit_reasons.append(reason)
        entry_prices.append(entry_price)

    result = df.copy()
    result["confirmed_entry_signal"] = confirmed
    result["action"] = actions
    result["position_at_close"] = positions
    result["exit_reason"] = exit_reasons
    result["entry_price_reference"] = entry_prices
    return result


def evaluate(result):
    close = pd.to_numeric(result["close"], errors="coerce")
    pos = result["position_at_close"].astype(float)

    # A position decided at candle close earns the following
    # close-to-close return, avoiding use of that candle's return
    # before the signal existed.
    market_return = close.pct_change().fillna(0.0)
    held_position = pos.shift(1).fillna(0.0)

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])

    gross = held_position * market_return
    costs = turnover * (FEE_RATE + SLIPPAGE_RATE)
    net = gross - costs

    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = gains / losses if losses > 0 else np.nan

    actions = result["action"]
    return {
        "candles": len(result),
        "BUY_entries": int((actions == "BUY").sum()),
        "SELL_entries": int((actions == "SELL").sum()),
        "EXIT_actions": int((actions == "EXIT").sum()),
        "signal_HOLDs": int((actions == "HOLD").sum()),
        "opposite_signal_exits": int(
            (result["exit_reason"] == "CONFIRMED_OPPOSITE_SIGNAL").sum()
        ),
        "time_exits": int(
            (result["exit_reason"] == "MAX_HOLD_TIME").sum()
        ),
        "gross_return_pct": (np.prod(1.0 + gross) - 1.0) * 100.0,
        "net_return_pct": (equity.iloc[-1] - 1.0) * 100.0,
        "profit_factor": pf,
        "max_drawdown_pct": drawdown.min() * 100.0,
    }


def main():
    df = pd.read_csv(INPUT)

    required = {"timestamps", "open", "high", "low", "close", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
            f"Unexpected signal values: {df.loc[invalid, 'signal'].unique()}"
        )

    result = simulate_position_manager(df)
    metrics = evaluate(result)

    print("\n=== KRONOS POSITION-MANAGER EXPERIMENT ===")
    print("Period UTC:", result["timestamps"].min(), "to",
          result["timestamps"].max())
    print("Entry confirmation: 2 consecutive directional signals")
    print("Exit rules: opposite confirmed signal or max hold")
    print("Maximum hold:", MAX_HOLD_BARS, "candles")
    print("Assumed fee + slippage per unit turnover:",
          f"{(FEE_RATE + SLIPPAGE_RATE) * 100:.3f}%")
    print("\n", pd.Series(metrics).round(4).to_string())

    print("\n=== RECENT ACTIONS ===")
    cols = [
        "timestamps", "close", "signal",
        "confirmed_entry_signal", "action",
        "position_at_close", "exit_reason"
    ]
    print(result[cols].tail(30).to_string(index=False))

    result.to_csv(OUTPUT, index=False)
    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
