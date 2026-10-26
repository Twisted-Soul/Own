import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
OUTPUT = Path("data/btc/BTCUSDT_5m_stability_experiment.csv")

# Illustrative costs per unit of position turnover.
# These are placeholders, NOT verified Truoux trading fees.
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002


def direct_flips(signals):
    prev = signals.shift(1)
    return int((
        prev.isin(["BUY", "SELL"])
        & signals.isin(["BUY", "SELL"])
        & prev.ne(signals)
    ).sum())


def active_direction_changes(signals):
    active = signals[signals.isin(["BUY", "SELL"])]
    if len(active) < 2:
        return 0
    return int(active.ne(active.shift()).sum() - 1)


def confirm_two_candles(raw):
    # Only accept a direction when the current and previous
    # raw signals agree. All other candles become HOLD.
    previous = raw.shift(1)
    confirmed = raw.where(
        raw.isin(["BUY", "SELL"]) & raw.eq(previous),
        "HOLD"
    )
    return confirmed


def evaluate(df, signal_col):
    close = pd.to_numeric(df["close"], errors="coerce")
    signals = df[signal_col].fillna("HOLD").astype(str).str.upper()

    # Signal at candle t is applied to the next candle's return.
    position = signals.map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    position = position.shift(1).fillna(0.0)

    market_return = close.pct_change().fillna(0.0)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])

    gross_return = position * market_return
    costs = turnover * (FEE_RATE + SLIPPAGE_RATE)
    net_return = gross_return - costs

    equity = (1.0 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    wins = net_return[net_return > 0].sum()
    losses = -net_return[net_return < 0].sum()
    profit_factor = wins / losses if losses > 0 else np.nan

    return {
        "candles": len(df),
        "BUY": int((signals == "BUY").sum()),
        "SELL": int((signals == "SELL").sum()),
        "HOLD": int((signals == "HOLD").sum()),
        "direct_flips": direct_flips(signals),
        "active_direction_changes": active_direction_changes(signals),
        "position_changes": int((turnover > 0).sum()),
        "gross_return_pct": (np.prod(1 + gross_return) - 1) * 100,
        "net_return_pct": (equity.iloc[-1] - 1) * 100,
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown.min() * 100,
    }


def main():
    df = pd.read_csv(INPUT)
    required = {"timestamps", "close", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamps", "close"])
    df = df.sort_values("timestamps").drop_duplicates(
        "timestamps"
    ).reset_index(drop=True)

    raw = df["signal"].fillna("HOLD").astype(str).str.upper()
    invalid = ~raw.isin(["BUY", "SELL", "HOLD"])
    if invalid.any():
        raise ValueError(
            "Unexpected signal values: "
            + str(raw[invalid].value_counts().to_dict())
        )

    df["baseline_signal"] = raw
    df["confirmed_signal"] = confirm_two_candles(raw)

    baseline = evaluate(df, "baseline_signal")
    confirmed = evaluate(df, "confirmed_signal")

    report = pd.DataFrame(
        [baseline, confirmed],
        index=["BASELINE", "CONFIRM_2_CANDLES"]
    )
    print("\n=== OFFLINE STABILITY EXPERIMENT ===")
    print(f"Data: {INPUT}")
    print(f"Period: {df['timestamps'].min()} to {df['timestamps'].max()}")
    print(f"Assumed fee + slippage: {(FEE_RATE + SLIPPAGE_RATE)*100:.3f}% per unit turnover")
    print("These cost assumptions are illustrative, not broker-verified.")
    print("\n", report.round(4).to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved comparison signals to: {OUTPUT}")


if __name__ == "__main__":
    main()
