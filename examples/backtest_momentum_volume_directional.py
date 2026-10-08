import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_momentum_volume_directional.csv")

STARTING_CAPITAL = 10_000.0

# 0.05% fee + 0.02% slippage per side
ROUND_TRIP_COST = 0.0014

VOLUME_THRESHOLD = 1.5
MOMENTUM_THRESHOLD = 0.002

# Test several holding periods.
HORIZONS = {
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
}

# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Loaded {len(df):,} candles")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")

# ============================================================
# SIGNALS
# ============================================================

long_signal = (
    (df["relative_volume"] >= VOLUME_THRESHOLD)
    & (df["return_1h"] <= -MOMENTUM_THRESHOLD)
)

short_signal = (
    (df["relative_volume"] >= VOLUME_THRESHOLD)
    & (df["return_1h"] >= MOMENTUM_THRESHOLD)
)

df["signal"] = 0
df.loc[long_signal, "signal"] = 1
df.loc[short_signal, "signal"] = -1

print()
print("Signal counts:")
print(f"Long signals:  {long_signal.sum():,}")
print(f"Short signals: {short_signal.sum():,}")

# ============================================================
# BACKTEST FUNCTION
# ============================================================

def backtest(horizon_bars):

    trades = []

    i = 0

    while i < len(df) - horizon_bars - 1:

        signal = df.at[i, "signal"]

        if signal == 0:
            i += 1
            continue

        # ----------------------------------------------------
        # Signal occurs at candle close.
        # Enter at NEXT candle open.
        # ----------------------------------------------------

        entry_index = i + 1
        exit_index = entry_index + horizon_bars - 1

        if exit_index >= len(df):
            break

        entry_price = df.at[entry_index, "open"]
        exit_price = df.at[exit_index, "close"]

        if signal == 1:
            raw_return = (exit_price / entry_price) - 1
            direction = "LONG"

        else:
            raw_return = (entry_price / exit_price) - 1
            direction = "SHORT"

        net_return = raw_return - ROUND_TRIP_COST

        capital_before = (
            STARTING_CAPITAL
            if not trades
            else trades[-1]["capital_after"]
        )

        capital_after = capital_before * (1 + net_return)

        trades.append({
            "signal_timestamp": df.at[i, "timestamps"],
            "entry_timestamp": df.at[entry_index, "timestamps"],
            "exit_timestamp": df.at[exit_index, "timestamps"],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return": raw_return,
            "net_return": net_return,
            "capital_before": capital_before,
            "capital_after": capital_after,
        })

        # ----------------------------------------------------
        # No overlapping trades.
        # ----------------------------------------------------

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# RUN ALL HORIZONS
# ============================================================

all_results = []

print()
print("=" * 75)
print("MOMENTUM + VOLUME DIRECTIONAL STRATEGY")
print("=" * 75)
print(f"Volume threshold:   {VOLUME_THRESHOLD}")
print(f"Momentum threshold:  {MOMENTUM_THRESHOLD:.2%}")
print(f"Round-trip cost:     {ROUND_TRIP_COST:.2%}")

for horizon_name, horizon_bars in HORIZONS.items():

    trades = backtest(horizon_bars)

    if trades.empty:
        print(f"\n{horizon_name}: no trades")
        continue

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_trades = len(trades)

    wins = (trades["net_return"] > 0).sum()
    win_rate = wins / total_trades

    avg_return = trades["net_return"].mean()
    median_return = trades["net_return"].median()

    final_capital = trades["capital_after"].iloc[-1]

    total_return = (
        final_capital / STARTING_CAPITAL
    ) - 1

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    equity = trades["capital_after"]

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Direction breakdown
    # --------------------------------------------------------

    long_trades = trades[trades["direction"] == "LONG"]
    short_trades = trades[trades["direction"] == "SHORT"]

    long_win = (
        (long_trades["net_return"] > 0).mean()
        if len(long_trades)
        else np.nan
    )

    short_win = (
        (short_trades["net_return"] > 0).mean()
        if len(short_trades)
        else np.nan
    )

    long_avg = (
        long_trades["net_return"].mean()
        if len(long_trades)
        else np.nan
    )

    short_avg = (
        short_trades["net_return"].mean()
        if len(short_trades)
        else np.nan
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    trades["horizon"] = horizon_name

    all_results.append(trades)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("-" * 75)
    print(horizon_name)
    print("-" * 75)

    print(f"Trades:          {total_trades:,}")
    print(f"Long trades:     {len(long_trades):,}")
    print(f"Short trades:    {len(short_trades):,}")
    print(f"Win rate:        {win_rate:.2%}")
    print(f"Average return:  {avg_return:.4%}")
    print(f"Median return:   {median_return:.4%}")
    print(f"Final capital:   ${final_capital:,.2f}")
    print(f"Total return:    {total_return:.2%}")
    print(f"Max drawdown:    {max_drawdown:.2%}")

    print()
    print("LONG")
    print(f"Trades:          {len(long_trades):,}")
    print(f"Win rate:        {long_win:.2%}")
    print(f"Average return:  {long_avg:.4%}")

    print()
    print("SHORT")
    print(f"Trades:          {len(short_trades):,}")
    print(f"Win rate:        {short_win:.2%}")
    print(f"Average return:  {short_avg:.4%}")

# ============================================================
# SAVE RESULTS
# ============================================================

if all_results:

    output = pd.concat(
        all_results,
        ignore_index=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 75)
    print("BACKTEST COMPLETE")
    print("=" * 75)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total trades across horizons: {len(output):,}")

else:

    print()
    print("No trades generated.")