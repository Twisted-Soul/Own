import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/btc/BTCUSDT_5m_1y_features.csv"
OUTPUT_FILE = "data/btc/BTCUSDT_5m_rsi_reversal_confirmation.csv"

STARTING_CAPITAL = 10_000.0

# 0.05% fee per side + 0.02% slippage per side
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)

# Holding periods in 5-minute candles
HORIZONS = {
    "5m": 1,
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

# Make sure required columns exist
required_columns = [
    "timestamps",
    "open",
    "close",
    "rsi_14",
    "relative_volume",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# CREATE REVERSAL SIGNALS
# ============================================================

print("\nCreating RSI reversal confirmation signals...")

# Previous candle RSI
df["previous_rsi"] = df["rsi_14"].shift(1)

# Long:
# Previous candle was oversold
# Current candle has recovered back to >= 30
# Current candle has meaningful volume
df["long_signal"] = (
    (df["previous_rsi"] < 30)
    & (df["rsi_14"] >= 30)
    & (df["relative_volume"] > 1.5)
)

# Short:
# Previous candle was overbought
# Current candle has fallen back to <= 70
# Current candle has meaningful volume
df["short_signal"] = (
    (df["previous_rsi"] > 70)
    & (df["rsi_14"] <= 70)
    & (df["relative_volume"] > 1.5)
)

print(f"Long signals:  {df['long_signal'].sum():,}")
print(f"Short signals: {df['short_signal'].sum():,}")


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def backtest(horizon_name, bars):
    """
    Signal is generated at candle i close.

    Entry:
        candle i+1 OPEN

    Exit:
        close of the candle after 'bars' completed
        from the entry candle.

    Example:
        5m  = entry next candle open -> same candle close
        15m = entry next candle open -> close 3 candles later
    """

    trades = []

    capital = STARTING_CAPITAL

    i = 1

    while i < len(df) - bars:

        # ----------------------------------------------------
        # Look for a signal on candle i
        # ----------------------------------------------------

        if df.loc[i, "long_signal"]:
            side = "LONG"

        elif df.loc[i, "short_signal"]:
            side = "SHORT"

        else:
            i += 1
            continue

        # ----------------------------------------------------
        # Entry happens on NEXT candle open
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(df):
            break

        entry_price = df.loc[entry_index, "open"]

        # ----------------------------------------------------
        # Exit after the selected holding period
        # ----------------------------------------------------

        exit_index = entry_index + bars - 1

        if exit_index >= len(df):
            break

        exit_price = df.loc[exit_index, "close"]

        # ----------------------------------------------------
        # Calculate gross return
        # ----------------------------------------------------

        if side == "LONG":
            gross_return = (exit_price / entry_price) - 1

        else:
            gross_return = (entry_price / exit_price) - 1

        # ----------------------------------------------------
        # Apply trading costs
        # ----------------------------------------------------

        net_return = gross_return - ROUND_TRIP_COST

        # Dollar P/L assuming full capital is allocated
        pnl = capital * net_return

        capital += pnl

        trades.append({
            "signal_time": df.loc[i, "timestamps"],
            "entry_time": df.loc[entry_index, "timestamps"],
            "exit_time": df.loc[exit_index, "timestamps"],
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": net_return,
            "pnl": pnl,
            "capital": capital,
        })

        # ----------------------------------------------------
        # Prevent overlapping trades
        # ----------------------------------------------------

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# RUN BACKTESTS
# ============================================================

all_results = []

print("\n" + "=" * 70)
print("RSI REVERSAL CONFIRMATION BACKTEST")
print("=" * 70)

print(f"Starting capital: ${STARTING_CAPITAL:,.2f}")
print(f"Round-trip cost:  {ROUND_TRIP_COST * 100:.2f}%")

for horizon_name, bars in HORIZONS.items():

    print("\n" + "-" * 70)
    print(f"HORIZON: {horizon_name}")
    print("-" * 70)

    results = backtest(horizon_name, bars)

    if results.empty:
        print("No trades.")
        continue

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_trades = len(results)

    long_trades = (results["side"] == "LONG").sum()
    short_trades = (results["side"] == "SHORT").sum()

    wins = (results["net_return"] > 0).sum()

    win_rate = wins / total_trades

    avg_return = results["net_return"].mean()
    median_return = results["net_return"].median()

    total_return = (results["capital"].iloc[-1] / STARTING_CAPITAL) - 1

    final_capital = results["capital"].iloc[-1]

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    equity = results["capital"]

    running_max = equity.cummax()

    drawdown = (equity / running_max) - 1

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Long / short statistics
    # --------------------------------------------------------

    long_results = results[results["side"] == "LONG"]
    short_results = results[results["side"] == "SHORT"]

    long_avg = (
        long_results["net_return"].mean()
        if not long_results.empty
        else np.nan
    )

    short_avg = (
        short_results["net_return"].mean()
        if not short_results.empty
        else np.nan
    )

    long_win = (
        (long_results["net_return"] > 0).mean()
        if not long_results.empty
        else np.nan
    )

    short_win = (
        (short_results["net_return"] > 0).mean()
        if not short_results.empty
        else np.nan
    )

    print(f"Trades:          {total_trades:,}")
    print(f"Long trades:     {long_trades:,}")
    print(f"Short trades:    {short_trades:,}")
    print(f"Win rate:        {win_rate * 100:.2f}%")
    print(f"Average return:  {avg_return * 100:.4f}%")
    print(f"Median return:   {median_return * 100:.4f}%")
    print(f"Final capital:   ${final_capital:,.2f}")
    print(f"Total return:    {total_return * 100:.2f}%")
    print(f"Max drawdown:    {max_drawdown * 100:.2f}%")

    print("\nLong:")
    print(f"  Win rate:      {long_win * 100:.2f}%")
    print(f"  Avg return:    {long_avg * 100:.4f}%")

    print("\nShort:")
    print(f"  Win rate:      {short_win * 100:.2f}%")
    print(f"  Avg return:    {short_avg * 100:.4f}%")

    # Add horizon to every trade
    results["horizon"] = horizon_name

    all_results.append(results)


# ============================================================
# SAVE RESULTS
# ============================================================

if all_results:

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    final_results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total recorded trades: {len(final_results):,}")

else:

    print("\nNo trades were generated.")


print("\nDone.")