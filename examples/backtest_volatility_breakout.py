import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_volatility_breakout.csv")

STARTING_CAPITAL = 10_000.0

# 0.05% fee + 0.02% slippage per side
ROUND_TRIP_COST = 0.0014

# Volatility compression:
# current volatility must be below this rolling percentile
VOL_LOOKBACK = 288       # 24 hours of 5m candles
VOL_PERCENTILE = 25

# Breakout lookback
BREAKOUT_LOOKBACK = 20   # 100 minutes

# Expansion requirement
EXPANSION_MULTIPLIER = 1.20

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

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

df = (
    df
    .sort_values("timestamps")
    .reset_index(drop=True)
)

print(f"Loaded {len(df):,} candles")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")

# ============================================================
# BUILD VOLATILITY BASELINE
# ============================================================

print()
print("Building volatility compression/expansion signals...")

# Previous volatility is used so the current candle does not
# influence its own compression threshold.

df["volatility_threshold"] = (
    df["volatility_20"]
    .rolling(
        VOL_LOOKBACK,
        min_periods=VOL_LOOKBACK
    )
    .quantile(VOL_PERCENTILE / 100)
    .shift(1)
)

df["compressed"] = (
    df["volatility_20"]
    <= df["volatility_threshold"]
)

# Previous volatility used for expansion comparison.
df["previous_volatility"] = (
    df["volatility_20"]
    .shift(1)
)

df["volatility_expanding"] = (
    df["volatility_20"]
    >= (
        df["previous_volatility"]
        * EXPANSION_MULTIPLIER
    )
)

# ============================================================
# BREAKOUT LEVELS
# ============================================================

# Shift by one candle so the current candle cannot be part
# of the breakout level.

df["breakout_high"] = (
    df["high"]
    .rolling(
        BREAKOUT_LOOKBACK,
        min_periods=BREAKOUT_LOOKBACK
    )
    .max()
    .shift(1)
)

df["breakout_low"] = (
    df["low"]
    .rolling(
        BREAKOUT_LOOKBACK,
        min_periods=BREAKOUT_LOOKBACK
    )
    .min()
    .shift(1)
)

# ============================================================
# BREAKOUT CONDITIONS
# ============================================================

long_breakout = (
    df["close"] > df["breakout_high"]
)

short_breakout = (
    df["close"] < df["breakout_low"]
)

# Require the candle immediately before the breakout to have
# been compressed. This gives us:
#
# compression -> breakout
#
# rather than simply trading every existing trend.

df["signal"] = 0

df.loc[
    df["compressed"].shift(1)
    & df["volatility_expanding"]
    & long_breakout,
    "signal"
] = 1

df.loc[
    df["compressed"].shift(1)
    & df["volatility_expanding"]
    & short_breakout,
    "signal"
] = -1

# ============================================================
# SIGNAL COUNTS
# ============================================================

print()
print("Signal counts:")

print(
    f"Long signals:  {(df['signal'] == 1).sum():,}"
)

print(
    f"Short signals: {(df['signal'] == -1).sum():,}"
)

print(
    f"Compressed candles: {df['compressed'].sum():,}"
)

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

        # Signal confirmed at current candle close.
        #
        # Execution occurs at next candle open.

        entry_index = i + 1
        exit_index = entry_index + horizon_bars - 1

        if exit_index >= len(df):
            break

        entry_price = df.at[entry_index, "open"]
        exit_price = df.at[exit_index, "close"]

        if signal == 1:

            direction = "LONG"

            gross_return = (
                exit_price / entry_price
            ) - 1

        else:

            direction = "SHORT"

            gross_return = (
                entry_price / exit_price
            ) - 1

        net_return = (
            gross_return
            - ROUND_TRIP_COST
        )

        if not trades:

            capital_before = STARTING_CAPITAL

        else:

            capital_before = (
                trades[-1]["capital_after"]
            )

        capital_after = (
            capital_before
            * (1 + net_return)
        )

        trades.append({
            "signal_timestamp":
                df.at[i, "timestamps"],

            "entry_timestamp":
                df.at[entry_index, "timestamps"],

            "exit_timestamp":
                df.at[exit_index, "timestamps"],

            "direction":
                direction,

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "gross_return":
                gross_return,

            "net_return":
                net_return,

            "capital_before":
                capital_before,

            "capital_after":
                capital_after,
        })

        # No overlapping positions.

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# RUN BACKTESTS
# ============================================================

all_results = []

print()
print("=" * 80)
print("VOLATILITY COMPRESSION + BREAKOUT STRATEGY")
print("=" * 80)

print(
    f"Volatility lookback:       "
    f"{VOL_LOOKBACK} candles"
)

print(
    f"Compression percentile:     "
    f"{VOL_PERCENTILE}"
)

print(
    f"Breakout lookback:          "
    f"{BREAKOUT_LOOKBACK} candles"
)

print(
    f"Expansion multiplier:       "
    f"{EXPANSION_MULTIPLIER:.2f}x"
)

print(
    f"Round-trip cost:             "
    f"{ROUND_TRIP_COST:.2%}"
)

# ============================================================
# EACH HORIZON
# ============================================================

for horizon_name, horizon_bars in HORIZONS.items():

    trades = backtest(horizon_bars)

    if trades.empty:

        print()
        print(f"{horizon_name}: NO TRADES")
        continue

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total_trades = len(trades)

    wins = (
        trades["net_return"] > 0
    ).sum()

    win_rate = (
        wins / total_trades
    )

    gross_avg = (
        trades["gross_return"].mean()
    )

    net_avg = (
        trades["net_return"].mean()
    )

    median_net = (
        trades["net_return"].median()
    )

    final_capital = (
        trades["capital_after"].iloc[-1]
    )

    total_return = (
        final_capital
        / STARTING_CAPITAL
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
    # Direction
    # --------------------------------------------------------

    longs = trades[
        trades["direction"] == "LONG"
    ]

    shorts = trades[
        trades["direction"] == "SHORT"
    ]

    long_win = (
        (longs["net_return"] > 0).mean()
        if len(longs)
        else np.nan
    )

    short_win = (
        (shorts["net_return"] > 0).mean()
        if len(shorts)
        else np.nan
    )

    long_avg = (
        longs["net_return"].mean()
        if len(longs)
        else np.nan
    )

    short_avg = (
        shorts["net_return"].mean()
        if len(shorts)
        else np.nan
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    trades["horizon"] = horizon_name

    all_results.append(trades)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(horizon_name)
    print("-" * 80)

    print(
        f"Trades:          {total_trades:,}"
    )

    print(
        f"Long trades:     {len(longs):,}"
    )

    print(
        f"Short trades:    {len(shorts):,}"
    )

    print(
        f"Win rate:        {win_rate:.2%}"
    )

    print(
        f"Average gross:   {gross_avg:.4%}"
    )

    print(
        f"Average net:     {net_avg:.4%}"
    )

    print(
        f"Median net:      {median_net:.4%}"
    )

    print(
        f"Final capital:   ${final_capital:,.2f}"
    )

    print(
        f"Total return:    {total_return:.2%}"
    )

    print(
        f"Max drawdown:    {max_drawdown:.2%}"
    )

    print()
    print("LONG")

    print(
        f"Trades:          {len(longs):,}"
    )

    print(
        f"Win rate:        {long_win:.2%}"
    )

    print(
        f"Average net:     {long_avg:.4%}"
    )

    print()
    print("SHORT")

    print(
        f"Trades:          {len(shorts):,}"
    )

    print(
        f"Win rate:        {short_win:.2%}"
    )

    print(
        f"Average net:     {short_avg:.4%}"
    )

# ============================================================
# SAVE
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
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Total trades across horizons: "
        f"{len(output):,}"
    )

else:

    print()
    print("No trades generated.")