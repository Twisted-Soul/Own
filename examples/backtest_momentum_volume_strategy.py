import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/btc/BTCUSDT_5m_1y_features.csv"
OUTPUT_FILE = "data/btc/BTCUSDT_5m_momentum_volume_strategy.csv"

STARTING_CAPITAL = 10_000.0

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)

# Signal thresholds
VOLUME_THRESHOLD = 1.5
MOMENTUM_THRESHOLD = 0.002

# Risk parameters
STOP_LOSS = 0.003
TAKE_PROFIT = 0.006

# Maximum holding period
MAX_HOLDING_BARS = 48


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

df = df.sort_values(
    "timestamps"
).reset_index(drop=True)

print(f"Loaded {len(df):,} candles")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")


# ============================================================
# VALIDATE
# ============================================================

required = [
    "timestamps",
    "open",
    "high",
    "low",
    "close",
    "relative_volume",
    "return_1h",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# SIGNALS
# ============================================================

df["long_signal"] = (
    (df["relative_volume"] >= VOLUME_THRESHOLD)
    &
    (df["return_1h"] <= -MOMENTUM_THRESHOLD)
)

df["short_signal"] = (
    (df["relative_volume"] >= VOLUME_THRESHOLD)
    &
    (df["return_1h"] >= MOMENTUM_THRESHOLD)
)

print("\nSignal counts:")
print(f"Long signals:  {df['long_signal'].sum():,}")
print(f"Short signals: {df['short_signal'].sum():,}")


# ============================================================
# BACKTEST
# ============================================================

def backtest():

    trades = []

    capital = STARTING_CAPITAL

    i = 0

    while i < len(df) - 2:

        # ----------------------------------------------------
        # Signal at candle close
        # ----------------------------------------------------

        if df.loc[i, "long_signal"]:
            side = "LONG"

        elif df.loc[i, "short_signal"]:
            side = "SHORT"

        else:
            i += 1
            continue

        # ----------------------------------------------------
        # Entry at next candle OPEN
        # ----------------------------------------------------

        entry_index = i + 1

        entry_price = df.loc[
            entry_index,
            "open"
        ]

        if side == "LONG":

            stop_price = (
                entry_price *
                (1 - STOP_LOSS)
            )

            target_price = (
                entry_price *
                (1 + TAKE_PROFIT)
            )

        else:

            stop_price = (
                entry_price *
                (1 + STOP_LOSS)
            )

            target_price = (
                entry_price *
                (1 - TAKE_PROFIT)
            )

        # ----------------------------------------------------
        # Search for exit
        # ----------------------------------------------------

        exit_index = None
        exit_price = None
        exit_reason = None

        last_index = min(
            entry_index + MAX_HOLDING_BARS - 1,
            len(df) - 1
        )

        for j in range(
            entry_index,
            last_index + 1
        ):

            high = df.loc[j, "high"]
            low = df.loc[j, "low"]

            if side == "LONG":

                hit_stop = low <= stop_price
                hit_target = high >= target_price

                if hit_stop and hit_target:
                    # Conservative assumption:
                    # if both occur in the same candle,
                    # assume the stop was hit first.
                    exit_price = stop_price
                    exit_reason = "STOP"

                elif hit_stop:

                    exit_price = stop_price
                    exit_reason = "STOP"

                elif hit_target:

                    exit_price = target_price
                    exit_reason = "TARGET"

            else:

                hit_stop = high >= stop_price
                hit_target = low <= target_price

                if hit_stop and hit_target:

                    exit_price = stop_price
                    exit_reason = "STOP"

                elif hit_stop:

                    exit_price = stop_price
                    exit_reason = "STOP"

                elif hit_target:

                    exit_price = target_price
                    exit_reason = "TARGET"

            if exit_price is not None:

                exit_index = j
                break

        # ----------------------------------------------------
        # If neither target nor stop was hit,
        # exit at final candle close.
        # ----------------------------------------------------

        if exit_index is None:

            exit_index = last_index

            exit_price = df.loc[
                exit_index,
                "close"
            ]

            exit_reason = "TIME"


        # ----------------------------------------------------
        # Gross return
        # ----------------------------------------------------

        if side == "LONG":

            gross_return = (
                exit_price /
                entry_price
            ) - 1

        else:

            gross_return = (
                entry_price /
                exit_price
            ) - 1


        # ----------------------------------------------------
        # Costs
        # ----------------------------------------------------

        net_return = (
            gross_return -
            ROUND_TRIP_COST
        )

        pnl = capital * net_return

        capital += pnl


        trades.append({
            "signal_time":
                df.loc[i, "timestamps"],

            "entry_time":
                df.loc[entry_index, "timestamps"],

            "exit_time":
                df.loc[exit_index, "timestamps"],

            "side":
                side,

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "stop_price":
                stop_price,

            "target_price":
                target_price,

            "exit_reason":
                exit_reason,

            "gross_return":
                gross_return,

            "net_return":
                net_return,

            "pnl":
                pnl,

            "capital":
                capital,

            "holding_bars":
                exit_index - entry_index + 1,
        })


        # ----------------------------------------------------
        # Prevent overlapping trades
        # ----------------------------------------------------

        i = exit_index + 1


    return pd.DataFrame(trades)


# ============================================================
# RUN
# ============================================================

print("\n")
print("=" * 75)
print("MOMENTUM + VOLUME STRATEGY")
print("=" * 75)

print(
    f"Volume threshold: "
    f"{VOLUME_THRESHOLD}"
)

print(
    f"Momentum threshold: "
    f"{MOMENTUM_THRESHOLD * 100:.2f}%"
)

print(
    f"Stop loss: "
    f"{STOP_LOSS * 100:.2f}%"
)

print(
    f"Take profit: "
    f"{TAKE_PROFIT * 100:.2f}%"
)

print(
    f"Max holding: "
    f"{MAX_HOLDING_BARS * 5} minutes"
)

print(
    f"Round-trip cost: "
    f"{ROUND_TRIP_COST * 100:.2f}%"
)


results = backtest()


# ============================================================
# RESULTS
# ============================================================

if results.empty:

    print("\nNo trades generated.")

else:

    total_trades = len(results)

    long_results = results[
        results["side"] == "LONG"
    ]

    short_results = results[
        results["side"] == "SHORT"
    ]

    wins = (
        results["net_return"] > 0
    ).sum()

    win_rate = (
        wins / total_trades
    )

    avg_return = (
        results["net_return"].mean()
    )

    median_return = (
        results["net_return"].median()
    )

    final_capital = (
        results["capital"].iloc[-1]
    )

    total_return = (
        final_capital /
        STARTING_CAPITAL
    ) - 1

    running_max = (
        results["capital"]
        .cummax()
    )

    drawdown = (
        results["capital"] /
        running_max
    ) - 1

    max_drawdown = drawdown.min()

    print("\n" + "-" * 75)
    print("OVERALL")
    print("-" * 75)

    print(
        f"Trades:          "
        f"{total_trades:,}"
    )

    print(
        f"Long trades:     "
        f"{len(long_results):,}"
    )

    print(
        f"Short trades:    "
        f"{len(short_results):,}"
    )

    print(
        f"Win rate:        "
        f"{win_rate * 100:.2f}%"
    )

    print(
        f"Average return:  "
        f"{avg_return * 100:.4f}%"
    )

    print(
        f"Median return:   "
        f"{median_return * 100:.4f}%"
    )

    print(
        f"Final capital:   "
        f"${final_capital:,.2f}"
    )

    print(
        f"Total return:    "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Max drawdown:    "
        f"{max_drawdown * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Long
    # --------------------------------------------------------

    print("\nLONG")

    if not long_results.empty:

        print(
            f"Trades:          "
            f"{len(long_results):,}"
        )

        print(
            f"Win rate:        "
            f"{(long_results['net_return'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"Average return:  "
            f"{long_results['net_return'].mean() * 100:.4f}%"
        )

    # --------------------------------------------------------
    # Short
    # --------------------------------------------------------

    print("\nSHORT")

    if not short_results.empty:

        print(
            f"Trades:          "
            f"{len(short_results):,}"
        )

        print(
            f"Win rate:        "
            f"{(short_results['net_return'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"Average return:  "
            f"{short_results['net_return'].mean() * 100:.4f}%"
        )


    # --------------------------------------------------------
    # Exit reasons
    # --------------------------------------------------------

    print("\nEXIT REASONS")

    exit_counts = (
        results["exit_reason"]
        .value_counts()
    )

    for reason, count in exit_counts.items():

        percentage = (
            count /
            total_trades *
            100
        )

        print(
            f"{reason:8s}: "
            f"{count:6,d} "
            f"({percentage:5.2f}%)"
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 75)
    print("BACKTEST COMPLETE")
    print("=" * 75)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Total trades: "
        f"{len(results):,}"
    )


print("\nDone.")