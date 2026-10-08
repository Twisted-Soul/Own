import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/btc/BTCUSDT_5m_1y_features.csv"
OUTPUT_FILE = "data/btc/BTCUSDT_5m_regime_strategy.csv"

STARTING_CAPITAL = 10_000.0

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)

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

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Loaded {len(df):,} candles")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "timestamps",
    "open",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "return_1h",
    "volatility_20",
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
# WALKING VOLATILITY THRESHOLD
# ============================================================
#
# IMPORTANT:
# We do NOT use the full dataset median.
#
# The threshold at candle i is calculated only from
# volatility values available BEFORE candle i.
#
# This avoids future information leaking into the signal.
# ============================================================

print("\nCalculating rolling volatility regime...")

VOLATILITY_LOOKBACK = 5000

df["volatility_threshold"] = (
    df["volatility_20"]
    .rolling(
        VOLATILITY_LOOKBACK,
        min_periods=1000
    )
    .median()
)

df["high_volatility"] = (
    df["volatility_20"]
    > df["volatility_threshold"]
)


# ============================================================
# MARKET REGIME
# ============================================================

df["trend"] = np.select(
    [
        (
            (df["ema_20"] < df["ema_50"])
            &
            (df["ema_50"] < df["ema_200"])
        ),

        (
            (df["ema_20"] > df["ema_50"])
            &
            (df["ema_50"] > df["ema_200"])
        ),
    ],
    [
        "BEAR",
        "BULL",
    ],
    default="MIXED"
)


# ============================================================
# MOMENTUM REGIME
# ============================================================

df["momentum"] = np.select(
    [
        df["return_1h"] > 0.002,
        df["return_1h"] < -0.002,
    ],
    [
        "POSITIVE",
        "NEGATIVE",
    ],
    default="NEUTRAL"
)


# ============================================================
# STRATEGY SIGNALS
# ============================================================

# ------------------------------------------------------------
# LONG REGIME
#
# Bearish trend
# + High volatility
# + Neutral momentum
#
# This is the regime that showed interesting historical
# behavior in our analysis.
# ------------------------------------------------------------

df["long_signal"] = (
    (df["trend"] == "BEAR")
    &
    df["high_volatility"]
    &
    (df["momentum"] == "NEUTRAL")
)


# ------------------------------------------------------------
# SHORT MIRROR
#
# Bullish trend
# + High volatility
# + Neutral momentum
#
# We test this separately rather than assuming symmetry.
# ------------------------------------------------------------

df["short_signal"] = (
    (df["trend"] == "BULL")
    &
    df["high_volatility"]
    &
    (df["momentum"] == "NEUTRAL")
)


print("\nSignal counts:")

print(
    f"Long signals:  "
    f"{df['long_signal'].sum():,}"
)

print(
    f"Short signals: "
    f"{df['short_signal'].sum():,}"
)


# ============================================================
# BACKTEST
# ============================================================

def backtest(bars):

    trades = []

    capital = STARTING_CAPITAL

    i = 0

    while i < len(df) - bars - 1:

        # ----------------------------------------------------
        # Signal is generated at candle i CLOSE.
        # ----------------------------------------------------

        if df.loc[i, "long_signal"]:
            side = "LONG"

        elif df.loc[i, "short_signal"]:
            side = "SHORT"

        else:
            i += 1
            continue

        # ----------------------------------------------------
        # Entry at NEXT candle OPEN.
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(df):
            break

        entry_price = df.loc[
            entry_index,
            "open"
        ]

        # ----------------------------------------------------
        # Exit after N candles.
        # ----------------------------------------------------

        exit_index = entry_index + bars - 1

        if exit_index >= len(df):
            break

        exit_price = df.loc[
            exit_index,
            "close"
        ]

        # ----------------------------------------------------
        # Gross return
        # ----------------------------------------------------

        if side == "LONG":

            gross_return = (
                exit_price / entry_price
            ) - 1

        else:

            gross_return = (
                entry_price / exit_price
            ) - 1

        # ----------------------------------------------------
        # Trading costs
        # ----------------------------------------------------

        net_return = (
            gross_return
            - ROUND_TRIP_COST
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

            "trend":
                df.loc[i, "trend"],

            "momentum":
                df.loc[i, "momentum"],

            "volatility":
                df.loc[i, "volatility_20"],

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "gross_return":
                gross_return,

            "net_return":
                net_return,

            "pnl":
                pnl,

            "capital":
                capital,
        })

        # ----------------------------------------------------
        # No overlapping positions.
        # ----------------------------------------------------

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# RUN TESTS
# ============================================================

all_results = []

print("\n")
print("=" * 75)
print("REGIME STRATEGY BACKTEST")
print("=" * 75)

print(
    f"Starting capital: "
    f"${STARTING_CAPITAL:,.2f}"
)

print(
    f"Round-trip cost: "
    f"{ROUND_TRIP_COST * 100:.2f}%"
)


for horizon, bars in HORIZONS.items():

    print("\n" + "-" * 75)
    print(f"HORIZON: {horizon}")
    print("-" * 75)

    results = backtest(bars)

    if results.empty:

        print("No trades.")

        continue

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = (
        results["capital"]
        .cummax()
    )

    drawdown = (
        results["capital"] /
        running_max
    ) - 1

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Long statistics
    # --------------------------------------------------------

    if not long_results.empty:

        long_win_rate = (
            long_results["net_return"] > 0
        ).mean()

        long_avg = (
            long_results["net_return"]
            .mean()
        )

    else:

        long_win_rate = np.nan
        long_avg = np.nan

    # --------------------------------------------------------
    # Short statistics
    # --------------------------------------------------------

    if not short_results.empty:

        short_win_rate = (
            short_results["net_return"] > 0
        ).mean()

        short_avg = (
            short_results["net_return"]
            .mean()
        )

    else:

        short_win_rate = np.nan
        short_avg = np.nan

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

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

    print("\nLong:")

    print(
        f"  Win rate:      "
        f"{long_win_rate * 100:.2f}%"
    )

    print(
        f"  Avg return:    "
        f"{long_avg * 100:.4f}%"
    )

    print("\nShort:")

    print(
        f"  Win rate:      "
        f"{short_win_rate * 100:.2f}%"
    )

    print(
        f"  Avg return:    "
        f"{short_avg * 100:.4f}%"
    )

    results["horizon"] = horizon

    all_results.append(results)


# ============================================================
# SAVE
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

    print("\n")
    print("=" * 75)
    print("BACKTEST COMPLETE")
    print("=" * 75)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Total recorded trades: "
        f"{len(final_results):,}"
    )

else:

    print("\nNo trades generated.")


print("\nDone.")