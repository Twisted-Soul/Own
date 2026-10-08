from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_test_predictions.csv"
)

OHLC_FILE = Path(
    "data/btc/BTCUSDT_5m_1y.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_test_backtest.csv"
)

STARTING_CAPITAL = 10_000.0

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = (
    2 * FEE_PER_SIDE
    + 2 * SLIPPAGE_PER_SIDE
)

HOLDING_BARS = 12


# ============================================================
# LOAD ML TEST PREDICTIONS
# ============================================================

print("Loading untouched test predictions...")

df = pd.read_csv(PREDICTIONS_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
)

print(
    f"Loaded {len(df):,} rows."
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "timestamps",
    "future_return_1h",
    "target",
    "logistic_prediction",
    "rf_class_prediction",
    "ridge_prediction",
    "rf_regression_prediction",
}

missing = (
    required_columns
    - set(df.columns)
)

if missing:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(sorted(missing))
    )


# ============================================================
# LOAD ORIGINAL OHLCV
# ============================================================

print()
print("Loading original OHLCV data...")

ohlc = pd.read_csv(OHLC_FILE)

ohlc["timestamps"] = pd.to_datetime(
    ohlc["timestamps"],
    utc=True,
)

ohlc = ohlc[
    [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
    ]
].copy()

ohlc = ohlc.sort_values(
    "timestamps"
).reset_index(drop=True)

print(
    f"Loaded {len(ohlc):,} OHLCV candles."
)


# ============================================================
# CREATE EXECUTION PRICES
# ============================================================
#
# Signal is generated at candle t close.
#
# Entry:
#   candle t+1 open
#
# Exit:
#   12 candles after entry = approximately 1 hour
#
# We create these prices directly in the OHLC dataframe
# before merging them into the test prediction dataframe.
# ============================================================

ohlc["next_open"] = (
    ohlc["open"].shift(-1)
)

ohlc["next_timestamp"] = (
    ohlc["timestamps"].shift(-1)
)

# Entry is next candle open.
# Exit is the open 12 bars after entry.
#
# If signal is at index i:
#   entry = i + 1
#   exit  = i + 13
#
# Therefore the exit open can be obtained with shift(-13).

ohlc["exit_open_1h"] = (
    ohlc["open"].shift(
        -(HOLDING_BARS + 1)
    )
)

ohlc["exit_timestamp_1h"] = (
    ohlc["timestamps"].shift(
        -(HOLDING_BARS + 1)
    )
)


# ============================================================
# MERGE EXECUTION DATA
# ============================================================

execution_columns = [
    "timestamps",
    "next_open",
    "next_timestamp",
    "exit_open_1h",
    "exit_timestamp_1h",
]

df = df.merge(
    ohlc[execution_columns],
    on="timestamps",
    how="left",
)

# Remove rows where we cannot execute the full trade.

df = df.dropna(
    subset=[
        "next_open",
        "next_timestamp",
        "exit_open_1h",
        "exit_timestamp_1h",
    ]
).copy()


# ============================================================
# VERIFY 5-MINUTE CONTINUITY
# ============================================================

time_difference = (
    df["next_timestamp"]
    - df["timestamps"]
)

valid_spacing = (
    time_difference
    == pd.Timedelta(
        minutes=5
    )
)

print()
print(
    f"Valid 5-minute transitions: "
    f"{valid_spacing.mean():.2%}"
)

df = df[
    valid_spacing
].copy()


# ============================================================
# MODEL DEFINITIONS
# ============================================================

strategies = {
    "logistic": (
        "logistic_prediction",
        "classification",
    ),
    "rf_classifier": (
        "rf_class_prediction",
        "classification",
    ),
    "ridge": (
        "ridge_prediction",
        "regression",
    ),
    "rf_regression": (
        "rf_regression_prediction",
        "regression",
    ),
}


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
    data,
    signal_column,
):
    """
    Non-overlapping 1-hour directional backtest.

    Signal:
        candle t

    Entry:
        candle t+1 open

    Exit:
        candle t+13 open

    Holding period:
        12 x 5-minute candles = 1 hour

    Costs:
        0.05% fee per side
        0.02% slippage per side
        0.14% total round trip
    """

    trades = []

    capital = STARTING_CAPITAL

    equity_points = []

    i = 0

    while i < len(data):

        row = data.iloc[i]

        raw_signal = row[
            signal_column
        ]

        # ----------------------------------------------------
        # Convert prediction to direction
        # ----------------------------------------------------

        if pd.isna(raw_signal):
            direction = 0

        elif raw_signal > 0:
            direction = 1

        elif raw_signal < 0:
            direction = -1

        else:
            direction = 0

        # ----------------------------------------------------
        # No trade
        # ----------------------------------------------------

        if direction == 0:

            equity_points.append(
                {
                    "timestamp": row[
                        "timestamps"
                    ],
                    "equity": capital,
                }
            )

            i += 1
            continue

        # ----------------------------------------------------
        # Execution prices
        # ----------------------------------------------------

        entry_price = row[
            "next_open"
        ]

        exit_price = row[
            "exit_open_1h"
        ]

        entry_timestamp = row[
            "next_timestamp"
        ]

        exit_timestamp = row[
            "exit_timestamp_1h"
        ]

        if (
            pd.isna(entry_price)
            or pd.isna(exit_price)
        ):
            break

        # ----------------------------------------------------
        # Gross directional return
        # ----------------------------------------------------

        if direction == 1:

            gross_return = (
                exit_price
                / entry_price
                - 1.0
            )

        else:

            gross_return = (
                entry_price
                / exit_price
                - 1.0
            )

        # ----------------------------------------------------
        # Trading costs
        # ----------------------------------------------------

        net_return = (
            gross_return
            - ROUND_TRIP_COST
        )

        capital_before = capital

        capital *= (
            1.0
            + net_return
        )

        trades.append(
            {
                "signal_timestamp": row[
                    "timestamps"
                ],
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": exit_timestamp,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "capital_before": capital_before,
                "capital_after": capital,
            }
        )

        equity_points.append(
            {
                "timestamp": exit_timestamp,
                "equity": capital,
            }
        )

        # ----------------------------------------------------
        # Skip while position is open
        # ----------------------------------------------------

        #
        # Signal at i
        # Entry at i+1
        # Exit at i+13
        #
        # Next usable signal is after the exit.
        #

        i += HOLDING_BARS + 2

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_points),
    )


# ============================================================
# BACKTEST ALL MODELS
# ============================================================

all_results = []

summary_rows = []


for strategy_name, (
    signal_column,
    model_type,
) in strategies.items():

    print()
    print("=" * 70)
    print(
        f"BACKTESTING: "
        f"{strategy_name.upper()}"
    )
    print("=" * 70)

    trades, equity = run_backtest(
        df,
        signal_column,
    )

    if trades.empty:

        print(
            "No trades generated."
        )

        continue

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    total_trades = len(trades)

    long_trades = int(
        (
            trades["direction"]
            == 1
        ).sum()
    )

    short_trades = int(
        (
            trades["direction"]
            == -1
        ).sum()
    )

    winning_trades = int(
        (
            trades["net_return"]
            > 0
        ).sum()
    )

    win_rate = (
        winning_trades
        / total_trades
    )

    avg_gross = (
        trades["gross_return"]
        .mean()
    )

    avg_net = (
        trades["net_return"]
        .mean()
    )

    median_net = (
        trades["net_return"]
        .median()
    )

    final_capital = (
        trades["capital_after"]
        .iloc[-1]
    )

    total_return = (
        final_capital
        / STARTING_CAPITAL
        - 1.0
    )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    winning_returns = trades.loc[
        trades["net_return"] > 0,
        "net_return",
    ]

    losing_returns = trades.loc[
        trades["net_return"] < 0,
        "net_return",
    ]

    gross_profit = (
        winning_returns.sum()
    )

    gross_loss = abs(
        losing_returns.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    equity_values = (
        equity["equity"]
        .to_numpy()
    )

    running_max = np.maximum.accumulate(
        equity_values
    )

    drawdowns = (
        equity_values
        / running_max
        - 1.0
    )

    max_drawdown = (
        drawdowns.min()
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Trades:              "
        f"{total_trades:,}"
    )

    print(
        f"Long trades:         "
        f"{long_trades:,}"
    )

    print(
        f"Short trades:        "
        f"{short_trades:,}"
    )

    print(
        f"Win rate:            "
        f"{win_rate:.2%}"
    )

    print(
        f"Avg gross/trade:     "
        f"{avg_gross:+.4%}"
    )

    print(
        f"Avg net/trade:       "
        f"{avg_net:+.4%}"
    )

    print(
        f"Median net/trade:    "
        f"{median_net:+.4%}"
    )

    print(
        f"Profit factor:       "
        f"{profit_factor:.3f}"
    )

    print(
        f"Final capital:       "
        f"${final_capital:,.2f}"
    )

    print(
        f"Total return:        "
        f"{total_return:+.2%}"
    )

    print(
        f"Max drawdown:        "
        f"{max_drawdown:.2%}"
    )

    # --------------------------------------------------------
    # Add strategy label
    # --------------------------------------------------------

    trades["strategy"] = (
        strategy_name
    )

    all_results.append(
        trades
    )

    summary_rows.append(
        {
            "strategy": strategy_name,
            "model_type": model_type,
            "trades": total_trades,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "win_rate": win_rate,
            "avg_gross": avg_gross,
            "avg_net": avg_net,
            "median_net": median_net,
            "profit_factor": profit_factor,
            "final_capital": final_capital,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
        }
    )


# ============================================================
# SAVE TRADE RESULTS
# ============================================================

print()
print("=" * 70)
print("BACKTEST SUMMARY")
print("=" * 70)

if all_results:

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved trades: "
        f"{OUTPUT_FILE}"
    )

else:

    print(
        "No strategies generated trades."
    )


# ============================================================
# BUY & HOLD REFERENCE
# ============================================================

print()
print("=" * 70)
print("BUY & HOLD REFERENCE")
print("=" * 70)

test_start = (
    df["timestamps"].iloc[0]
)

test_end = (
    df["timestamps"].iloc[-1]
)

first_price = (
    df["next_open"].iloc[0]
)

# Last available close in the test period.

last_price = (
    ohlc.loc[
        ohlc["timestamps"]
        <= test_end,
        "close",
    ]
    .iloc[-1]
)

buy_hold_return = (
    last_price
    / first_price
    - 1.0
)

print(
    f"Test period: "
    f"{test_start} → {test_end}"
)

print(
    f"Buy & hold return: "
    f"{buy_hold_return:+.2%}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ML TEST BACKTEST COMPLETE")
print("=" * 70)

print(
    f"Round-trip trading cost: "
    f"{ROUND_TRIP_COST:.2%}"
)

print(
    f"Holding period: "
    f"{HOLDING_BARS * 5} minutes"
)