from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================================
# CONFIG
# =====================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_event_history.csv"
)

TRADES_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_event_trade_setups.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_event_trade_setup_summary.csv"
)

MATRIX_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_event_tp_sl_matrix.csv"
)


# ---------------------------------------------------------------------
# Exploratory setup thresholds
# ---------------------------------------------------------------------

# Minimum event shock over a rolling 5-minute window.
SHOCK_THRESHOLD_PCT = 0.10

# Minimum recovery/rejection from the detected extreme.
RECOVERY_THRESHOLD_PCT = 0.20

# Earliest minute after release at which we can enter.
MIN_ENTRY_MINUTE = 5

# Latest minute after release at which we search for an entry.
MAX_ENTRY_MINUTE = 60

# Confirmation structure:
# LONG  -> close above previous N-candle high
# SHORT -> close below previous N-candle low
CONFIRMATION_LOOKBACK = 3

# Once a shock occurs, don't allow a new extreme to be considered
# indefinitely.
EXTREME_LOOKBACK_MINUTES = 5

# Maximum holding period for trade simulation.
MAX_HOLD_MINUTES = 60

# Reference round-trip transaction cost.
ROUND_TRIP_COST_PCT = 0.14


# TP / SL levels are underlying BTC price movements.
TP_LEVELS_PCT = [
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

SL_LEVELS_PCT = [
    0.20,
    0.30,
    0.50,
]


# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 80)
print("KRONOS 1M EVENT TRADE SETUP BACKTEST")
print("=" * 80)
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce",
)

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"],
    utc=True,
    errors="coerce",
)

df["minutes_from_release"] = pd.to_numeric(
    df["minutes_from_release"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "timestamp",
        "event_timestamp",
        "minutes_from_release",
        "open",
        "high",
        "low",
        "close",
    ]
)

df = df.sort_values(
    ["event_timestamp", "timestamp"]
).reset_index(drop=True)


print(f"Total candles: {len(df):,}")
print(
    f"Events:        {df['event_timestamp'].nunique()}"
)
print()


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def pct_change(start, end):
    if start == 0 or pd.isna(start) or pd.isna(end):
        return np.nan

    return (end / start - 1.0) * 100.0


def directional_return(direction, entry, exit_price):
    if direction == "LONG":
        return pct_change(entry, exit_price)

    return pct_change(exit_price, entry)


def find_entry(event_df):
    """
    Find the first non-lookahead LONG/SHORT setup.

    LONG:
        1. 5-minute downside shock <= -threshold
        2. Local low develops
        3. Price recovers >= recovery threshold from low
        4. Close breaks above previous N-candle high

    SHORT:
        1. 5-minute upside shock >= threshold
        2. Local high develops
        3. Price rejects >= recovery threshold from high
        4. Close breaks below previous N-candle low

    Entry occurs at the NEXT candle open after confirmation.
    """

    work = event_df.copy()

    work = work.sort_values(
        "minutes_from_release"
    ).reset_index(drop=True)

    # Only search after release.
    post = work[
        work["minutes_from_release"] >= MIN_ENTRY_MINUTE
    ].copy()

    post = post[
        post["minutes_from_release"] <= MAX_ENTRY_MINUTE
    ].copy()

    if len(post) < 10:
        return None

    # -------------------------------------------------------------
    # Calculate rolling 5-minute return.
    # -------------------------------------------------------------

    work["close_5m_ago"] = (
        work["close"].shift(5)
    )

    work["return_5m_pct"] = (
        (work["close"] / work["close_5m_ago"] - 1.0)
        * 100.0
    )

    post = work[
        (work["minutes_from_release"] >= MIN_ENTRY_MINUTE)
        & (work["minutes_from_release"] <= MAX_ENTRY_MINUTE)
    ].copy()

    # -------------------------------------------------------------
    # Detect the first meaningful shock.
    # -------------------------------------------------------------

    downside = post[
        post["return_5m_pct"] <= -SHOCK_THRESHOLD_PCT
    ]

    upside = post[
        post["return_5m_pct"] >= SHOCK_THRESHOLD_PCT
    ]

    candidates = []

    # =============================================================
    # LONG RECOVERY
    # =============================================================

    if not downside.empty:

        shock_row = downside.iloc[0]

        shock_minute = shock_row["minutes_from_release"]

        after_shock = work[
            work["minutes_from_release"] >= shock_minute
        ].copy()

        after_shock = after_shock[
            after_shock["minutes_from_release"]
            <= MAX_ENTRY_MINUTE
        ]

        if not after_shock.empty:

            # Find lowest low after shock.
            low_idx = after_shock["low"].idxmin()

            low_row = work.loc[low_idx]

            recovery_low = float(low_row["low"])
            low_minute = float(
                low_row["minutes_from_release"]
            )

            # Need time after the low to confirm recovery.
            confirm = work[
                work["minutes_from_release"] > low_minute
            ].copy()

            confirm = confirm[
                confirm["minutes_from_release"]
                <= MAX_ENTRY_MINUTE
            ]

            if not confirm.empty:

                for idx in confirm.index:

                    row = work.loc[idx]

                    recovery_pct = pct_change(
                        recovery_low,
                        row["close"],
                    )

                    if (
                        pd.isna(recovery_pct)
                        or recovery_pct < RECOVERY_THRESHOLD_PCT
                    ):
                        continue

                    # Previous N candles only.
                    previous = work.loc[
                        max(0, idx - CONFIRMATION_LOOKBACK):
                        idx - 1
                    ]

                    if len(previous) < CONFIRMATION_LOOKBACK:
                        continue

                    previous_high = previous["high"].max()

                    if row["close"] <= previous_high:
                        continue

                    # Entry is next candle OPEN.
                    next_rows = work[
                        work["timestamp"] > row["timestamp"]
                    ]

                    if next_rows.empty:
                        continue

                    next_row = next_rows.iloc[0]

                    candidates.append(
                        {
                            "direction": "LONG",
                            "event_timestamp":
                                row["event_timestamp"],
                            "event_type":
                                row["event_type"],
                            "importance":
                                row["importance"],
                            "shock_minute":
                                shock_minute,
                            "shock_return_5m_pct":
                                shock_row["return_5m_pct"],
                            "extreme_minute":
                                low_minute,
                            "extreme_price":
                                recovery_low,
                            "confirmation_minute":
                                row["minutes_from_release"],
                            "confirmation_price":
                                row["close"],
                            "entry_minute":
                                next_row["minutes_from_release"],
                            "entry_timestamp":
                                next_row["timestamp"],
                            "entry_price":
                                next_row["open"],
                            "recovery_pct":
                                recovery_pct,
                        }
                    )

                    # First valid setup only.
                    break

    # =============================================================
    # SHORT REJECTION
    # =============================================================

    if not upside.empty:

        shock_row = upside.iloc[0]

        shock_minute = shock_row["minutes_from_release"]

        after_shock = work[
            work["minutes_from_release"] >= shock_minute
        ].copy()

        after_shock = after_shock[
            after_shock["minutes_from_release"]
            <= MAX_ENTRY_MINUTE
        ]

        if not after_shock.empty:

            high_idx = after_shock["high"].idxmax()

            high_row = work.loc[high_idx]

            rejection_high = float(high_row["high"])
            high_minute = float(
                high_row["minutes_from_release"]
            )

            confirm = work[
                work["minutes_from_release"] > high_minute
            ].copy()

            confirm = confirm[
                confirm["minutes_from_release"]
                <= MAX_ENTRY_MINUTE
            ]

            if not confirm.empty:

                for idx in confirm.index:

                    row = work.loc[idx]

                    rejection_pct = pct_change(
                        row["close"],
                        rejection_high,
                    )

                    if (
                        pd.isna(rejection_pct)
                        or rejection_pct < RECOVERY_THRESHOLD_PCT
                    ):
                        continue

                    previous = work.loc[
                        max(0, idx - CONFIRMATION_LOOKBACK):
                        idx - 1
                    ]

                    if len(previous) < CONFIRMATION_LOOKBACK:
                        continue

                    previous_low = previous["low"].min()

                    if row["close"] >= previous_low:
                        continue

                    next_rows = work[
                        work["timestamp"] > row["timestamp"]
                    ]

                    if next_rows.empty:
                        continue

                    next_row = next_rows.iloc[0]

                    candidates.append(
                        {
                            "direction": "SHORT",
                            "event_timestamp":
                                row["event_timestamp"],
                            "event_type":
                                row["event_type"],
                            "importance":
                                row["importance"],
                            "shock_minute":
                                shock_minute,
                            "shock_return_5m_pct":
                                shock_row["return_5m_pct"],
                            "extreme_minute":
                                high_minute,
                            "extreme_price":
                                rejection_high,
                            "confirmation_minute":
                                row["minutes_from_release"],
                            "confirmation_price":
                                row["close"],
                            "entry_minute":
                                next_row["minutes_from_release"],
                            "entry_timestamp":
                                next_row["timestamp"],
                            "entry_price":
                                next_row["open"],
                            "recovery_pct":
                                rejection_pct,
                        }
                    )

                    break

    if not candidates:
        return None

    # -------------------------------------------------------------
    # If both directions somehow qualify, choose the earliest entry.
    # -------------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x["entry_timestamp"],
    )

    return candidates[0]


# =====================================================================
# SIMULATE TRADE
# =====================================================================

def simulate_trade(
    event_df,
    setup,
    tp_pct,
    sl_pct,
):
    direction = setup["direction"]

    entry_timestamp = setup["entry_timestamp"]
    entry_price = float(setup["entry_price"])

    future = event_df[
        event_df["timestamp"] >= entry_timestamp
    ].copy()

    future = future[
        future["minutes_from_release"]
        <= setup["entry_minute"] + MAX_HOLD_MINUTES
    ].copy()

    if future.empty:
        return None

    tp_price = (
        entry_price * (1.0 + tp_pct / 100.0)
        if direction == "LONG"
        else entry_price * (1.0 - tp_pct / 100.0)
    )

    sl_price = (
        entry_price * (1.0 - sl_pct / 100.0)
        if direction == "LONG"
        else entry_price * (1.0 + sl_pct / 100.0)
    )

    exit_price = None
    exit_timestamp = None
    exit_minute = None
    exit_reason = None

    # -------------------------------------------------------------
    # Intrabar TP/SL handling.
    #
    # If both TP and SL are touched within the same candle,
    # assume SL happened first.
    # This is conservative.
    # -------------------------------------------------------------

    for _, row in future.iterrows():

        high = float(row["high"])
        low = float(row["low"])

        if direction == "LONG":

            hit_sl = low <= sl_price
            hit_tp = high >= tp_price

        else:

            hit_sl = high >= sl_price
            hit_tp = low <= tp_price

        if hit_sl and hit_tp:

            exit_price = sl_price
            exit_reason = "SL_AND_TP_SAME_CANDLE_SL_FIRST"

        elif hit_sl:

            exit_price = sl_price
            exit_reason = "SL"

        elif hit_tp:

            exit_price = tp_price
            exit_reason = "TP"

        else:

            continue

        exit_timestamp = row["timestamp"]
        exit_minute = row["minutes_from_release"]

        break

    # -------------------------------------------------------------
    # Time exit if neither TP nor SL hit.
    # -------------------------------------------------------------

    if exit_price is None:

        final_row = future.iloc[-1]

        exit_price = float(final_row["close"])
        exit_timestamp = final_row["timestamp"]
        exit_minute = final_row["minutes_from_release"]
        exit_reason = "TIME_EXIT"

    gross_return = directional_return(
        direction,
        entry_price,
        exit_price,
    )

    net_return = (
        gross_return
        - ROUND_TRIP_COST_PCT
    )

    return {
        "direction": direction,
        "entry_timestamp": entry_timestamp,
        "entry_minute": setup["entry_minute"],
        "entry_price": entry_price,
        "exit_timestamp": exit_timestamp,
        "exit_minute": exit_minute,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
    }


# =====================================================================
# BUILD SETUPS
# =====================================================================

print("Searching for 1m event setups...")
print()

setups = []

for event_timestamp, event_df in df.groupby(
    "event_timestamp",
    sort=True,
):

    event_df = event_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    setup = find_entry(event_df)

    if setup is not None:
        setups.append(setup)


print(
    f"Detected setups: {len(setups)}"
)

print()


# =====================================================================
# SAVE BASIC SETUPS
# =====================================================================

if not setups:

    print("No setups detected.")
    print()
    print(
        "Try inspecting the thresholds before changing the strategy."
    )
    raise SystemExit(0)


setup_df = pd.DataFrame(setups)

setup_df.to_csv(
    TRADES_FILE,
    index=False,
)

print(
    f"Saved setup candidates:\n{TRADES_FILE}"
)

print()


# =====================================================================
# TP / SL MATRIX
# =====================================================================

results = []

for setup in setups:

    event_df = df[
        df["event_timestamp"]
        == setup["event_timestamp"]
    ].copy()

    for tp_pct in TP_LEVELS_PCT:

        for sl_pct in SL_LEVELS_PCT:

            result = simulate_trade(
                event_df,
                setup,
                tp_pct,
                sl_pct,
            )

            if result is None:
                continue

            result.update(
                {
                    "event_timestamp":
                        setup["event_timestamp"],
                    "event_type":
                        setup["event_type"],
                    "importance":
                        setup["importance"],
                    "shock_minute":
                        setup["shock_minute"],
                    "shock_return_5m_pct":
                        setup["shock_return_5m_pct"],
                    "extreme_minute":
                        setup["extreme_minute"],
                    "extreme_price":
                        setup["extreme_price"],
                    "confirmation_minute":
                        setup["confirmation_minute"],
                    "confirmation_price":
                        setup["confirmation_price"],
                    "recovery_pct":
                        setup["recovery_pct"],
                }
            )

            results.append(result)


results_df = pd.DataFrame(results)

results_df.to_csv(
    MATRIX_FILE,
    index=False,
)


# =====================================================================
# SUMMARY
# =====================================================================

summary_rows = []

for (tp_pct, sl_pct), group in results_df.groupby(
    ["tp_pct", "sl_pct"]
):

    net = group["net_return_pct"]

    wins = net > 0
    losses = net < 0

    gross_profit = net[net > 0].sum()
    gross_loss = abs(net[net < 0].sum())

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = np.inf

    summary_rows.append(
        {
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "trades": len(group),
            "win_rate_pct":
                wins.mean() * 100,
            "avg_gross_return_pct":
                group["gross_return_pct"].mean(),
            "median_gross_return_pct":
                group["gross_return_pct"].median(),
            "avg_net_return_pct":
                net.mean(),
            "median_net_return_pct":
                net.median(),
            "profit_factor":
                profit_factor,
            "avg_entry_minute":
                group["entry_minute"].mean(),
            "avg_exit_minute":
                group["exit_minute"].mean(),
            "tp_exits_pct":
                (
                    group["exit_reason"]
                    == "TP"
                ).mean() * 100,
            "sl_exits_pct":
                group["exit_reason"]
                .isin(
                    [
                        "SL",
                        "SL_AND_TP_SAME_CANDLE_SL_FIRST",
                    ]
                ).mean() * 100,
            "time_exits_pct":
                (
                    group["exit_reason"]
                    == "TIME_EXIT"
                ).mean() * 100,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
).sort_values(
    ["tp_pct", "sl_pct"]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =====================================================================
# CONSOLE OUTPUT
# =====================================================================

print("=" * 80)
print("SETUP SUMMARY")
print("=" * 80)

print(
    f"Total events:       {df['event_timestamp'].nunique()}"
)

print(
    f"Detected setups:    {len(setups)}"
)

print(
    f"LONG setups:        "
    f"{sum(x['direction'] == 'LONG' for x in setups)}"
)

print(
    f"SHORT setups:       "
    f"{sum(x['direction'] == 'SHORT' for x in setups)}"
)

print()

print("SETUP CANDIDATES")
print("-" * 80)

display_columns = [
    "event_timestamp",
    "event_type",
    "direction",
    "shock_minute",
    "shock_return_5m_pct",
    "extreme_minute",
    "confirmation_minute",
    "entry_minute",
    "entry_price",
    "recovery_pct",
]

print(
    setup_df[display_columns]
    .to_string(index=False)
)

print()

print("=" * 80)
print("TP / SL RESULTS")
print("=" * 80)

print(
    summary_df.to_string(index=False)
)

print()

print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Setups:\n{TRADES_FILE}"
)

print(
    f"\nTP/SL trades:\n{MATRIX_FILE}"
)

print(
    f"\nSummary:\n{SUMMARY_FILE}"
)

print()

print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "This is an exploratory price-action backtest."
)

print(
    "It does NOT represent leveraged account ROI."
)

print(
    "It does NOT prove future profitability."
)

print(
    "39 economic events is a small sample."
)

print(
    "Do not optimize thresholds against this dataset."
)

print()