# examples/analyze_1m_event_trade_setups.py

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

EVENT_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_history.csv"

OUTPUT_SETUPS = (
    BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_trade_setups.csv"
)

OUTPUT_MATRIX = (
    BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_tp_sl_matrix.csv"
)

OUTPUT_SUMMARY = (
    BASE_DIR / "data" / "btc" / "BTCUSDT_1m_event_trade_setup_summary.csv"
)


# Exploratory thresholds.
# DO NOT optimize these on the 39 events.
SHOCK_THRESHOLD = 0.0010       # 0.10%
RECOVERY_THRESHOLD = 0.0020    # 0.20%

MIN_ENTRY_MINUTE = 5
MAX_ENTRY_MINUTE = 60

CONFIRM_LOOKBACK = 3

MAX_HOLD_MINUTES = 60

ROUND_TRIP_COST = 0.0014       # 0.14%


TP_LEVELS = [
    0.0020,   # 0.20%
    0.0030,   # 0.30%
    0.0050,   # 0.50%
    0.0075,   # 0.75%
    0.0100,   # 1.00%
]

SL_LEVELS = [
    0.0020,   # 0.20%
    0.0030,   # 0.30%
    0.0050,   # 0.50%
]


# ============================================================
# HELPERS
# ============================================================

def pct(x):
    return x * 100.0


def calculate_rolling_return(df, bars=5):
    """
    Return over the previous N completed 1-minute candles.

    At candle t:
        close[t] / close[t-bars] - 1

    This uses only information available at candle t close.
    """
    return df["close"] / df["close"].shift(bars) - 1.0


def confirmation_long(df, pos):
    """
    Long confirmation:
    current close > highest high of previous CONFIRM_LOOKBACK candles.

    The current candle is NOT included in the reference range.
    """
    if pos < CONFIRM_LOOKBACK:
        return False

    previous = df.iloc[
        pos - CONFIRM_LOOKBACK:pos
    ]

    previous_high = previous["high"].max()

    return df.iloc[pos]["close"] > previous_high


def confirmation_short(df, pos):
    """
    Short confirmation:
    current close < lowest low of previous CONFIRM_LOOKBACK candles.
    """
    if pos < CONFIRM_LOOKBACK:
        return False

    previous = df.iloc[
        pos - CONFIRM_LOOKBACK:pos
    ]

    previous_low = previous["low"].min()

    return df.iloc[pos]["close"] < previous_low


# ============================================================
# SEQUENTIAL SETUP DETECTION
# ============================================================

def find_entry(event_df):
    """
    Detect ONE trade setup sequentially.

    Important:
    This function deliberately processes candles one at a time.

    It NEVER looks ahead to find the eventual extreme.

    Sequence:

    1. Wait for first qualifying shock.
    2. Lock direction.
    3. Track running extreme.
    4. Wait for recovery/rejection.
    5. Check confirmation.
    6. Enter next candle open.
    """

    df = event_df.reset_index(drop=True).copy()

    if len(df) == 0:
        return None

    # Ensure rolling return exists.
    if "return_5m" not in df.columns:
        df["return_5m"] = calculate_rolling_return(df)

    active_direction = None

    running_extreme = None
    shock_pos = None
    extreme_pos = None
    confirmation_pos = None

    release_timestamp = df.iloc[0]["event_timestamp"]

    for pos in range(len(df)):

        row = df.iloc[pos]

        minutes_from_release = row["minutes_from_release"]

        # We only start searching after MIN_ENTRY_MINUTE.
        if minutes_from_release < MIN_ENTRY_MINUTE:
            continue

        # Stop looking for new entries after the allowed window.
        if minutes_from_release > MAX_ENTRY_MINUTE:
            break

        rolling_return = row["return_5m"]

        # ====================================================
        # PHASE 1 — DETECT FIRST SHOCK
        # ====================================================

        if active_direction is None:

            # Downside shock -> look for LONG recovery
            if rolling_return <= -SHOCK_THRESHOLD:

                active_direction = "LONG"
                shock_pos = pos

                running_extreme = row["low"]
                extreme_pos = pos

                continue

            # Upside shock -> look for SHORT rejection
            if rolling_return >= SHOCK_THRESHOLD:

                active_direction = "SHORT"
                shock_pos = pos

                running_extreme = row["high"]
                extreme_pos = pos

                continue

            continue

        # ====================================================
        # PHASE 2 — LONG
        # ====================================================

        if active_direction == "LONG":

            # Update running low.
            if row["low"] < running_extreme:
                running_extreme = row["low"]
                extreme_pos = pos

            recovery = (
                row["close"] / running_extreme
            ) - 1.0

            # Recovery has not happened yet.
            if recovery < RECOVERY_THRESHOLD:
                continue

            # Need confirmation.
            if not confirmation_long(df, pos):
                continue

            # Must have a next candle for entry.
            if pos + 1 >= len(df):
                continue

            entry_pos = pos + 1
            entry_row = df.iloc[entry_pos]

            return {
                "direction": "LONG",
                "shock_pos": shock_pos,
                "extreme_pos": extreme_pos,
                "confirmation_pos": pos,
                "entry_pos": entry_pos,

                "shock_timestamp": df.iloc[shock_pos]["timestamp"],
                "extreme_timestamp": df.iloc[extreme_pos]["timestamp"],
                "confirmation_timestamp": df.iloc[pos]["timestamp"],
                "entry_timestamp": entry_row["timestamp"],

                "shock_price": df.iloc[shock_pos]["close"],
                "extreme_price": running_extreme,
                "confirmation_price": row["close"],
                "entry_price": entry_row["open"],

                "recovery_pct": recovery,

                "minutes_from_release_entry":
                    entry_row["minutes_from_release"],

                "release_timestamp": release_timestamp,
            }

        # ====================================================
        # PHASE 2 — SHORT
        # ====================================================

        if active_direction == "SHORT":

            # Update running high.
            if row["high"] > running_extreme:
                running_extreme = row["high"]
                extreme_pos = pos

            rejection = (
                running_extreme / row["close"]
            ) - 1.0

            # Rejection has not happened yet.
            if rejection < RECOVERY_THRESHOLD:
                continue

            # Need confirmation.
            if not confirmation_short(df, pos):
                continue

            # Must have a next candle for entry.
            if pos + 1 >= len(df):
                continue

            entry_pos = pos + 1
            entry_row = df.iloc[entry_pos]

            return {
                "direction": "SHORT",
                "shock_pos": shock_pos,
                "extreme_pos": extreme_pos,
                "confirmation_pos": pos,
                "entry_pos": entry_pos,

                "shock_timestamp": df.iloc[shock_pos]["timestamp"],
                "extreme_timestamp": df.iloc[extreme_pos]["timestamp"],
                "confirmation_timestamp": df.iloc[pos]["timestamp"],
                "entry_timestamp": entry_row["timestamp"],

                "shock_price": df.iloc[shock_pos]["close"],
                "extreme_price": running_extreme,
                "confirmation_price": row["close"],
                "entry_price": entry_row["open"],

                "recovery_pct": rejection,

                "minutes_from_release_entry":
                    entry_row["minutes_from_release"],

                "release_timestamp": release_timestamp,
            }

    return None


# ============================================================
# TP / SL SIMULATION
# ============================================================

def simulate_trade(
    event_df,
    setup,
    tp,
    sl,
):
    """
    Simulate the trade AFTER entry.

    This portion is allowed to use future candles because
    it is simulating what happened after the trade was entered.

    Conservative rule:
    If TP and SL are both touched in the same candle,
    assume SL happened first.
    """

    df = event_df.reset_index(drop=True)

    entry_pos = setup["entry_pos"]

    entry_price = setup["entry_price"]

    direction = setup["direction"]

    max_exit_pos = min(
        entry_pos + MAX_HOLD_MINUTES,
        len(df) - 1,
    )

    tp_price = None
    sl_price = None

    if direction == "LONG":

        tp_price = entry_price * (1.0 + tp)
        sl_price = entry_price * (1.0 - sl)

    else:

        tp_price = entry_price * (1.0 - tp)
        sl_price = entry_price * (1.0 + sl)

    exit_reason = "TIME"
    exit_pos = max_exit_pos
    exit_price = df.iloc[max_exit_pos]["close"]

    hit_tp = False
    hit_sl = False

    for pos in range(entry_pos, max_exit_pos + 1):

        row = df.iloc[pos]

        high = row["high"]
        low = row["low"]

        if direction == "LONG":

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

        else:

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

        # Conservative same-candle handling.
        if hit_tp and hit_sl:

            exit_reason = "SL"
            exit_pos = pos
            exit_price = sl_price

            break

        if hit_sl:

            exit_reason = "SL"
            exit_pos = pos
            exit_price = sl_price

            break

        if hit_tp:

            exit_reason = "TP"
            exit_pos = pos
            exit_price = tp_price

            break

    # Raw price return.
    if direction == "LONG":

        gross_return = (
            exit_price / entry_price
        ) - 1.0

    else:

        gross_return = (
            entry_price / exit_price
        ) - 1.0

    net_return = gross_return - ROUND_TRIP_COST

    # MAE / MFE after entry.
    future = df.iloc[
        entry_pos:max_exit_pos + 1
    ]

    if direction == "LONG":

        mfe = (
            future["high"].max() / entry_price
        ) - 1.0

        mae = (
            future["low"].min() / entry_price
        ) - 1.0

    else:

        mfe = (
            entry_price / future["low"].min()
        ) - 1.0

        mae = (
            entry_price / future["high"].max()
        ) - 1.0

    hold_minutes = (
        df.iloc[exit_pos]["minutes_from_release"]
        - df.iloc[entry_pos]["minutes_from_release"]
    )

    full_horizon = (
        exit_pos >= entry_pos + MAX_HOLD_MINUTES
        or exit_pos == len(df) - 1
    )

    return {
        "tp_pct": pct(tp),
        "sl_pct": pct(sl),

        "exit_timestamp":
            df.iloc[exit_pos]["timestamp"],

        "exit_price": exit_price,

        "exit_reason": exit_reason,

        "gross_return_pct": pct(gross_return),
        "net_return_pct": pct(net_return),

        "mfe_pct": pct(mfe),
        "mae_pct": pct(mae),

        "hold_minutes": hold_minutes,

        "tp_hit": exit_reason == "TP",
        "sl_hit": exit_reason == "SL",

        "full_horizon":
            full_horizon,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 68)
print("BTC/USDT 1m EVENT TRADE SETUP BACKTEST")
print("NO-LOOKAHEAD VERSION")
print("=" * 68)

print()

df = pd.read_csv(EVENT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"],
    utc=True,
)

df = df.sort_values(
    ["event_timestamp", "timestamp"]
).reset_index(drop=True)

df["return_5m"] = (
    df.groupby("event_timestamp")["close"]
    .transform(
        lambda x: x / x.shift(5) - 1.0
    )
)


# ============================================================
# EVENT PROCESSING
# ============================================================

all_setups = []
all_results = []

events = list(
    df.groupby(
        [
            "event_timestamp",
            "event_type",
        ],
        sort=True,
    )
)

print(f"Loaded candles : {len(df):,}")
print(f"Events         : {len(events)}")
print()

for event_number, (
    (event_timestamp, event_type),
    event_df,
) in enumerate(events, start=1):

    event_df = (
        event_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    setup = find_entry(event_df)

    if setup is None:
        continue

    setup_record = {
        "event_timestamp": event_timestamp,
        "event_type": event_type,

        "direction": setup["direction"],

        "shock_timestamp": setup["shock_timestamp"],
        "extreme_timestamp": setup["extreme_timestamp"],
        "confirmation_timestamp":
            setup["confirmation_timestamp"],
        "entry_timestamp": setup["entry_timestamp"],

        "shock_price": setup["shock_price"],
        "extreme_price": setup["extreme_price"],
        "confirmation_price":
            setup["confirmation_price"],
        "entry_price": setup["entry_price"],

        "recovery_pct":
            setup["recovery_pct"] * 100,

        "entry_minutes_from_release":
            setup["minutes_from_release_entry"],
    }

    all_setups.append(setup_record)

    # --------------------------------------------------------
    # TP / SL MATRIX
    # --------------------------------------------------------

    for tp in TP_LEVELS:

        for sl in SL_LEVELS:

            result = simulate_trade(
                event_df,
                setup,
                tp,
                sl,
            )

            record = {
                "event_timestamp": event_timestamp,
                "event_type": event_type,

                "direction": setup["direction"],

                "entry_timestamp":
                    setup["entry_timestamp"],

                "entry_price":
                    setup["entry_price"],

                "entry_minutes_from_release":
                    setup["minutes_from_release_entry"],

                "recovery_pct":
                    setup["recovery_pct"] * 100,

                **result,
            }

            all_results.append(record)


# ============================================================
# SAVE SETUPS
# ============================================================

setups_df = pd.DataFrame(all_setups)

results_df = pd.DataFrame(all_results)

if len(setups_df) > 0:

    setups_df.to_csv(
        OUTPUT_SETUPS,
        index=False,
    )

if len(results_df) > 0:

    results_df.to_csv(
        OUTPUT_MATRIX,
        index=False,
    )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

if len(results_df) > 0:

    for (tp, sl), group in results_df.groupby(
        ["tp_pct", "sl_pct"]
    ):

        wins = group[
            group["exit_reason"] == "TP"
        ]

        losses = group[
            group["exit_reason"] == "SL"
        ]

        time_exits = group[
            group["exit_reason"] == "TIME"
        ]

        gross_wins = wins["gross_return_pct"].sum()

        gross_losses = abs(
            losses["gross_return_pct"].sum()
        )

        if gross_losses > 0:

            profit_factor = (
                gross_wins / gross_losses
            )

        else:

            profit_factor = np.inf

        summary_rows.append({

            "tp_pct": tp,
            "sl_pct": sl,

            "trades": len(group),

            "wins": len(wins),
            "losses": len(losses),
            "time_exits": len(time_exits),

            "win_rate_pct":
                len(wins) / len(group) * 100,

            "profit_factor":
                profit_factor,

            "avg_gross_return_pct":
                group["gross_return_pct"].mean(),

            "avg_net_return_pct":
                group["net_return_pct"].mean(),

            "median_net_return_pct":
                group["net_return_pct"].median(),

            "total_net_return_pct":
                group["net_return_pct"].sum(),

            "avg_mfe_pct":
                group["mfe_pct"].mean(),

            "avg_mae_pct":
                group["mae_pct"].mean(),

            "avg_hold_minutes":
                group["hold_minutes"].mean(),

            "full_horizon_pct":
                group["full_horizon"].mean() * 100,
        })


summary_df = pd.DataFrame(summary_rows)

if len(summary_df) > 0:

    summary_df = summary_df.sort_values(
        "avg_net_return_pct",
        ascending=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 68)
print("SETUP RESULTS")
print("=" * 68)

print(
    f"Events with setup : {len(setups_df)} / {len(events)}"
)

if len(setups_df) > 0:

    print(
        f"LONG setups       : "
        f"{(setups_df['direction'] == 'LONG').sum()}"
    )

    print(
        f"SHORT setups      : "
        f"{(setups_df['direction'] == 'SHORT').sum()}"
    )

    print()

    display_cols = [
        "event_timestamp",
        "event_type",
        "direction",
        "entry_timestamp",
        "entry_minutes_from_release",
        "recovery_pct",
    ]

    print(
        setups_df[display_cols]
        .to_string(index=False)
    )

print()

print("=" * 68)
print("TP / SL RESULTS")
print("=" * 68)

if len(summary_df) > 0:

    display_summary = summary_df.copy()

    numeric_cols = [
        "win_rate_pct",
        "profit_factor",
        "avg_gross_return_pct",
        "avg_net_return_pct",
        "median_net_return_pct",
        "total_net_return_pct",
        "avg_mfe_pct",
        "avg_mae_pct",
        "avg_hold_minutes",
    ]

    for col in numeric_cols:
        if col in display_summary:
            display_summary[col] = (
                display_summary[col].round(4)
            )

    print(
        display_summary.to_string(index=False)
    )

print()

print("=" * 68)
print("OUTPUT FILES")
print("=" * 68)

print(OUTPUT_SETUPS)
print(OUTPUT_MATRIX)
print(OUTPUT_SUMMARY)

print()

print("IMPORTANT:")
print("- Setup detection is sequential.")
print("- No future extreme is used to determine entry.")
print("- TP/SL simulation uses future candles only AFTER entry.")
print("- Results are exploratory, not proof of profitability.")