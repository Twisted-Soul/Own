
from pathlib import Path

import pandas as pd


# ============================================================
# Kronos chronological backtest
# Version: timestamp-gap audit fix
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "btc" / "BTCUSDT_5m_full_signals_experiment.csv"

TRADES_OUTPUT = (
    ROOT / "data" / "btc" / "BTCUSDT_chronological_trades.csv"
)
SUMMARY_OUTPUT = (
    ROOT / "data" / "btc" / "BTCUSDT_chronological_summary.csv"
)
AUDIT_OUTPUT = (
    ROOT / "data" / "btc" / "BTCUSDT_chronological_exclusions.csv"
)

# Original test assumptions
SPLIT_TIME = pd.Timestamp("2026-06-01T00:00:00Z")
EXPECTED_CANDLE_MINUTES = 5
MAX_HOLD_BARS = 12

TP_LEVELS = [0.10, 0.15, 0.20]  # Percent, not decimal
SL_LEVELS = [0.10, 0.15, 0.20]  # Percent, not decimal

# Round-trip trading costs, expressed as percentage points.
# Example: 0.14 means 0.14%, not 14%.
COST_LEVELS = [0.00, 0.04, 0.14]

# Predeclared configuration for the main trade output
SELECTED_TP = 0.20
SELECTED_SL = 0.20

VALID_SIGNALS = {"BUY", "SELL"}


# ============================================================
# Load and validate data
# ============================================================

def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{SOURCE}"
        )

    df = pd.read_csv(SOURCE)

    required = {
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "signal",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )

    if df["timestamps"].isna().any():
        count = int(df["timestamps"].isna().sum())
        raise ValueError(
            f"Found {count} invalid timestamps. "
            "Fix the input data before backtesting."
        )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(
            "OHLC columns contain missing or non-numeric values."
        )

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Found non-positive OHLC prices.")

    if (df["high"] < df["low"]).any():
        raise ValueError("Found candles where high is below low.")

    # Sort chronologically, but never silently deduplicate candles.
    df = df.sort_values("timestamps").reset_index(drop=True)

    if df["timestamps"].duplicated().any():
        examples = df.loc[
            df["timestamps"].duplicated(keep=False), "timestamps"
        ].head(10).tolist()
        raise ValueError(
            "Duplicate timestamps found. Resolve them before testing. "
            f"Examples: {examples}"
        )

    df["signal"] = (
        df["signal"].astype(str).str.strip().str.upper()
    )

    return df


def is_expected_interval(previous_time, current_time):
    """True only when two candles are exactly five minutes apart."""
    previous_time = pd.Timestamp(previous_time)
    current_time = pd.Timestamp(current_time)

    delta = current_time - previous_time
    expected = pd.Timedelta(minutes=EXPECTED_CANDLE_MINUTES)

    return delta == expected


# ============================================================
# Generate entries without crossing data gaps
# ============================================================

def make_entries(df):
    entries = []
    signals = df["signal"].tolist()
    times = df["timestamps"].tolist()

    for i in range(1, len(df) - 1):
        direction = signals[i]

        if direction not in VALID_SIGNALS:
            continue

        # Require two consecutive same-direction signals.
        if signals[i - 1] != direction:
            continue

        # Keep the original rule: only take the first pair
        # in a continuous run of identical signals.
        if i >= 2 and signals[i - 2] == direction:
            continue

        # The two signal candles must be consecutive.
        if not is_expected_interval(times[i - 1], times[i]):
            continue

        # Entry is at the next candle's open. Do not enter
        # if that next candle is separated by a data gap.
        if not is_expected_interval(times[i], times[i + 1]):
            continue

        entries.append({
            "signal_index": i,
            "entry_index": i + 1,
            "signal": direction,
        })

    return entries


# ============================================================
# Simulate one trade
# ============================================================

def simulate_one(df, entry, tp_pct, sl_pct, split_name):
    ei = entry["entry_index"]
    direction = 1 if entry["signal"] == "BUY" else -1

    entry_price = float(df.at[ei, "open"])
    entry_time = df.at[ei, "timestamps"]

    tp = tp_pct / 100.0
    sl = sl_pct / 100.0

    if direction == 1:
        target = entry_price * (1 + tp)
        stop = entry_price * (1 - sl)
    else:
        target = entry_price * (1 - tp)
        stop = entry_price * (1 + sl)

    last_checked = min(
        ei + MAX_HOLD_BARS - 1,
        len(df) - 1,
    )

    exit_price = None
    exit_index = None
    reason = None

    for j in range(ei, last_checked + 1):

        # CRITICAL FIX:
        # A trade cannot be simulated normally across a missing
        # candle. Exclude it instead of inventing a price path.
        if j > ei:
            previous_time = df.at[j - 1, "timestamps"]
            current_time = df.at[j, "timestamps"]

            if not is_expected_interval(previous_time, current_time):
                return {
                    "status": "EXCLUDED_DATA_GAP",
                    "exclude_index": j,
                    "exclude_time": current_time,
                    "split": split_name,
                    "signal_time": df.at[
                        entry["signal_index"], "timestamps"
                    ],
                    "entry_time": entry_time,
                    "signal": entry["signal"],
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "reason": "DATA_GAP_DURING_TRADE",
                }

        o = float(df.at[j, "open"])
        h = float(df.at[j, "high"])
        l = float(df.at[j, "low"])

        # First check opening gaps. If the candle opens beyond
        # a stop or target, use the opening price as the fill.
        if direction == 1:
            if o <= stop:
                exit_price, reason = o, "STOP_GAP"
            elif o >= target:
                exit_price, reason = o, "TARGET_GAP"
        else:
            if o >= stop:
                exit_price, reason = o, "STOP_GAP"
            elif o <= target:
                exit_price, reason = o, "TARGET_GAP"

        if reason is not None:
            exit_index = j
            break

        # Check the candle's high and low.
        if direction == 1:
            stop_hit = l <= stop
            target_hit = h >= target
        else:
            stop_hit = h >= stop
            target_hit = l <= target

        # Conservative assumption: if both are touched within
        # the same candle, assume the stop is hit first.
        if stop_hit and target_hit:
            exit_price, reason = stop, "BOTH_HIT_STOP_FIRST"
        elif stop_hit:
            exit_price, reason = stop, "STOP_LOSS"
        elif target_hit:
            exit_price, reason = target, "TAKE_PROFIT"

        if reason is not None:
            exit_index = j
            break

    if exit_price is None:
        # Exit at the open after MAX_HOLD_BARS.
        scheduled = ei + MAX_HOLD_BARS

        if scheduled >= len(df):
            return {
                "status": "EXCLUDED_INSUFFICIENT_FUTURE_DATA",
                "exclude_index": len(df) - 1,
                "exclude_time": df.at[len(df) - 1, "timestamps"],
                "split": split_name,
                "signal_time": df.at[
                    entry["signal_index"], "timestamps"
                ],
                "entry_time": entry_time,
                "signal": entry["signal"],
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "reason": "INSUFFICIENT_FUTURE_CANDLES",
            }

        # Also check the interval to the scheduled exit candle.
        # Otherwise a gap at the time exit could be missed.
        previous_time = df.at[scheduled - 1, "timestamps"]
        scheduled_time = df.at[scheduled, "timestamps"]

        if not is_expected_interval(previous_time, scheduled_time):
            return {
                "status": "EXCLUDED_DATA_GAP",
                "exclude_index": scheduled,
                "exclude_time": scheduled_time,
                "split": split_name,
                "signal_time": df.at[
                    entry["signal_index"], "timestamps"
                ],
                "entry_time": entry_time,
                "signal": entry["signal"],
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "reason": "DATA_GAP_BEFORE_TIME_EXIT",
            }

        exit_index = scheduled
        exit_price = float(df.at[exit_index, "open"])
        reason = "MAX_HOLD_TIME"

    gross = direction * (exit_price / entry_price - 1) * 100.0

    return {
        "status": "COMPLETED",
        "split": split_name,
        "signal_time": df.at[
            entry["signal_index"], "timestamps"
        ],
        "entry_time": entry_time,
        "exit_time": df.at[exit_index, "timestamps"],
        "signal": entry["signal"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "holding_minutes": (
            df.at[exit_index, "timestamps"] - entry_time
        ).total_seconds() / 60.0,
        "exit_reason": reason,
        "gross_return_pct": gross,
        "_entry_index": ei,
        "_exit_index": exit_index,
    }


# ============================================================
# Run trades sequentially without overlapping positions
# ============================================================

def run_non_overlapping(df, tp, sl, split_name):
    entries = make_entries(df)

    trades = []
    exclusions = []

    next_available_index = 0

    for entry in entries:
        if entry["entry_index"] < next_available_index:
            continue

        result = simulate_one(
            df, entry, tp, sl, split_name
        )

        if result["status"] == "COMPLETED":
            trades.append(result)
            next_available_index = result["_exit_index"] + 1

        else:
            # Report exclusions explicitly.
            exclusions.append(result)

            # If the simulated position encounters a gap, we
            # cannot know its true exit time. Do not start another
            # trade on the first candle after that gap.
            if result["status"] == "EXCLUDED_DATA_GAP":
                next_available_index = (
                    result["exclude_index"] + 1
                )
            else:
                # Near the end of the dataset, there may not be
                # enough future candles to complete the trade.
                next_available_index = len(df)

    return trades, exclusions


# ============================================================
# Performance metrics
# ============================================================

def profit_factor(returns):
    returns = pd.Series(returns, dtype=float)

    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def compounded_return_pct(returns):
    returns = pd.Series(returns, dtype=float)

    if returns.empty:
        return 0.0

    # Percentage returns are converted to decimal returns.
    return float(((1 + returns / 100.0).prod() - 1) * 100.0)


def summarize_trades(trades, exclusions, tp, sl, split_name):
    rows = []

    completed = [
        t for t in trades
        if t["split"] == split_name
        and t["tp_pct"] == tp
        and t["sl_pct"] == sl
    ]

    excluded = [
        e for e in exclusions
        if e["split"] == split_name
        and e["tp_pct"] == tp
        and e["sl_pct"] == sl
    ]

    gross = pd.Series(
        [t["gross_return_pct"] for t in completed],
        dtype=float,
    )

    for cost in COST_LEVELS:
        net = gross - cost

        rows.append({
            "split": split_name,
            "tp_pct": tp,
            "sl_pct": sl,
            "round_trip_cost_pct": cost,
            "completed_trades": len(completed),
            "excluded_trades": len(excluded),
            "excluded_data_gap": sum(
                e["status"] == "EXCLUDED_DATA_GAP"
                for e in excluded
            ),
            "excluded_insufficient_data": sum(
                e["status"] == "EXCLUDED_INSUFFICIENT_FUTURE_DATA"
                for e in excluded
            ),
            "gross_win_rate_pct": (
                float((gross > 0).mean() * 100.0)
                if len(gross) else 0.0
            ),
            "mean_gross_return_pct": (
                float(gross.mean()) if len(gross) else 0.0
            ),
            "mean_net_return_pct": (
                float(net.mean()) if len(net) else 0.0
            ),
            "gross_profit_factor": profit_factor(gross),
            "net_profit_factor": profit_factor(net),
            "compounded_net_return_pct": compounded_return_pct(net),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("KRONOS CHRONOLOGICAL BACKTEST — GAP-AWARE VERSION")
    print("=" * 68)

    df = load_data()

    print(f"Input file: {SOURCE}")
    print(f"Valid candles: {len(df):,}")
    print(f"First timestamp: {df['timestamps'].iloc[0]}")
    print(f"Last timestamp:  {df['timestamps'].iloc[-1]}")
    print(f"Development ends before: {SPLIT_TIME}")
    print(f"Maximum hold: {MAX_HOLD_BARS * EXPECTED_CANDLE_MINUTES} minutes")
    print()

    # The split is chronological. Each split is simulated
    # independently so a trade cannot cross the split boundary.
    development = df[df["timestamps"] < SPLIT_TIME].copy()
    holdout = df[df["timestamps"] >= SPLIT_TIME].copy()

    splits = {
        "DEVELOPMENT": development.reset_index(drop=True),
        "HOLDOUT": holdout.reset_index(drop=True),
    }

    all_trades = []
    all_exclusions = []
    all_summaries = []

    for tp in TP_LEVELS:
        for sl in SL_LEVELS:
            print(f"Testing TP={tp:.2f}% / SL={sl:.2f}%")

            config_trades = []
            config_exclusions = []

            for split_name, split_df in splits.items():
                trades, exclusions = run_non_overlapping(
                    split_df, tp, sl, split_name
                )

                config_trades.extend(trades)
                config_exclusions.extend(exclusions)

                print(
                    f"  {split_name}: "
                    f"{len(trades):,} completed, "
                    f"{len(exclusions):,} excluded"
                )

                all_summaries.extend(
                    summarize_trades(
                        trades, exclusions, tp, sl, split_name
                    )
                )

            all_trades.extend(config_trades)
            all_exclusions.extend(config_exclusions)

            print()

    # Main trades file contains the predeclared TP/SL configuration.
    selected_trades = [
        t for t in all_trades
        if t["tp_pct"] == SELECTED_TP
        and t["sl_pct"] == SELECTED_SL
    ]

    selected_exclusions = [
        e for e in all_exclusions
        if e["tp_pct"] == SELECTED_TP
        and e["sl_pct"] == SELECTED_SL
    ]

    trades_df = pd.DataFrame(selected_trades)
    exclusions_df = pd.DataFrame(selected_exclusions)
    summary_df = pd.DataFrame(all_summaries)

    # Internal row indexes are for simulation only.
    for column in ["_entry_index", "_exit_index"]:
        if column in trades_df.columns:
            trades_df.drop(columns=column, inplace=True)

    TRADES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(TRADES_OUTPUT, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    exclusions_df.to_csv(AUDIT_OUTPUT, index=False)

    print("=" * 68)
    print("RESULTS SAVED")
    print("=" * 68)
    print(f"Selected TP/SL: {SELECTED_TP:.2f}% / {SELECTED_SL:.2f}%")
    print(f"Completed selected trades: {len(trades_df):,}")
    print(f"Excluded selected trades: {len(exclusions_df):,}")
    print(f"Trades CSV:   {TRADES_OUTPUT}")
    print(f"Summary CSV:  {SUMMARY_OUTPUT}")
    print(f"Exclusions:   {AUDIT_OUTPUT}")
    print()

    if not exclusions_df.empty:
        print("Exclusion reasons:")
        print(
            exclusions_df["reason"]
            .value_counts(dropna=False)
            .to_string()
        )
        print()

    print("Selected configuration summary:")
    selected_summary = summary_df[
        (summary_df["tp_pct"] == SELECTED_TP)
        & (summary_df["sl_pct"] == SELECTED_SL)
    ]

    if not selected_summary.empty:
        print(selected_summary.to_string(index=False))

    print()
    print("IMPORTANT:")
    print("- Results are historical simulations, not forecasts.")
    print("- Costs are simplified round-trip percentage assumptions.")
    print("- Funding, leverage, liquidation, and exact execution")
    print("  behavior are not modeled by this script.")
    print("- Same-candle TP/SL ambiguity is resolved stop-first.")
    print("- Trades crossing timestamp gaps are excluded and logged.")


if __name__ == "__main__":
    main()
