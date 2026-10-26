
from pathlib import Path
from platform import release

import numpy as np
import pandas as pd


# ============================================================
# CONFIG — FROZEN ORIGINAL STRATEGY
# ============================================================

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "btc"

EVENT_FILE = DATA / "BTCUSDT_1m_event_history.csv"
CANDLE_FILE = DATA / "BTCUSDT_1m_continuous_2020_2026.parquet"

OUTPUT_AUDIT = DATA / "BTCUSDT_binance_event_coverage.csv"
OUTPUT_SETUPS = DATA / "BTCUSDT_binance_event_setups.csv"
OUTPUT_MATRIX = DATA / "BTCUSDT_binance_event_tp_sl_matrix.csv"
OUTPUT_SUMMARY = DATA / "BTCUSDT_binance_event_summary.csv"

SHOCK_THRESHOLD = 0.0010
RECOVERY_THRESHOLD = 0.0020
MIN_ENTRY_MINUTE = 5
MAX_ENTRY_MINUTE = 60
CONFIRM_LOOKBACK = 3
MAX_HOLD_MINUTES = 60
ROUND_TRIP_COST = 0.0014

TP_LEVELS = [0.0020, 0.0030, 0.0050, 0.0075, 0.0100]
SL_LEVELS = [0.0020, 0.0030, 0.0050]

# Recreate the original event window: -89 through +179 minutes.
WINDOW_BEFORE = 89
WINDOW_AFTER = 179
EXPECTED_CANDLES = WINDOW_BEFORE + WINDOW_AFTER + 1


def pct(value):
    return value * 100.0


def confirmation_long(df, pos):
    if pos < CONFIRM_LOOKBACK:
        return False
    previous = df.iloc[pos - CONFIRM_LOOKBACK:pos]
    return df.iloc[pos]["close"] > previous["high"].max()


def confirmation_short(df, pos):
    if pos < CONFIRM_LOOKBACK:
        return False
    previous = df.iloc[pos - CONFIRM_LOOKBACK:pos]
    return df.iloc[pos]["close"] < previous["low"].min()


def find_entry(event_df):
    """Sequential setup detection; no future extreme is used."""
    df = event_df.reset_index(drop=True).copy()

    if len(df) == 0:
        return None

    df["return_5m"] = df["close"] / df["close"].shift(5) - 1.0

    direction = None
    running_extreme = None
    shock_pos = None
    extreme_pos = None

    release_timestamp = df.iloc[0]["event_timestamp"]

    for pos in range(len(df)):
        row = df.iloc[pos]
        minute = row["minutes_from_release"]

        if minute < MIN_ENTRY_MINUTE:
            continue
        if minute > MAX_ENTRY_MINUTE:
            break

        rolling_return = row["return_5m"]

        # First qualifying shock locks the direction.
        if direction is None:
            if rolling_return <= -SHOCK_THRESHOLD:
                direction = "LONG"
                shock_pos = pos
                running_extreme = row["low"]
                extreme_pos = pos
                continue

            if rolling_return >= SHOCK_THRESHOLD:
                direction = "SHORT"
                shock_pos = pos
                running_extreme = row["high"]
                extreme_pos = pos
                continue

            continue

        if direction == "LONG":
            if row["low"] < running_extreme:
                running_extreme = row["low"]
                extreme_pos = pos

            recovery = row["close"] / running_extreme - 1.0
            if recovery < RECOVERY_THRESHOLD:
                continue
            if not confirmation_long(df, pos):
                continue

        else:
            if row["high"] > running_extreme:
                running_extreme = row["high"]
                extreme_pos = pos

            recovery = running_extreme / row["close"] - 1.0
            if recovery < RECOVERY_THRESHOLD:
                continue
            if not confirmation_short(df, pos):
                continue

        # Enter at the next candle's open.
        if pos + 1 >= len(df):
            continue

        entry_pos = pos + 1
        entry_row = df.iloc[entry_pos]

        return {
            "direction": direction,
            "shock_timestamp": df.iloc[shock_pos]["timestamp"],
            "extreme_timestamp": df.iloc[extreme_pos]["timestamp"],
            "confirmation_timestamp": row["timestamp"],
            "entry_timestamp": entry_row["timestamp"],
            "shock_price": df.iloc[shock_pos]["close"],
            "extreme_price": running_extreme,
            "confirmation_price": row["close"],
            "entry_price": entry_row["open"],
            "recovery_pct": recovery,
            "entry_minutes_from_release": entry_row["minutes_from_release"],
            "entry_pos": entry_pos,
            "release_timestamp": release_timestamp,
        }

    return None



def simulate_trade(event_df, setup, tp, sl):
    """Simulate exits using candles after entry."""
    df = event_df.reset_index(drop=True)
    entry_pos = setup["entry_pos"]
    entry_price = setup["entry_price"]
    direction = setup["direction"]

    # Preserves the original script's inclusive holding horizon.
    max_exit_pos = min(entry_pos + MAX_HOLD_MINUTES, len(df) - 1)

    if direction == "LONG":
        tp_price = entry_price * (1.0 + tp)
        sl_price = entry_price * (1.0 - sl)
    else:
        tp_price = entry_price * (1.0 - tp)
        sl_price = entry_price * (1.0 + sl)

    exit_reason = "TIME"
    exit_pos = max_exit_pos
    exit_price = df.iloc[max_exit_pos]["close"]

    for pos in range(entry_pos, max_exit_pos + 1):
        row = df.iloc[pos]

        candle_open = float(row["open"])
        candle_high = float(row["high"])
        candle_low = float(row["low"])

        if direction == "LONG":
            if candle_open <= sl_price:
                exit_reason, exit_pos, exit_price = "SL", pos, candle_open
                break

            if candle_open >= tp_price:
                exit_reason, exit_pos, exit_price = "TP", pos, tp_price
                break

            hit_tp = candle_high >= tp_price
            hit_sl = candle_low <= sl_price
        else:
            if candle_open >= sl_price:
                exit_reason, exit_pos, exit_price = "SL", pos, candle_open
                break

            if candle_open <= tp_price:
                exit_reason, exit_pos, exit_price = "TP", pos, tp_price
                break

            hit_tp = candle_low <= tp_price
            hit_sl = candle_high >= sl_price

        # If both levels are touched, assume the stop is hit first.
        if hit_tp and hit_sl:
            exit_reason, exit_pos, exit_price = "SL", pos, sl_price
            break

        if hit_sl:
            exit_reason, exit_pos, exit_price = "SL", pos, sl_price
            break

        if hit_tp:
            exit_reason, exit_pos, exit_price = "TP", pos, tp_price
            break

    if direction == "LONG":
        gross_return = exit_price / entry_price - 1.0
    else:
        gross_return = entry_price / exit_price - 1.0

    net_return = gross_return - ROUND_TRIP_COST

    # Measure excursions from entry through the exit candle.
    trade_path = df.iloc[entry_pos:exit_pos + 1]

    if direction == "LONG":
        mfe = trade_path["high"].max() / entry_price - 1.0
        mae = trade_path["low"].min() / entry_price - 1.0
    else:
        mfe = entry_price / trade_path["low"].min() - 1.0
        mae = entry_price / trade_path["high"].max() - 1.0

    hold_minutes = (
        df.iloc[exit_pos]["minutes_from_release"]
        - df.iloc[entry_pos]["minutes_from_release"]
    )

    return {
        "tp_pct": pct(tp),
        "sl_pct": pct(sl),
        "exit_timestamp": df.iloc[exit_pos]["timestamp"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return_pct": pct(gross_return),
        "net_return_pct": pct(net_return),
        "mfe_pct": pct(mfe),
        "mae_pct": pct(mae),
        "hold_minutes": hold_minutes,
        "full_horizon": (
            exit_pos >= entry_pos + MAX_HOLD_MINUTES
            or exit_pos == len(df) - 1
        ),
    }


def main():
    print("=" * 72)
    print("BINANCE BTCUSDT 1m EVENT STRATEGY REPLICATION")
    print("Frozen original parameters; no threshold tuning")
    print("=" * 72)

    events_df = pd.read_csv(EVENT_FILE)
    events_df["event_timestamp"] = pd.to_datetime(
        events_df["event_timestamp"], utc=True
    )

    events = (
        events_df[["event_timestamp", "event_type"]]
        .drop_duplicates()
        .sort_values(["event_timestamp", "event_type"])
        .reset_index(drop=True)
    )

    candles = pd.read_parquet(
        CANDLE_FILE,
        columns=["timestamp_utc", "open", "high", "low", "close"]
    )
    candles = candles.rename(columns={"timestamp_utc": "timestamp"})
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    candles = candles.sort_values("timestamp").reset_index(drop=True)

    if candles["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in Binance candles.")
    if not candles["timestamp"].is_monotonic_increasing:
        raise ValueError("Binance candles are not chronologically sorted.")

    candle_times = candles["timestamp"]
    candle_start = candle_times.min()
    candle_end = candle_times.max()

    # Exact timestamp lookup makes event windows quick to construct.
    candles = candles.set_index("timestamp", drop=False)

    audit_rows = []
    setup_rows = []
    trade_rows = []

    for _, event in events.iterrows():
        release = event["event_timestamp"]
        event_type = event["event_type"]

        start = release - pd.Timedelta(WINDOW_BEFORE, unit="min")
        end = release + pd.Timedelta(WINDOW_AFTER, unit="min")

        in_bounds = start >= candle_start and end <= candle_end

        if in_bounds:
            window = candles.loc[start:end].copy()
            window = window.reset_index(drop=True)
        else:
            window = pd.DataFrame()

        # Check count and exact one-minute continuity.
        complete = (
            len(window) == EXPECTED_CANDLES
            and window["timestamp"].iloc[0] == start
            and window["timestamp"].iloc[-1] == end
            and window["timestamp"].diff().dropna().eq(
                pd.Timedelta(1, unit="min")
            ).all()
        )

        audit_rows.append({
            "event_timestamp": release,
            "event_type": event_type,
            "window_start": start,
            "window_end": end,
            "candles_in_window": len(window),
            "expected_candles": EXPECTED_CANDLES,
            "full_window_by_bounds": in_bounds,
            "window_complete": complete,
            "status": "READY" if complete else "EXCLUDED_INCOMPLETE_WINDOW",
        })

        if not complete:
            continue

        window["event_timestamp"] = release
        window["event_type"] = event_type
        window["minutes_from_release"] = (
            (window["timestamp"] - release).dt.total_seconds() / 60
        ).astype(int)

        setup = find_entry(window)
        if setup is None:
            continue

        setup_rows.append({
            "data_source": "BINANCE_USDT_PERPETUAL_DATASET",
            "event_timestamp": release,
            "event_type": event_type,
            "direction": setup["direction"],
            "shock_timestamp": setup["shock_timestamp"],
            "extreme_timestamp": setup["extreme_timestamp"],
            "confirmation_timestamp": setup["confirmation_timestamp"],
            "entry_timestamp": setup["entry_timestamp"],
            "shock_price": setup["shock_price"],
            "extreme_price": setup["extreme_price"],
            "confirmation_price": setup["confirmation_price"],
            "entry_price": setup["entry_price"],
            "recovery_pct": pct(setup["recovery_pct"]),
            "entry_minutes_from_release": setup["entry_minutes_from_release"],
        })

        for tp in TP_LEVELS:
            for sl in SL_LEVELS:
                result = simulate_trade(window, setup, tp, sl)
                trade_rows.append({
                    "data_source": "BINANCE_USDT_PERPETUAL_DATASET",
                    "event_timestamp": release,
                    "event_type": event_type,
                    "direction": setup["direction"],
                    "entry_timestamp": setup["entry_timestamp"],
                    "entry_price": setup["entry_price"],
                    "entry_minutes_from_release": setup["entry_minutes_from_release"],
                    "recovery_pct": pct(setup["recovery_pct"]),
                    **result,
                })

    audit_df = pd.DataFrame(audit_rows)
    setups_df = pd.DataFrame(setup_rows)
    trades_df = pd.DataFrame(trade_rows)

    audit_df.to_csv(OUTPUT_AUDIT, index=False)
    setups_df.to_csv(OUTPUT_SETUPS, index=False)
    trades_df.to_csv(OUTPUT_MATRIX, index=False)

    summary_rows = []
    if not trades_df.empty:
        for (tp, sl), group in trades_df.groupby(["tp_pct", "sl_pct"]):
            wins = group[group["exit_reason"] == "TP"]
            losses = group[group["exit_reason"] == "SL"]
            times = group[group["exit_reason"] == "TIME"]


            # Profit factors calculated separately before and after costs.
            gross_positive = group.loc[
                group["gross_return_pct"] > 0, "gross_return_pct"
            ].sum()
            gross_negative = abs(group.loc[
                group["gross_return_pct"] < 0, "gross_return_pct"
            ].sum())

            net_positive = group.loc[
                group["net_return_pct"] > 0, "net_return_pct"
            ].sum()
            net_negative = abs(group.loc[
                group["net_return_pct"] < 0, "net_return_pct"
            ].sum())

            gross_pf = (
                gross_positive / gross_negative
                if gross_negative > 0 else np.inf
            )
            net_pf = (
                net_positive / net_negative
                if net_negative > 0 else np.inf
            )

            summary_rows.append({
                "data_source": "BINANCE_USDT_PERPETUAL_DATASET",
                "tp_pct": tp,
                "sl_pct": sl,
                "trades": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "time_exits": len(times),
                "win_rate_pct": len(wins) / len(group) * 100,
                "gross_profit_factor": gross_pf,
                "net_profit_factor": net_pf,
                "avg_gross_return_pct": group["gross_return_pct"].mean(),
                "avg_net_return_pct": group["net_return_pct"].mean(),
                "median_net_return_pct": group["net_return_pct"].median(),
                "total_net_return_pct": group["net_return_pct"].sum(),
                "avg_mfe_pct": group["mfe_pct"].mean(),
                "avg_mae_pct": group["mae_pct"].mean(),
                "avg_hold_minutes": group["hold_minutes"].mean(),
                "full_horizon_pct": group["full_horizon"].mean() * 100,
            })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            "avg_net_return_pct", ascending=False
        )
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\nDataset range: {candle_start} to {candle_end}")
    print(f"Events in original history: {len(events)}")
    print(f"Complete event windows: {int(audit_df['window_complete'].sum())}/{len(audit_df)}")
    print(f"Events with detected setups: {len(setups_df)}")
    print(f"LONG setups: {(setups_df['direction'] == 'LONG').sum() if not setups_df.empty else 0}")
    print(f"SHORT setups: {(setups_df['direction'] == 'SHORT').sum() if not setups_df.empty else 0}")

    print("\nEvent coverage:")
    print(audit_df[["event_timestamp", "event_type", "candles_in_window", "status"]].to_string(index=False))

    print("\nTP/SL results:")
    if summary_df.empty:
        print("No trades were generated. Check the event windows and input files.")
    else:
        print(summary_df.round(4).to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_AUDIT)
    print(OUTPUT_SETUPS)
    print(OUTPUT_MATRIX)
    print(OUTPUT_SUMMARY)

    print("\nCaution: exploratory replication only, not evidence of profitability.")


if __name__ == "__main__":
    main()