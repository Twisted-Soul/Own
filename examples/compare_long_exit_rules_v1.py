from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/btc")
SETUPS_FILE = DATA / "BTCUSDT_binance_event_setups_60m_corrected.csv"
CANDLES_FILE = DATA / "BTCUSDT_1m_continuous_2020_2026.parquet"

COST_PCT = 0.14
MAX_HOLD = 60

POLICIES = [
    {"name": "Fixed 30m", "tp": None, "sl": None, "hold": 30},
    {"name": "Fixed 60m", "tp": None, "sl": None, "hold": 60},
    {"name": "TP30_SL30", "tp": 0.30, "sl": 0.30, "hold": 60},
    {"name": "TP50_SL30", "tp": 0.50, "sl": 0.30, "hold": 60},
    {"name": "TP50_SL50", "tp": 0.50, "sl": 0.50, "hold": 60},
    {"name": "TP75_SL50", "tp": 0.75, "sl": 0.50, "hold": 60},
]


def main():
    for path in (SETUPS_FILE, CANDLES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    setups = pd.read_csv(SETUPS_FILE)
    setups["entry_timestamp"] = pd.to_datetime(
        setups["entry_timestamp"], utc=True
    )
    setups = setups[
        setups["direction"].astype(str).str.upper() == "LONG"
    ].copy()

    candles = pd.read_parquet(CANDLES_FILE)
    candles["timestamp_utc"] = pd.to_datetime(
        candles["timestamp_utc"], utc=True
    )
    candles = candles.sort_values("timestamp_utc").reset_index(drop=True)

    if candles["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate candle timestamps found.")

    gaps = candles["timestamp_utc"].diff().dropna()
    gap_count = int((gaps != pd.Timedelta(minutes=1)).sum())
    if gap_count:
        raise ValueError(f"Found {gap_count} timestamp gaps. Aborting.")

    times = pd.DatetimeIndex(candles["timestamp_utc"])
    trade_rows = []

    for _, setup in setups.iterrows():
        entry_time = setup["entry_timestamp"]
        pos = times.searchsorted(entry_time)

        if pos >= len(candles) or times[pos] != entry_time:
            raise ValueError(f"Entry candle not found: {entry_time}")

        entry_price = float(candles.iloc[pos]["open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Invalid entry price at {entry_time}")

        for policy in POLICIES:
            hold = policy["hold"]
            end_pos = pos + hold - 1

            if end_pos >= len(candles):
                raise ValueError(f"Insufficient future candles at {entry_time}")

            exit_reason = "TIME"
            exit_price = float(candles.iloc[end_pos]["close"])
            exit_time = candles.iloc[end_pos]["close_time_utc"]
            hit_both_same_candle = False

            if policy["tp"] is not None:
                tp_price = entry_price * (1 + policy["tp"] / 100)
                sl_price = entry_price * (1 - policy["sl"] / 100)

                for j in range(pos, end_pos + 1):
                    candle = candles.iloc[j]
                    hit_tp = float(candle["high"]) >= tp_price
                    hit_sl = float(candle["low"]) <= sl_price

                    if hit_tp and hit_sl:
                        # Conservative assumption: stop happens first.
                        exit_reason = "SL_BOTH_TOUCHED"
                        exit_price = sl_price
                        exit_time = candle["close_time_utc"]
                        hit_both_same_candle = True
                        break
                    elif hit_sl:
                        exit_reason = "SL"
                        exit_price = sl_price
                        exit_time = candle["close_time_utc"]
                        break
                    elif hit_tp:
                        exit_reason = "TP"
                        exit_price = tp_price
                        exit_time = candle["close_time_utc"]
                        break

            gross_pct = (exit_price / entry_price - 1) * 100
            net_est_pct = gross_pct - COST_PCT

            trade_rows.append({
                "event_timestamp": setup["event_timestamp"],
                "event_type": setup["event_type"],
                "entry_timestamp": entry_time,
                "policy": policy["name"],
                "entry_price": entry_price,
                "exit_timestamp": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "both_levels_touched_same_candle": hit_both_same_candle,
                "gross_return_pct": gross_pct,
                "net_est_return_pct": net_est_pct,
                "hold_limit_minutes": hold,
            })

    trades = pd.DataFrame(trade_rows)

    print("=== LONG EXIT-RULE EXPERIMENT ===")
    print(f"Fixed LONG setups: {len(setups)}")
    print(f"Policies tested: {len(POLICIES)}")
    print(f"Trade-policy outcomes: {len(trades)}")
    print(f"One-minute continuity: verified; gaps={gap_count}")
    print(f"Round-trip cost assumption: {COST_PCT:.3f}%")
    print("No files written; baseline and entry rules unchanged.")

    summary = trades.groupby("policy", sort=False).agg(
        trades=("net_est_return_pct", "count"),
        positive_net_count=("net_est_return_pct", lambda s: int((s > 0).sum())),
        positive_net_rate_pct=("net_est_return_pct", lambda s: (s > 0).mean() * 100),
        average_gross_pct=("gross_return_pct", "mean"),
        average_net_est_pct=("net_est_return_pct", "mean"),
        median_net_est_pct=("net_est_return_pct", "median"),
        total_net_est_pct=("net_est_return_pct", "sum"),
        stop_exits=("exit_reason", lambda s: int(s.str.startswith("SL").sum())),
        target_exits=("exit_reason", lambda s: int((s == "TP").sum())),
        time_exits=("exit_reason", lambda s: int((s == "TIME").sum())),
        ambiguous_candles=("both_levels_touched_same_candle", "sum"),
    )

    print("\n=== POLICY SUMMARY ===")
    print(summary.round(4).to_string())

    print("\n=== PER-TRADE RESULTS ===")
    print(trades[
        [
            "event_timestamp", "event_type", "policy",
            "exit_reason", "gross_return_pct", "net_est_return_pct",
        ]
    ].round(4).to_string(index=False))

    print("\nRead-only experiment complete.")


if __name__ == "__main__":
    main()
