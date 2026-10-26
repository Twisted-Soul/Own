from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/btc")
SETUPS_FILE = DATA / "BTCUSDT_binance_event_setups_60m_corrected.csv"
CANDLES_FILE = DATA / "BTCUSDT_1m_continuous_2020_2026.parquet"

ATR_PERIOD = 14
PRE_EVENT_CUTOFF_MINUTES = 5
MAX_HOLD_MINUTES = 60
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 3.0
COST_SCENARIOS = (0.14, 0.20)


def main():
    for path in (SETUPS_FILE, CANDLES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    setups = pd.read_csv(SETUPS_FILE)
    setups["event_timestamp"] = pd.to_datetime(
        setups["event_timestamp"], utc=True
    )
    setups["entry_timestamp"] = pd.to_datetime(
        setups["entry_timestamp"], utc=True
    )
    setups = setups[
        setups["direction"].astype(str).str.upper() == "LONG"
    ].copy().reset_index(drop=True)

    candles = pd.read_parquet(CANDLES_FILE)
    candles["timestamp_utc"] = pd.to_datetime(
        candles["timestamp_utc"], utc=True
    )
    candles["close_time_utc"] = pd.to_datetime(
        candles["close_time_utc"], utc=True
    )
    candles = candles.sort_values("timestamp_utc").reset_index(drop=True)

    times = pd.DatetimeIndex(candles["timestamp_utc"])
    if times.has_duplicates:
        raise ValueError("Duplicate candle timestamps found.")

    gaps = times.to_series().diff().dropna()
    gap_count = int((gaps != pd.Timedelta(minutes=1)).sum())
    if gap_count:
        raise ValueError(f"Found {gap_count} one-minute gaps; aborting.")

    # True range uses the previous candle close.
    prev_close = candles["close"].shift(1)
    tr = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - prev_close).abs(),
            (candles["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    candles["true_range"] = tr

    results = []

    for _, setup in setups.iterrows():
        event_time = setup["event_timestamp"]
        entry_time = setup["entry_timestamp"]

        # Only candles strictly before T-5 minutes are eligible.
        cutoff = event_time - pd.Timedelta(
            minutes=PRE_EVENT_CUTOFF_MINUTES
        )
        window_start = cutoff - pd.Timedelta(minutes=60)
        pre = candles[
            (candles["timestamp_utc"] >= window_start)
            & (candles["timestamp_utc"] < cutoff)
        ]

        if len(pre) != 60:
            raise ValueError(
                f"Expected 60 pre-event candles for {event_time}; "
                f"found {len(pre)}."
            )

        atr_values = pre["true_range"].tail(ATR_PERIOD)
        if len(atr_values) != ATR_PERIOD or not np.isfinite(
            atr_values.to_numpy()
        ).all():
            raise ValueError(f"Invalid ATR input for {event_time}.")

        atr = float(atr_values.mean())

        pos = times.searchsorted(entry_time)
        if pos >= len(candles) or times[pos] != entry_time:
            raise ValueError(f"Entry candle missing: {entry_time}")

        end_pos = pos + MAX_HOLD_MINUTES - 1
        if end_pos >= len(candles):
            raise ValueError(f"Insufficient future data: {entry_time}")

        entry = float(candles.iloc[pos]["open"])
        if not np.isfinite(entry) or entry <= 0 or atr <= 0:
            raise ValueError(f"Invalid entry or ATR: {entry_time}")

        sl_price = entry - SL_ATR_MULTIPLIER * atr
        tp_price = entry + TP_ATR_MULTIPLIER * atr

        exit_price = float(candles.iloc[end_pos]["close"])
        exit_reason = "TIME"
        ambiguous = False

        for j in range(pos, end_pos + 1):
            candle = candles.iloc[j]
            hit_sl = float(candle["low"]) <= sl_price
            hit_tp = float(candle["high"]) >= tp_price

            if hit_sl and hit_tp:
                exit_price = sl_price
                exit_reason = "SL_BOTH_TOUCHED"
                ambiguous = True
                break
            if hit_sl:
                exit_price = sl_price
                exit_reason = "SL"
                break
            if hit_tp:
                exit_price = tp_price
                exit_reason = "TP"
                break

        gross_pct = (exit_price / entry - 1) * 100

        results.append({
            "event_timestamp": event_time,
            "event_type": setup["event_type"],
            "entry_timestamp": entry_time,
            "entry_price": entry,
            "atr_price": atr,
            "atr_pct_of_entry": atr / entry * 100,
            "stop_price": sl_price,
            "target_price": tp_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "ambiguous_candle": ambiguous,
            "gross_return_pct": gross_pct,
        })

    trades = pd.DataFrame(results)
    if len(trades) != len(setups):
        raise ValueError("Trade count mismatch.")

    print("=== ATR EXIT EXPERIMENT V1 ===")
    print(f"Fixed LONG entries: {len(setups)}")
    print(f"ATR period: {ATR_PERIOD} one-minute true ranges")
    print("ATR input: 60-minute window ending before release T-5")
    print(f"Stop: {SL_ATR_MULTIPLIER} x ATR")
    print(f"Target: {TP_ATR_MULTIPLIER} x ATR")
    print(f"Maximum holding period: {MAX_HOLD_MINUTES} minutes")
    print(f"One-minute continuity: verified; gaps={gap_count}")
    print("Original entry rules unchanged; no files written.")

    print("\n=== SUMMARY BY COST ===")
    for cost in COST_SCENARIOS:
        net = trades["gross_return_pct"] - cost
        print(
            f"cost={cost:.2f}% | trades={len(net)} | "
            f"positive={int((net > 0).sum())}/{len(net)} | "
            f"win_rate={(net > 0).mean() * 100:.2f}% | "
            f"avg_gross={trades['gross_return_pct'].mean():.4f}% | "
            f"avg_net={net.mean():.4f}% | "
            f"median_net={net.median():.4f}% | "
            f"total_net={net.sum():.4f}%"
        )

    print("\n=== EXIT REASONS ===")
    print(trades["exit_reason"].value_counts().to_string())

    print("\n=== PER-TRADE RESULTS ===")
    cols = [
        "event_timestamp", "event_type", "atr_pct_of_entry",
        "exit_reason", "gross_return_pct",
    ]
    print(trades[cols].round(4).to_string(index=False))
    print("\nATR experiment complete; no files written.")


if __name__ == "__main__":
    main()
