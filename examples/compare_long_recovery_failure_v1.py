from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/btc")
SETUPS_FILE = DATA / "BTCUSDT_binance_event_setups_60m_corrected.csv"
CANDLES_FILE = DATA / "BTCUSDT_1m_continuous_2020_2026.parquet"

ATR_PERIOD = 14
CUTOFF_MINUTES = 5
MAX_HOLD = 60
RECOVERY_MINUTES = 20
RECOVERY_ATR_MULTIPLIER = 0.5
COSTS = (0.14, 0.20)


def main():
    setups = pd.read_csv(SETUPS_FILE)
    candles = pd.read_parquet(CANDLES_FILE)

    setups["event_timestamp"] = pd.to_datetime(
        setups["event_timestamp"], utc=True
    )
    setups["entry_timestamp"] = pd.to_datetime(
        setups["entry_timestamp"], utc=True
    )
    candles["timestamp_utc"] = pd.to_datetime(
        candles["timestamp_utc"], utc=True
    )
    candles = candles.sort_values("timestamp_utc").reset_index(drop=True)

    setups = setups[
        setups["direction"].astype(str).str.upper() == "LONG"
    ].copy().reset_index(drop=True)

    times = pd.DatetimeIndex(candles["timestamp_utc"])
    if times.has_duplicates:
        raise ValueError("Duplicate candle timestamps found.")
    gaps = times.to_series().diff().dropna()
    if (gaps != pd.Timedelta(minutes=1)).any():
        raise ValueError("One-minute candle gaps found; aborting.")

    prev_close = candles["close"].shift(1)
    candles["true_range"] = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - prev_close).abs(),
            (candles["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    results = []

    for _, setup in setups.iterrows():
        event_time = setup["event_timestamp"]
        entry_time = setup["entry_timestamp"]

        cutoff = event_time - pd.Timedelta(minutes=CUTOFF_MINUTES)
        pre = candles[
            (candles["timestamp_utc"] >= cutoff - pd.Timedelta(minutes=60))
            & (candles["timestamp_utc"] < cutoff)
        ]
        if len(pre) != 60:
            raise ValueError(f"Expected 60 pre-event candles: {event_time}")

        atr = float(pre["true_range"].tail(ATR_PERIOD).mean())
        pos = times.searchsorted(entry_time)

        if pos >= len(candles) or times[pos] != entry_time:
            raise ValueError(f"Entry candle missing: {entry_time}")
        if pos + MAX_HOLD > len(candles):
            raise ValueError(f"Insufficient future candles: {entry_time}")

        entry = float(candles.iloc[pos]["open"])
        if not np.isfinite(atr) or atr <= 0 or entry <= 0:
            raise ValueError(f"Invalid ATR or entry: {entry_time}")

        recovery_level = entry + RECOVERY_ATR_MULTIPLIER * atr

        # First 20 entry-minute candles, including the entry candle.
        early = candles.iloc[pos:pos + RECOVERY_MINUTES]
        recovered = bool((early["high"] >= recovery_level).any())

        # Same 60-minute baseline exit for every entry.
        baseline_exit = float(
            candles.iloc[pos + MAX_HOLD - 1]["close"]
        )

        if recovered:
            test_exit = baseline_exit
            reason = "RECOVERED_HOLD_60M"
        else:
            test_exit = float(early.iloc[-1]["close"])
            reason = "FAILED_EXIT_20M"

        results.append({
            "event_timestamp": event_time,
            "event_type": setup["event_type"],
            "entry_timestamp": entry_time,
            "atr_pct": atr / entry * 100,
            "recovered_within_20m": recovered,
            "baseline_gross_pct": (baseline_exit / entry - 1) * 100,
            "test_gross_pct": (test_exit / entry - 1) * 100,
            "exit_reason": reason,
        })

    df = pd.DataFrame(results)

    print("=== RECOVERY-FAILURE EXIT EXPERIMENT ===")
    print(f"Fixed LONG entries: {len(df)}")
    print(f"Recovery test: entry + {RECOVERY_ATR_MULTIPLIER} x pre-event ATR")
    print(f"Recovery window: {RECOVERY_MINUTES} minutes")
    print(f"Maximum hold: {MAX_HOLD} minutes")
    print("Entry rules unchanged; one-minute continuity verified.")
    print("No files written.")

    print("\n=== POLICY COMPARISON ===")
    for cost in COSTS:
        for label, col in [
            ("FIXED_60M", "baseline_gross_pct"),
            ("RECOVERY_FAILURE", "test_gross_pct"),
        ]:
            gross = df[col]
            net = gross - cost
            print(
                f"cost={cost:.2f}% | policy={label} | "
                f"trades={len(net)} | positive={(net > 0).sum()}/{len(net)} | "
                f"avg_gross={gross.mean():.4f}% | "
                f"avg_net={net.mean():.4f}% | "
                f"median_net={net.median():.4f}% | "
                f"total_net={net.sum():.4f}%"
            )

    print("\n=== EXIT REASONS ===")
    print(df["exit_reason"].value_counts().to_string())

    print("\n=== PER-TRADE RESULTS ===")
    print(df[
        ["event_timestamp", "event_type", "recovered_within_20m",
         "baseline_gross_pct", "test_gross_pct", "exit_reason"]
    ].round(4).to_string(index=False))


if __name__ == "__main__":
    for path in (SETUPS_FILE, CANDLES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    main()