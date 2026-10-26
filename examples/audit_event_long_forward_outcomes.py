from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/btc")
SETUP_FILE = DATA / "BTCUSDT_binance_event_setups_60m_corrected.csv"
CANDLE_FILE = DATA / "BTCUSDT_1m_continuous_2020_2026.parquet"

OUTPUT_TRADES = DATA / "BTCUSDT_event_long_forward_outcomes_30m_60m.csv"
OUTPUT_SUMMARY = DATA / "BTCUSDT_event_long_forward_summary_30m_60m.csv"

ROUND_TRIP_COST_PCT = 0.14
RANGE_LOOKBACK = 60


def main():
    for path in (SETUP_FILE, CANDLE_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    for path in (OUTPUT_TRADES, OUTPUT_SUMMARY):
        if path.exists():
            raise FileExistsError(
                f"Output already exists; refusing to overwrite: {path}"
            )

    setups = pd.read_csv(SETUP_FILE)
    setups["entry_timestamp"] = pd.to_datetime(
        setups["entry_timestamp"], utc=True
    )

    setups = setups[
        setups["direction"].astype(str).str.upper() == "LONG"
    ].copy()

    candles = pd.read_parquet(CANDLE_FILE)
    candles["timestamp_utc"] = pd.to_datetime(
        candles["timestamp_utc"], utc=True
    )
    candles = candles.sort_values("timestamp_utc").reset_index(drop=True)

    if candles["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate candle timestamps detected.")

    if not candles["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("Candle timestamps are not sorted.")

    # Confirm one-minute continuity; do not silently fill missing candles.
    gaps = candles["timestamp_utc"].diff().dropna()
    gap_count = int((gaps != pd.Timedelta(minutes=1)).sum())
    if gap_count:
        raise ValueError(
            f"Found {gap_count} non-one-minute timestamp intervals. "
            "Resolve data continuity before running this audit."
        )

    candles = candles.set_index("timestamp_utc", drop=False)
    candle_times = candles.index
    records = []

    for _, setup in setups.iterrows():
        entry_time = setup["entry_timestamp"]

        # Entry must align exactly with a candle opening timestamp.
        entry_pos = candle_times.searchsorted(entry_time)
        if (
            entry_pos >= len(candles)
            or candle_times[entry_pos] != entry_time
        ):
            records.append({
                "event_timestamp": setup["event_timestamp"],
                "event_type": setup["event_type"],
                "entry_timestamp": entry_time,
                "direction": "LONG",
                "entry_price": setup["entry_price"],
                "status": "ENTRY_CANDLE_NOT_FOUND",
            })
            continue

        entry = candles.iloc[entry_pos]
        entry_price = float(entry["open"])

        if not np.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Invalid entry price at {entry_time}")

        row = {
            "event_timestamp": setup["event_timestamp"],
            "event_type": setup["event_type"],
            "shock_timestamp": setup.get("shock_timestamp"),
            "extreme_timestamp": setup.get("extreme_timestamp"),
            "confirmation_timestamp": setup.get("confirmation_timestamp"),
            "entry_timestamp": entry_time,
            "direction": "LONG",
            "entry_price_from_setup": float(setup["entry_price"]),
            "entry_open_from_candles": entry_price,
            "entry_price_difference_pct": (
                (entry_price / float(setup["entry_price"]) - 1) * 100
            ),
            "entry_minutes_from_release": setup["entry_minutes_from_release"],
            "recovery_pct": float(setup["recovery_pct"]),
            "status": "OK",
        }

        # Undervalue proxies use only candles BEFORE entry.
        # These are descriptive filters, not proof of fair value.
        if entry_pos >= RANGE_LOOKBACK:
            prior = candles.iloc[entry_pos - RANGE_LOOKBACK:entry_pos]
            prior_low = float(prior["low"].min())
            prior_high = float(prior["high"].max())
            prior_range = prior_high - prior_low

            row["prior_60m_low"] = prior_low
            row["prior_60m_high"] = prior_high
            row["distance_from_prior_60m_low_pct"] = (
                (entry_price / prior_low - 1) * 100
                if prior_low > 0 else np.nan
            )
            row["prior_60m_range_position_pct"] = (
                (entry_price - prior_low) / prior_range * 100
                if prior_range > 0 else np.nan
            )
        else:
            row["prior_60m_low"] = np.nan
            row["prior_60m_high"] = np.nan
            row["distance_from_prior_60m_low_pct"] = np.nan
            row["prior_60m_range_position_pct"] = np.nan

        for horizon in (30, 60):
            # Includes the entry candle and ends at the close of the
            # (horizon-th) one-minute candle: entry + horizon - 1.
            end_pos = entry_pos + horizon - 1

            if end_pos >= len(candles):
                row[f"return_{horizon}m_gross_pct"] = np.nan
                row[f"return_{horizon}m_net_est_pct"] = np.nan
                row[f"mfe_{horizon}m_pct"] = np.nan
                row[f"mae_{horizon}m_pct"] = np.nan
                row[f"close_timestamp_{horizon}m"] = None
                continue

            window = candles.iloc[entry_pos:end_pos + 1]
            final_candle = candles.iloc[end_pos]

            gross = (float(final_candle["close"]) / entry_price - 1) * 100
            mfe = (float(window["high"].max()) / entry_price - 1) * 100
            mae = (float(window["low"].min()) / entry_price - 1) * 100

            row[f"return_{horizon}m_gross_pct"] = gross
            row[f"return_{horizon}m_net_est_pct"] = (
                gross - ROUND_TRIP_COST_PCT
            )
            row[f"mfe_{horizon}m_pct"] = mfe
            row[f"mae_{horizon}m_pct"] = mae
            row[f"close_timestamp_{horizon}m"] = (
                final_candle["close_time_utc"]
            )

        records.append(row)

    results = pd.DataFrame(records)

    # Summary is descriptive only; no strategy thresholds are tuned here.
    summary_records = []
    valid = results[results["status"] == "OK"]

    for horizon in (30, 60):
        col = f"return_{horizon}m_net_est_pct"
        valid_horizon = valid[valid[col].notna()]
        returns = valid_horizon[col].astype(float)

        summary_records.append({
            "horizon_minutes": horizon,
            "valid_setups": len(valid_horizon),
            "positive_net_estimate_count": int((returns > 0).sum()),
            "negative_net_estimate_count": int((returns < 0).sum()),
            "zero_net_estimate_count": int((returns == 0).sum()),
            "positive_net_estimate_rate_pct": (
                float((returns > 0).mean() * 100)
                if len(returns) else np.nan
            ),
            "average_gross_return_pct": (
                float(valid_horizon[f"return_{horizon}m_gross_pct"].mean())
                if len(valid_horizon) else np.nan
            ),
            "average_net_estimate_pct": (
                float(returns.mean()) if len(returns) else np.nan
            ),
            "median_net_estimate_pct": (
                float(returns.median()) if len(returns) else np.nan
            ),
            "average_mfe_pct": (
                float(valid_horizon[f"mfe_{horizon}m_pct"].mean())
                if len(valid_horizon) else np.nan
            ),
            "average_mae_pct": (
                float(valid_horizon[f"mae_{horizon}m_pct"].mean())
                if len(valid_horizon) else np.nan
            ),
            "assumed_round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        })

    summary = pd.DataFrame(summary_records)

    results.to_csv(OUTPUT_TRADES, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("=== LONG FORWARD-OUTCOME AUDIT COMPLETE ===")
    print(f"LONG setups loaded: {len(setups)}")
    print(f"Outcome rows: {len(results)}")
    print(f"Entry rows not aligned: "
          f"{int((results['status'] != 'OK').sum())}")
    print(f"Input candle continuity: verified; gaps={gap_count}")
    print(f"Round-trip cost assumption: {ROUND_TRIP_COST_PCT:.3f}%")
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== OUTPUT FILES ===")
    print(OUTPUT_TRADES)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
