import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
EVENTS_OUT = Path("data/btc/BTCUSDT_entry_quality_events.csv")
SUMMARY_OUT = Path("data/btc/BTCUSDT_entry_quality_summary.csv")

HORIZONS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "60m": 12,
}


def main():
    df = pd.read_csv(INPUT)

    required = {"timestamps", "open", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["signal"] = (
        df["signal"].fillna("HOLD").astype(str).str.upper().str.strip()
    )

    if not df["signal"].isin(["BUY", "SELL", "HOLD"]).all():
        raise ValueError("Unexpected signal values found.")

    df = (
        df.dropna(subset=["timestamps", "open"])
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )

    # Two consecutive matching directional signals are required.
    previous = df["signal"].shift(1)
    confirmed = df["signal"].where(
        df["signal"].isin(["BUY", "SELL"])
        & df["signal"].eq(previous),
        "HOLD",
    )

    # Keep only the beginning of each confirmed directional episode.
    # This avoids counting every candle in a prolonged signal as
    # a completely new, independent entry.
    previous_confirmed = confirmed.shift(1).fillna("HOLD")
    entry_mask = (
        confirmed.isin(["BUY", "SELL"])
        & confirmed.ne(previous_confirmed)
    )

    entries = []

    for i in df.index[entry_mask]:
        entry_index = i + 1

        # Signal is known at candle i close; hypothetical entry is
        # at the next candle open.
        if entry_index >= len(df):
            continue

        direction = 1 if confirmed.iloc[i] == "BUY" else -1
        entry_price = float(df["open"].iloc[entry_index])

        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        row = {
            "signal_time": df["timestamps"].iloc[i],
            "entry_time": df["timestamps"].iloc[entry_index],
            "signal": confirmed.iloc[i],
            "entry_index": entry_index,
            "entry_price": entry_price,
        }

        for label, bars in HORIZONS.items():
            exit_index = entry_index + bars

            if exit_index >= len(df):
                row[f"underlying_return_{label}_pct"] = np.nan
                row[f"signal_return_{label}_pct"] = np.nan
                continue

            exit_price = float(df["open"].iloc[exit_index])

            # Underlying return is the BTC price change.
            underlying_return = exit_price / entry_price - 1

            # Signal return is positive when the signal's direction
            # matches the observed price movement.
            signal_return = direction * underlying_return

            row[f"underlying_return_{label}_pct"] = (
                underlying_return * 100
            )
            row[f"signal_return_{label}_pct"] = (
                signal_return * 100
            )

        entries.append(row)

    events = pd.DataFrame(entries)

    if events.empty:
        raise ValueError(
            "No confirmed signal episodes found. Check signal values."
        )

    summaries = []

    for label in HORIZONS:
        col = f"signal_return_{label}_pct"

        for signal_name in ["ALL", "BUY", "SELL"]:
            if signal_name == "ALL":
                sample = events[col].dropna()
            else:
                sample = events.loc[
                    events["signal"] == signal_name, col
                ].dropna()

            gains = sample[sample > 0].sum()
            losses = -sample[sample < 0].sum()

            summaries.append({
                "horizon": label,
                "signal": signal_name,
                "samples": len(sample),
                "win_rate_pct": (
                    (sample > 0).mean() * 100
                    if len(sample) else np.nan
                ),
                "mean_signal_return_pct": (
                    sample.mean() if len(sample) else np.nan
                ),
                "median_signal_return_pct": (
                    sample.median() if len(sample) else np.nan
                ),
                "profit_factor": (
                    gains / losses if losses > 0
                    else np.nan
                ),
            })

    summary = pd.DataFrame(summaries)

    events.to_csv(EVENTS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n=== ENTRY QUALITY DIAGNOSTIC ===")
    print("Confirmed directional episodes:", len(events))
    print("BUY episodes:", (events["signal"] == "BUY").sum())
    print("SELL episodes:", (events["signal"] == "SELL").sum())
    print("Entry assumption: next candle open")
    print("Horizons: 5, 15, 30 and 60 minutes")
    print("Costs: not included")
    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print("\nSaved events:", EVENTS_OUT)
    print("Saved summary:", SUMMARY_OUT)


if __name__ == "__main__":
    main()
