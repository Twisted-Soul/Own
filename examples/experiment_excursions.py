import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_5m_signals.csv")
EVENTS_OUT = Path("data/btc/BTCUSDT_excursion_events.csv")
SUMMARY_OUT = Path("data/btc/BTCUSDT_excursion_summary.csv")

HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}


def main():
    df = pd.read_csv(INPUT)

    required = {"timestamps", "open", "high", "low", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            "Check the source CSV before proceeding."
        )

    df["timestamps"] = pd.to_datetime(
        df["timestamps"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["signal"] = (
        df["signal"].fillna("HOLD").astype(str).str.upper().str.strip()
    )

    df = (
        df.dropna(subset=["timestamps", "open", "high", "low"])
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )

    previous = df["signal"].shift(1)
    confirmed = df["signal"].where(
        df["signal"].isin(["BUY", "SELL"])
        & df["signal"].eq(previous),
        "HOLD",
    )

    prior_confirmed = confirmed.shift(1).fillna("HOLD")
    entry_mask = (
        confirmed.isin(["BUY", "SELL"])
        & confirmed.ne(prior_confirmed)
    )

    records = []

    for signal_index in df.index[entry_mask]:
        entry_index = signal_index + 1

        if entry_index >= len(df):
            continue

        direction = 1 if confirmed.iloc[signal_index] == "BUY" else -1
        entry = float(df.at[entry_index, "open"])

        if not np.isfinite(entry) or entry <= 0:
            continue

        row = {
            "signal_time": df.at[signal_index, "timestamps"],
            "entry_time": df.at[entry_index, "timestamps"],
            "signal": confirmed.iloc[signal_index],
            "entry_price": entry,
        }

        for label, bars in HORIZONS.items():
            # Include the entry candle and the next bars-1 candles.
            end = entry_index + bars
            window = df.iloc[entry_index:min(end, len(df))]

            if len(window) != bars:
                row[f"mfe_{label}_pct"] = np.nan
                row[f"mae_{label}_pct"] = np.nan
                continue

            highest = float(window["high"].max())
            lowest = float(window["low"].min())

            if direction == 1:
                mfe = (highest / entry - 1) * 100
                mae = (lowest / entry - 1) * 100
            else:
                mfe = (entry - lowest) / entry * 100
                mae = (entry - highest) / entry * 100

            row[f"mfe_{label}_pct"] = mfe
            row[f"mae_{label}_pct"] = mae

        records.append(row)

    events = pd.DataFrame(records)

    if events.empty:
        raise ValueError("No confirmed entries found.")

    summaries = []

    for label in HORIZONS:
        mfe_col = f"mfe_{label}_pct"
        mae_col = f"mae_{label}_pct"

        for side in ["ALL", "BUY", "SELL"]:
            sample = events if side == "ALL" else events[
                events["signal"] == side
            ]

            mfe = sample[mfe_col].dropna()
            mae = sample[mae_col].dropna()

            summaries.append({
                "horizon": label,
                "signal": side,
                "samples": len(mfe),
                "mean_mfe_pct": mfe.mean(),
                "median_mfe_pct": mfe.median(),
                "mean_mae_pct": mae.mean(),
                "median_mae_pct": mae.median(),
                "mfe_ge_0_10pct_pct": (
                    (mfe >= 0.10).mean() * 100 if len(mfe) else np.nan
                ),
                "mfe_ge_0_20pct_pct": (
                    (mfe >= 0.20).mean() * 100 if len(mfe) else np.nan
                ),
            })

    summary = pd.DataFrame(summaries)
    events.to_csv(EVENTS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n=== MFE / MAE DIAGNOSTIC ===")
    print("Confirmed episodes:", len(events))
    print("Entry assumption: next candle open")
    print("MFE = best favorable excursion; MAE = worst adverse excursion")
    print("Excursions are hindsight measurements, not realized profits.")
    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print("\nSaved events:", EVENTS_OUT)
    print("Saved summary:", SUMMARY_OUT)


if __name__ == "__main__":
    main()
