import pandas as pd
import numpy as np
from pathlib import Path


BLS_FILE = Path("data/bls_economic_releases.csv")
BTC_FILE = Path("data/btc/BTCUSDT_5m_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_economic_features.csv")


def main():
    print("Loading BLS release data...")
    econ = pd.read_csv(BLS_FILE)

    econ["release_time"] = pd.to_datetime(
        econ["release_time"],
        utc=True,
    )

    econ["observation_period"] = pd.to_datetime(
        econ["observation_period"],
        utc=True,
    )

    # Remove rows where no official release timestamp is available.
    econ = econ.dropna(subset=["release_time"]).copy()

    econ = econ.sort_values("release_time")

    print(f"Usable economic releases: {len(econ)}")

    # Calculate changes within each economic series.
    econ["change_from_previous"] = (
        econ.groupby("event")["value"].diff()
    )

    econ["pct_change"] = (
        econ.groupby("event")["value"]
        .pct_change()
    )

    print("\nEconomic release summary:")
    print(
        econ[
            [
                "event",
                "observation_period",
                "value",
                "change_from_previous",
                "pct_change",
                "release_time",
            ]
        ].tail(15).to_string(index=False)
    )

    print("\nLoading BTC features...")
    btc = pd.read_csv(BTC_FILE)

    btc["timestamps"] = pd.to_datetime(
        btc["timestamps"],
        utc=True,
    )

    btc = btc.sort_values("timestamps").reset_index(drop=True)

    print(f"BTC candles: {len(btc)}")

    # We'll build one feature per economic event.
    #
    # merge_asof means:
    # for each BTC candle, use the most recent economic release
    # that had ALREADY happened.
    #
    # This is the key anti-look-ahead mechanism.

    event_frames = []

    for event_name in econ["event"].unique():

        event_df = econ[
            econ["event"] == event_name
        ].copy()

        event_df = event_df.sort_values("release_time")

        event_df = event_df[
            [
                "release_time",
                "value",
                "change_from_previous",
                "pct_change",
            ]
        ]

        event_df = event_df.rename(
            columns={
                "value": f"{event_name}_value",
                "change_from_previous": (
                    f"{event_name}_change"
                ),
                "pct_change": (
                    f"{event_name}_pct_change"
                ),
            }
        )

        merged = pd.merge_asof(
            btc[["timestamps"]],
            event_df,
            left_on="timestamps",
            right_on="release_time",
            direction="backward",
        )

        # How old was the latest release?
        merged[f"{event_name}_hours_since_release"] = (
            merged["timestamps"] - merged["release_time"]
        ).dt.total_seconds() / 3600

        merged = merged.drop(
            columns=["release_time"]
        )

        event_frames.append(merged)

    # Combine all economic features.
    result = btc.copy()

    for event_frame in event_frames:
        feature_columns = [
            column
            for column in event_frame.columns
            if column != "timestamps"
        ]

        result = result.merge(
            event_frame[
                ["timestamps"] + feature_columns
            ],
            on="timestamps",
            how="left",
        )

    # Event freshness flags.
    for event_name in econ["event"].unique():

        hours_column = (
            f"{event_name}_hours_since_release"
        )

        result[
            f"{event_name}_recent_24h"
        ] = (
            result[hours_column].notna()
            & (result[hours_column] <= 24)
        )

        result[
            f"{event_name}_recent_1h"
        ] = (
            result[hours_column].notna()
            & (result[hours_column] <= 1)
        )

    # Save.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Economic features created.")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")

    print("\nEconomic feature columns:")

    economic_columns = [
        column
        for column in result.columns
        if any(
            event in column
            for event in econ["event"].unique()
        )
    ]

    print("\n".join(economic_columns))

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()