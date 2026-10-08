from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

BTC_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_1y_features.csv"
RELEASE_FILE = BASE_DIR / "data" / "bls_economic_releases.csv"

OUTPUT_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_event_reactions.csv"


HORIZONS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
}


def classify_event(row):
    """
    Convert the economic-release description into a broad event category.
    We intentionally keep this simple for the first iteration.
    """

    text = " ".join(
        str(row.get(col, ""))
        for col in row.index
        if "event" in str(col).lower()
        or "release" in str(col).lower()
        or "indicator" in str(col).lower()
        or "name" in str(col).lower()
    ).lower()

    if "fomc" in text or "fed" in text or "interest" in text:
        return "FOMC"

    if "cpi" in text or "consumer price" in text:
        return "CPI"

    if "ppi" in text or "producer price" in text:
        return "PPI"

    if "employment" in text or "nonfarm" in text or "nfp" in text:
        return "NFP"

    if "pce" in text or "personal consumption" in text:
        return "PCE"

    if "gdp" in text:
        return "GDP"

    if "retail" in text:
        return "RETAIL_SALES"

    return "OTHER"


def find_release_timestamp(df):
    """
    Locate the most likely release timestamp column.
    """

    candidates = [
        "release_timestamp",
        "release_time",
        "timestamp",
        "datetime",
        "date",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a release timestamp column. "
        f"Available columns: {list(df.columns)}"
    )


def main():
    print("=" * 80)
    print("KRONOS EVENT / RELEASE REACTION ANALYSIS")
    print("=" * 80)

    if not BTC_FILE.exists():
        raise FileNotFoundError(f"BTC feature file not found: {BTC_FILE}")

    if not RELEASE_FILE.exists():
        raise FileNotFoundError(f"Release file not found: {RELEASE_FILE}")

    btc = pd.read_csv(BTC_FILE)
    releases = pd.read_csv(RELEASE_FILE)

    print(f"Loaded BTC candles: {len(btc):,}")
    print(f"Loaded releases:    {len(releases):,}")

    # ------------------------------------------------------------------
    # BTC preparation
    # ------------------------------------------------------------------

    btc["timestamps"] = pd.to_datetime(
        btc["timestamps"],
        utc=True,
        errors="coerce",
    )

    btc = btc.dropna(subset=["timestamps"]).copy()
    btc = btc.sort_values("timestamps").reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_cols:
        if col in btc.columns:
            btc[col] = pd.to_numeric(btc[col], errors="coerce")

    btc = btc.dropna(subset=["close"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Release preparation
    # ------------------------------------------------------------------

    release_time_col = find_release_timestamp(releases)

    releases["release_timestamp"] = pd.to_datetime(
        releases[release_time_col],
        utc=True,
        errors="coerce",
    )

    releases = releases.dropna(
        subset=["release_timestamp"]
    ).copy()

    releases = releases.sort_values(
        "release_timestamp"
    ).reset_index(drop=True)

    releases["event_type"] = releases.apply(
        classify_event,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Match every release to the first BTC candle at or after release.
    #
    # This is important:
    # We do NOT use the candle immediately before the release as the
    # reaction candle. The release itself happens during/around a clock
    # time, so the first completed 5m candle after it is our anchor.
    # ------------------------------------------------------------------

    results = []

    btc_times = btc["timestamps"].values

    for _, event in releases.iterrows():

        release_time = event["release_timestamp"]

        idx = btc["timestamps"].searchsorted(
            release_time,
            side="left",
        )

        if idx >= len(btc):
            continue

        anchor = btc.iloc[idx]

        # Need enough future candles for the largest horizon.
        if idx + max(HORIZONS.values()) >= len(btc):
            continue

        anchor_close = float(anchor["close"])

        row = {
            "release_timestamp": release_time,
            "event_type": event["event_type"],
            "btc_anchor_timestamp": anchor["timestamps"],
            "btc_anchor_close": anchor_close,
        }

        # Preserve useful release metadata where available.
        for col in releases.columns:
            if col in {
                release_time_col,
                "release_timestamp",
                "event_type",
            }:
                continue

            value = event[col]

            if pd.notna(value):
                row[f"event_{col}"] = value

        # --------------------------------------------------------------
        # Price reaction
        # --------------------------------------------------------------

        for name, bars in HORIZONS.items():

            future = btc.iloc[idx + bars]

            future_close = float(future["close"])

            ret = future_close / anchor_close - 1.0

            row[f"future_return_{name}"] = ret

            # Absolute movement tells us how much BTC actually moved,
            # regardless of direction.
            row[f"abs_return_{name}"] = abs(ret)

        # --------------------------------------------------------------
        # Immediate volatility / candle reaction
        # --------------------------------------------------------------

        first_5m = btc.iloc[idx]

        row["first_candle_range_pct"] = (
            float(first_5m["high"])
            / float(first_5m["low"])
            - 1.0
        ) if float(first_5m["low"]) > 0 else np.nan

        row["first_candle_body_pct"] = (
            float(first_5m["close"])
            / float(first_5m["open"])
            - 1.0
        ) if float(first_5m["open"]) > 0 else np.nan

        # Compare reaction candle volume with the feature's normal volume
        # when available.
        if "relative_volume" in btc.columns:
            row["reaction_relative_volume"] = float(
                first_5m["relative_volume"]
            )

        if "volatility_20" in btc.columns:
            row["reaction_volatility_20"] = float(
                first_5m["volatility_20"]
            )

        results.append(row)

    output = pd.DataFrame(results)

    if output.empty:
        print("\nNo events could be matched to BTC candles.")
        return

    # ------------------------------------------------------------------
    # Save event-level dataset
    # ------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nMatched events: {len(output):,}")
    print(f"Saved: {OUTPUT_FILE}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVENT REACTION SUMMARY")
    print("=" * 80)

    for event_type, group in output.groupby("event_type"):

        print(f"\n{event_type}")
        print("-" * 60)
        print(f"Events: {len(group)}")

        for horizon in HORIZONS:

            col = f"future_return_{horizon}"

            values = group[col].dropna()

            if values.empty:
                continue

            print(
                f"{horizon:>3}: "
                f"avg={values.mean() * 100:+.4f}% | "
                f"median={values.median() * 100:+.4f}% | "
                f"UP={(values > 0).mean() * 100:.2f}% | "
                f"abs avg={values.abs().mean() * 100:.4f}%"
            )

    # ------------------------------------------------------------------
    # Overall event-vs-normal comparison
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL RELEASES")
    print("=" * 80)

    for horizon in HORIZONS:

        col = f"future_return_{horizon}"

        values = output[col].dropna()

        if values.empty:
            continue

        print(
            f"{horizon:>3}: "
            f"avg={values.mean() * 100:+.4f}% | "
            f"median={values.median() * 100:+.4f}% | "
            f"UP={(values > 0).mean() * 100:.2f}% | "
            f"abs avg={values.abs().mean() * 100:.4f}%"
        )

    print("\n" + "=" * 80)
    print("LARGEST EVENT REACTIONS")
    print("=" * 80)

    ranking_col = "abs_return_1h"

    largest = (
        output
        .sort_values(ranking_col, ascending=False)
        .head(15)
    )

    for _, row in largest.iterrows():

        direction = (
            "UP"
            if row["future_return_1h"] > 0
            else "DOWN"
        )

        print(
            f"{row['release_timestamp']} | "
            f"{row['event_type']:12s} | "
            f"1h={row['future_return_1h'] * 100:+.3f}% | "
            f"reaction={direction}"
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()