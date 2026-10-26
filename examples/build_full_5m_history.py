from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path("data/btc/5m_raw")

MISSING_FILE = (
    RAW_DIR
    / "BTCUSDT_5m_missing_2025-02_to_2025-09.csv"
)

EXISTING_FEATURE_FILE = (
    Path("data/btc")
    / "BTCUSDT_5m_1y_features.csv"
)

OUTPUT_FILE = (
    Path("data/btc")
    / "BTCUSDT_5m_full_raw.csv"
)


# ============================================================
# LOAD MISSING HISTORICAL DATA
# ============================================================

print("=" * 70)
print("BUILD FULL BTCUSDT 5m RAW HISTORY")
print("=" * 70)

print()
print("Loading missing historical data...")

old_data = pd.read_csv(
    MISSING_FILE
)

old_data["timestamps"] = pd.to_datetime(
    old_data["timestamp"],
    utc=True,
    errors="coerce",
)

old_data = old_data[
    [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
].copy()

print(
    f"Historical rows: {len(old_data):,}"
)

print(
    f"Historical range: "
    f"{old_data['timestamps'].min()} → "
    f"{old_data['timestamps'].max()}"
)


# ============================================================
# LOAD EXISTING DATA
# ============================================================

print()
print("Loading existing 5m dataset...")

existing = pd.read_csv(
    EXISTING_FEATURE_FILE
)

existing["timestamps"] = pd.to_datetime(
    existing["timestamps"],
    utc=True,
    errors="coerce",
)

existing = existing[
    [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
].copy()

print(
    f"Existing rows: {len(existing):,}"
)

print(
    f"Existing range: "
    f"{existing['timestamps'].min()} → "
    f"{existing['timestamps'].max()}"
)


# ============================================================
# COMBINE
# ============================================================

print()
print("Combining datasets...")

combined = pd.concat(
    [
        old_data,
        existing,
    ],
    ignore_index=True,
)


# ============================================================
# CLEAN
# ============================================================

combined = (
    combined
    .dropna(subset=["timestamps"])
    .sort_values("timestamps")
    .drop_duplicates(
        subset=["timestamps"],
        keep="first",
    )
    .reset_index(drop=True)
)


# ============================================================
# NUMERIC VALIDATION
# ============================================================

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

for column in numeric_columns:

    combined[column] = pd.to_numeric(
        combined[column],
        errors="coerce",
    )


combined = combined.dropna(
    subset=numeric_columns
).reset_index(drop=True)


# ============================================================
# CHECK 5-MINUTE CONTINUITY
# ============================================================

time_diff = (
    combined["timestamps"]
    .diff()
    .dropna()
)

expected_delta = pd.Timedelta(minutes=5)

gaps = time_diff[
    time_diff != expected_delta
]

print()
print("=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print(
    f"Combined rows: {len(combined):,}"
)

print(
    f"Start: {combined['timestamps'].iloc[0]}"
)

print(
    f"End:   {combined['timestamps'].iloc[-1]}"
)

print(
    f"Duplicate timestamps: "
    f"{combined['timestamps'].duplicated().sum()}"
)

print(
    f"Missing values: "
    f"{combined.isna().sum().sum()}"
)

print(
    f"5m continuity gaps: {len(gaps):,}"
)


if len(gaps) > 0:

    print()
    print("Largest gaps:")

    gap_table = (
        time_diff
        .loc[gaps.index]
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print(gap_table)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(combined):,}"
)

print(
    f"Range: "
    f"{combined['timestamps'].iloc[0]} → "
    f"{combined['timestamps'].iloc[-1]}"
)