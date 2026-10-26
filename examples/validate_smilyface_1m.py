
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("data/btc/smilyface_1m/raw/klines_1m/BTCUSDT")
TIME_CANDIDATES = [
    "open_time_ms", "open_time", "timestamp",
    "time", "datetime", "date", "open_timestamp"
]
PRICE_COLS = ["open", "high", "low", "close", "volume"]

files = sorted(ROOT.glob("*.parquet"))

if not files:
    raise SystemExit(f"No Parquet files found in: {ROOT.resolve()}")

print("=" * 72)
print("SMILYFACE BTCUSDT 1-MINUTE DATA VALIDATION")
print("=" * 72)
print(f"Files found: {len(files):,}")

all_times = []
total_rows = 0
null_ohlcv = 0
bad_ohlc = 0
file_errors = []
schemas = {}
sample_rows = []

def get_time_column(columns):
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in TIME_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    for c in columns:
        name = str(c).lower()
        if "time" in name or "timestamp" in name or name == "date":
            return c
    return None

def to_utc(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        valid = values.dropna()
        if valid.empty:
            return pd.to_datetime(series, errors="coerce", utc=True)

        magnitude = abs(float(valid.iloc[0]))
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"

        return pd.to_datetime(values, unit=unit, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)

for i, file in enumerate(files, start=1):
    try:
        df = pd.read_parquet(file)
        total_rows += len(df)

        schema = tuple(map(str, df.columns))
        schemas[schema] = schemas.get(schema, 0) + 1

        if i <= 3:
            sample_rows.append((file.name, list(df.columns), df.head(2)))

        time_col = get_time_column(df.columns)
        if time_col is None:
            file_errors.append(f"{file.name}: no timestamp column found")
            continue

        times = to_utc(df[time_col])
        valid = times.notna()
        if not valid.all():
            file_errors.append(
                f"{file.name}: {int((~valid).sum())} invalid timestamps"
            )

        # Store valid timestamps as integer nanoseconds for global checks.
        ns = times[valid].astype("int64").to_numpy()
        all_times.append(ns)

        lower_map = {str(c).lower(): c for c in df.columns}
        if all(c in lower_map for c in PRICE_COLS):
            cols = [lower_map[c] for c in PRICE_COLS]
            numeric = df[cols].apply(pd.to_numeric, errors="coerce")
            null_ohlcv += int(numeric.isna().any(axis=1).sum())

            o = numeric[lower_map["open"]]
            h = numeric[lower_map["high"]]
            l = numeric[lower_map["low"]]
            c = numeric[lower_map["close"]]
            v = numeric[lower_map["volume"]]

            invalid = (
                (h < l) | (h < o) | (h < c) |
                (l > o) | (l > c) |
                (numeric[lower_map["open"]] <= 0) |
                (numeric[lower_map["high"]] <= 0) |
                (numeric[lower_map["low"]] <= 0) |
                (numeric[lower_map["close"]] <= 0) |
                (v < 0)
            )
            bad_ohlc += int(invalid.fillna(False).sum())
        else:
            file_errors.append(
                f"{file.name}: missing one or more standard OHLCV columns"
            )

        if i % 200 == 0 or i == len(files):
            print(f"Processed {i:,}/{len(files):,} files...")

    except Exception as exc:
        file_errors.append(f"{file.name}: {type(exc).__name__}: {exc}")

print("\n" + "=" * 72)
print("SCHEMA")
print("=" * 72)
for schema, count in schemas.items():
    print(f"\nFiles with this schema: {count:,}")
    print(list(schema))

for name, columns, preview in sample_rows:
    print(f"\nSample file: {name}")
    print(preview.to_string(index=False))

if not all_times:
    raise SystemExit("\nNo usable timestamps found. Check the schema above.")

times_ns = np.concatenate(all_times)
times_ns.sort()

unique_ns = np.unique(times_ns)
duplicate_count = len(times_ns) - len(unique_ns)

first = pd.to_datetime(unique_ns[0], unit="ns", utc=True)
last = pd.to_datetime(unique_ns[-1], unit="ns", utc=True)

delta = np.diff(unique_ns)
minute_ns = 60 * 1_000_000_000
gap_deltas = delta[delta > minute_ns]
irregular_deltas = delta[delta % minute_ns != 0]

missing_minutes = int(
    np.maximum(delta // minute_ns - 1, 0).sum()
)
gap_count = int(len(gap_deltas))
irregular_count = int(len(irregular_deltas))

print("\n" + "=" * 72)
print("VALIDATION SUMMARY")
print("=" * 72)
print(f"Files successfully scanned: {len(schemas):,} schema groups")
print(f"Total rows read:             {total_rows:,}")
print(f"Valid unique timestamps:     {len(unique_ns):,}")
print(f"Duplicate timestamp rows:    {duplicate_count:,}")
print(f"First candle timestamp UTC:  {first}")
print(f"Last candle timestamp UTC:   {last}")
print(f"OHLCV rows with nulls:       {null_ohlcv:,}")
print(f"Rows with invalid OHLCV:     {bad_ohlc:,}")
print(f"Gaps longer than 1 minute:   {gap_count:,}")
print(f"Missing minutes in gaps:     {missing_minutes:,}")
print(f"Non-60-second intervals:     {irregular_count:,}")
print(f"Files with errors/warnings:  {len(file_errors):,}")

if gap_count:
    print("\nLargest gaps:")
    gap_indices = np.where(delta > minute_ns)[0]
    largest = sorted(
        gap_indices, key=lambda idx: delta[idx], reverse=True
    )[:10]
    for idx in largest:
        start = pd.to_datetime(unique_ns[idx], unit="ns", utc=True)
        end = pd.to_datetime(unique_ns[idx + 1], unit="ns", utc=True)
        minutes = delta[idx] / minute_ns
        print(f"  {start} -> {end} ({minutes:.0f} minutes apart)")

if file_errors:
    print("\nFirst 20 file warnings/errors:")
    for message in file_errors[:20]:
        print("  -", message)

# Save a compact report without changing source data.
report = {
    "files_found": len(files),
    "total_rows": total_rows,
    "unique_timestamps": len(unique_ns),
    "duplicate_timestamp_rows": duplicate_count,
    "first_timestamp_utc": str(first),
    "last_timestamp_utc": str(last),
    "ohlcv_rows_with_nulls": null_ohlcv,
    "invalid_ohlcv_rows": bad_ohlc,
    "gaps_over_one_minute": gap_count,
    "missing_minutes_estimated": missing_minutes,
    "non_60_second_intervals": irregular_count,
    "file_warnings_errors": len(file_errors),
}
pd.DataFrame([report]).to_csv(
    "data/btc/smilyface_1m_validation_summary.csv", index=False
)

print("\nSaved report: data/btc/smilyface_1m_validation_summary.csv")