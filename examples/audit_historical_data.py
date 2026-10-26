import pandas as pd
from pathlib import Path

FILES = [
    Path("data/btc/BTCUSDT_5m_signals.csv"),
    Path("data/btc/BTCUSDT_5m_features.csv"),
    Path("data/btc/BTCUSDT_5m_1y_features.csv"),
    Path("data/btc/BTCUSDT_5m_full_features.csv"),
]

for path in FILES:
    print("\n" + "=" * 75)
    print(path)

    if not path.exists():
        print("NOT FOUND")
        continue

    df = pd.read_csv(path, nrows=5)
    print("Columns:", list(df.columns))

    # Read the full file to establish row count and timestamp range.
    full = pd.read_csv(path)
    print("Rows:", len(full))

    time_col = next(
        (
            c for c in ["timestamps", "timestamp", "open_time",
                        "open_time_ms", "datetime", "date"]
            if c in full.columns
        ),
        None,
    )

    if time_col:
        ts = full[time_col]

        if time_col == "open_time_ms":
            parsed = pd.to_datetime(
                ts, unit="ms", utc=True, errors="coerce"
            )
        else:
            parsed = pd.to_datetime(ts, utc=True, errors="coerce")

        print("Timestamp column:", time_col)
        print("First valid timestamp:", parsed.min())
        print("Last valid timestamp:", parsed.max())
        print("Invalid timestamps:", int(parsed.isna().sum()))
    else:
        print("No recognized timestamp column found.")

    print("Sample:")
    print(full.head(2).to_string(index=False))
