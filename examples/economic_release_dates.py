import pandas as pd
from pathlib import Path


BLS_FILE = Path("data/bls_economic_data.csv")
OUTPUT_FILE = Path("data/bls_economic_releases.csv")


# Official BLS release dates for the observations in our dataset.
# Times are 08:30 Eastern Time.
#
# Format:
# (event, observation_year, observation_month): release_timestamp
#
# The dates below are based on BLS published release schedules.

RELEASE_DATES = {
    # 2025
    ("Employment", 2024, 12): "2025-01-10 08:30",
    ("CPI", 2024, 12): "2025-01-15 08:30",

    ("Employment", 2025, 1): "2025-02-07 08:30",
    ("CPI", 2025, 1): "2025-02-12 08:30",

    ("Employment", 2025, 2): "2025-03-07 08:30",
    ("CPI", 2025, 2): "2025-03-12 08:30",

    ("Employment", 2025, 3): "2025-04-04 08:30",
    ("CPI", 2025, 3): "2025-04-10 08:30",

    ("Employment", 2025, 4): "2025-05-02 08:30",
    ("CPI", 2025, 4): "2025-05-13 08:30",

    ("Employment", 2025, 5): "2025-06-06 08:30",
    ("CPI", 2025, 5): "2025-06-11 08:30",

    ("Employment", 2025, 6): "2025-07-03 08:30",
    ("CPI", 2025, 6): "2025-07-15 08:30",

    ("Employment", 2025, 7): "2025-08-01 08:30",
    ("CPI", 2025, 7): "2025-08-12 08:30",

    ("Employment", 2025, 8): "2025-09-05 08:30",
    ("CPI", 2025, 8): "2025-09-11 08:30",

    # BLS schedules were disrupted later in 2025.
    ("Employment", 2025, 9): "2025-11-20 08:30",
    ("CPI", 2025, 9): "2025-10-24 08:30",

    ("Employment", 2025, 10): None,
    ("CPI", 2025, 10): None,

    ("Employment", 2025, 11): "2025-12-16 08:30",
    ("CPI", 2025, 11): "2025-12-18 08:30",

    # 2026
    ("Employment", 2025, 12): "2026-01-09 08:30",
    ("CPI", 2025, 12): "2026-01-13 08:30",

    ("Employment", 2026, 1): "2026-02-11 08:30",
    ("CPI", 2026, 1): "2026-02-13 08:30",

    ("Employment", 2026, 2): "2026-03-06 08:30",
    ("CPI", 2026, 2): "2026-03-11 08:30",

    ("Employment", 2026, 3): "2026-04-03 08:30",
    ("CPI", 2026, 3): "2026-04-10 08:30",

    ("Employment", 2026, 4): "2026-05-08 08:30",
    ("CPI", 2026, 4): "2026-05-12 08:30",

    ("Employment", 2026, 5): "2026-06-05 08:30",
    ("CPI", 2026, 5): "2026-06-10 08:30",

    ("Employment", 2026, 6): "2026-07-02 08:30",
    ("CPI", 2026, 6): "2026-07-14 08:30",

    ("Employment", 2026, 7): "2026-08-07 08:30",
    ("CPI", 2026, 7): "2026-08-12 08:30",

    ("Employment", 2026, 8): "2026-09-04 08:30",
    ("CPI", 2026, 8): "2026-09-11 08:30",

    ("Employment", 2026, 9): "2026-10-02 08:30",
    ("CPI", 2026, 9): "2026-10-14 08:30",
}


def get_release_time(row):
    event = row["event"]

    if event == "Nonfarm Payrolls" or event == "Unemployment Rate":
        event_key = "Employment"
    elif event == "CPI":
        event_key = "CPI"
    else:
        return pd.NaT

    key = (
        event_key,
        row["observation_period"].year,
        row["observation_period"].month,
    )

    release = RELEASE_DATES.get(key)

    if release is None:
        return pd.NaT

    # BLS schedule times are Eastern Time.
    return pd.Timestamp(
        release,
        tz="America/New_York",
    ).tz_convert("UTC")


def main():
    print("Loading BLS observations...")

    df = pd.read_csv(BLS_FILE)

    df["observation_period"] = pd.to_datetime(
        df["observation_period"],
        utc=True,
    )

    print(f"Loaded {len(df)} observations.")

    df["release_time"] = df.apply(
        get_release_time,
        axis=1,
    )

    df["release_timezone"] = "America/New_York"

    df["importance"] = df["event"].map({
        "CPI": "high",
        "Nonfarm Payrolls": "high",
        "Unemployment Rate": "high",
    })

    df["release_status"] = df["release_time"].apply(
        lambda x: (
            "verified_schedule"
            if pd.notna(x)
            else "missing_schedule"
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Release timestamps populated.")
    print()

    print(
        df[
            [
                "event",
                "observation_period",
                "value",
                "release_time",
                "release_status",
            ]
        ].tail(20).to_string(index=False)
    )

    print()
    print("Release status:")
    print(
        df["release_status"]
        .value_counts()
        .to_string()
    )

    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()