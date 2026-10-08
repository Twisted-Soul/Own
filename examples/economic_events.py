import pandas as pd
from pathlib import Path


OUTPUT_FILE = Path("data/economic_events.csv")


EVENTS = [
    {
        "event": "CPI",
        "category": "inflation",
        "importance": "high",
        "source": "BLS",
    },
    {
        "event": "Core CPI",
        "category": "inflation",
        "importance": "high",
        "source": "BLS",
    },
    {
        "event": "Employment Situation",
        "category": "employment",
        "importance": "high",
        "source": "BLS",
    },
    {
        "event": "PPI",
        "category": "inflation",
        "importance": "medium",
        "source": "BLS",
    },
    {
        "event": "FOMC",
        "category": "monetary_policy",
        "importance": "high",
        "source": "Federal Reserve",
    },
    {
        "event": "PCE",
        "category": "inflation",
        "importance": "high",
        "source": "BEA/FRED",
    },
    {
        "event": "GDP",
        "category": "growth",
        "importance": "high",
        "source": "BEA/FRED",
    },
    {
        "event": "Retail Sales",
        "category": "consumer",
        "importance": "medium",
        "source": "Census/FRED",
    },
]


def main():
    df = pd.DataFrame(EVENTS)

    # Columns that will eventually contain the actual release information.
    df["release_time"] = pd.NaT
    df["reference_period"] = ""
    df["actual"] = pd.NA
    df["forecast"] = pd.NA
    df["previous"] = pd.NA
    df["surprise"] = pd.NA

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Economic event framework created.")
    print()
    print(df.to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()