import requests
import pandas as pd
from pathlib import Path


OUTPUT_FILE = Path("data/bls_economic_data.csv")

# BLS public series IDs
SERIES = {
    "CPI": "CUUR0000SA0",
    "Unemployment Rate": "LNS14000000",
    "Nonfarm Payrolls": "CES0000000001",
}


def fetch_bls_data(series_ids, start_year, end_year):
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    payload = {
        "seriesid": list(series_ids.values()),
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    print("Requesting BLS data...")

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("status") == "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS API request failed: {data}"
        )

    return data


def parse_bls_data(data, series_ids):
    reverse_map = {
        series_id: name
        for name, series_id in series_ids.items()
    }

    rows = []

    for series in data["Results"]["series"]:
        series_id = series["seriesID"]
        event_name = reverse_map.get(series_id, series_id)

        for item in series["data"]:
            period = item["period"]

            # Skip annual averages.
            if not period.startswith("M"):
                continue

            month = int(period[1:])

            timestamp = pd.Timestamp(
                year=int(item["year"]),
                month=month,
                day=1,
                tz="UTC",
            )

            rows.append({
                "event": event_name,
                "series_id": series_id,
                "observation_period": timestamp,
                "value": (
                    float(item["value"])
                    if item["value"] != "-"
                    else None
                ),
                "period": period,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No BLS observations were returned.")

    return df.sort_values(
        ["observation_period", "event"]
    ).reset_index(drop=True)


def main():
    print("Kronos BLS economic data importer")
    print()

    # Start with two years so we have enough history
    # for later analysis.
    start_year = 2025
    end_year = 2026

    data = fetch_bls_data(
        SERIES,
        start_year,
        end_year,
    )

    df = parse_bls_data(
        data,
        SERIES,
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
    print(f"Downloaded {len(df)} observations.")
    print()

    print("Observations by event:")
    print(df["event"].value_counts().to_string())

    print()
    print("Latest observations:")
    print(
        df.tail(15).to_string(index=False)
    )

    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()