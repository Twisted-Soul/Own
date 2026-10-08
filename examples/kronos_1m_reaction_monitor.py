from pathlib import Path
import subprocess
import time
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

LIVE_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live.csv"
)

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live_features.csv"
)

ANALYZER = (
    BASE_DIR
    / "examples"
    / "analyze_1m_reaction_state.py"
)

POLL_SECONDS = 10


print("=" * 80)
print("KRONOS 1M REACTION MONITOR")
print("=" * 80)
print("Market timestamps: UTC internally / MYT for display")
print(f"Polling every {POLL_SECONDS} seconds")
print()
print("Waiting for new completed 1m candles...")
print()


last_timestamp = None
last_state = None


while True:

    try:

        if not LIVE_FILE.exists():

            print(
                "Waiting for live 1m data file..."
            )

            time.sleep(POLL_SECONDS)
            continue


        df = pd.read_csv(
            LIVE_FILE
        )

        if df.empty:

            time.sleep(POLL_SECONDS)
            continue


        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )


        current_timestamp = (
            df["timestamp"].iloc[-1]
        )


        # -------------------------------------------------------------
        # Only process a completed candle once.
        # -------------------------------------------------------------

        if (
            last_timestamp is not None
            and current_timestamp <= last_timestamp
        ):

            time.sleep(POLL_SECONDS)
            continue


        last_timestamp = current_timestamp


        # -------------------------------------------------------------
        # Rebuild features
        # -------------------------------------------------------------

        result = subprocess.run(
            [
                "python",
                str(
                    BASE_DIR
                    / "examples"
                    / "build_1m_live_features.py"
                ),
            ],
            capture_output=True,
            text=True,
        )


        if result.returncode != 0:

            print(
                "Feature build failed:"
            )

            print(
                result.stderr
            )

            time.sleep(POLL_SECONDS)
            continue


        # -------------------------------------------------------------
        # Run reaction analyzer
        # -------------------------------------------------------------

        result = subprocess.run(
            [
                "python",
                str(ANALYZER),
            ],
            capture_output=True,
            text=True,
        )


        if result.returncode != 0:

            print(
                "Reaction analyzer failed:"
            )

            print(
                result.stderr
            )

            time.sleep(POLL_SECONDS)
            continue


        output = result.stdout


        # -------------------------------------------------------------
        # Extract state
        # -------------------------------------------------------------

        state = None
        bias = None


        for line in output.splitlines():

            line = line.strip()

            if line.startswith("STATE:"):

                state = (
                    line.split(
                        "STATE:",
                        1
                    )[1]
                    .strip()
                )

            elif line.startswith("BIAS:"):

                bias = (
                    line.split(
                        "BIAS:",
                        1
                    )[1]
                    .strip()
                )


        # -------------------------------------------------------------
        # Display every new candle.
        # -------------------------------------------------------------

        timestamp_myt = (
            current_timestamp
            .tz_convert(
                "Asia/Kuala_Lumpur"
            )
        )


        print(
            f"[{timestamp_myt:%Y-%m-%d %H:%M:%S} MYT]"
            f" STATE={state}"
            f" BIAS={bias}"
        )


        # -------------------------------------------------------------
        # Highlight state transitions.
        # -------------------------------------------------------------

        if (
            last_state is not None
            and state != last_state
        ):

            print()
            print(
                ">>> STATE CHANGE"
            )

            print(
                f"    {last_state}"
                f" -> "
                f"{state}"
            )

            print(
                f"    Bias: {bias}"
            )

            print()


        last_state = state


        time.sleep(
            POLL_SECONDS
        )


    except KeyboardInterrupt:

        print()
        print(
            "KRONOS 1M REACTION MONITOR STOPPED."
        )

        break


    except Exception as exc:

        print()
        print(
            f"Monitor error: {exc}"
        )

        print(
            "Retrying..."
        )

        time.sleep(
            POLL_SECONDS
        )