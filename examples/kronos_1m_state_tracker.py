from pathlib import Path
import time
import subprocess
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

LIVE_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live.csv"
)

ANALYZER = (
    BASE_DIR
    / "examples"
    / "analyze_1m_reaction_state.py"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_state_history.csv"
)

POLL_SECONDS = 10

# Number of consecutive completed candles required
# before considering a state persistent.
PERSISTENCE_REQUIRED = 3


print("=" * 80)
print("KRONOS 1M PERSISTENT STATE TRACKER")
print("=" * 80)

print(
    f"Persistence requirement: "
    f"{PERSISTENCE_REQUIRED} candles"
)

print(
    "Timestamps: UTC internally / MYT display"
)

print()


# ---------------------------------------------------------------------
# Existing history
# ---------------------------------------------------------------------

if OUTPUT_FILE.exists():

    history = pd.read_csv(
        OUTPUT_FILE
    )

else:

    history = pd.DataFrame()


# ---------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------

last_timestamp = None

raw_state = None
raw_state_count = 0

persistent_state = None


# ---------------------------------------------------------------------
# Helper: save history
# ---------------------------------------------------------------------

def save_history():

    if history.empty:
        return

    history.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ---------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------

while True:

    try:

        if not LIVE_FILE.exists():

            time.sleep(
                POLL_SECONDS
            )

            continue


        df = pd.read_csv(
            LIVE_FILE
        )


        if df.empty:

            time.sleep(
                POLL_SECONDS
            )

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
        # Only process each completed candle once.
        # -------------------------------------------------------------

        if (
            last_timestamp is not None
            and current_timestamp <= last_timestamp
        ):

            time.sleep(
                POLL_SECONDS
            )

            continue


        last_timestamp = current_timestamp


        # -------------------------------------------------------------
        # Rebuild features.
        # -------------------------------------------------------------

        feature_result = subprocess.run(
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


        if feature_result.returncode != 0:

            print(
                "Feature build failed:"
            )

            print(
                feature_result.stderr
            )

            time.sleep(
                POLL_SECONDS
            )

            continue


        # -------------------------------------------------------------
        # Run raw reaction analyzer.
        # -------------------------------------------------------------

        analyzer_result = subprocess.run(
            [
                "python",
                str(ANALYZER),
            ],
            capture_output=True,
            text=True,
        )


        if analyzer_result.returncode != 0:

            print(
                "Analyzer failed:"
            )

            print(
                analyzer_result.stderr
            )

            time.sleep(
                POLL_SECONDS
            )

            continue


        output = analyzer_result.stdout


        state = "UNKNOWN"
        bias = "WAIT"


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
        # Raw-state persistence.
        # -------------------------------------------------------------

        if state == raw_state:

            raw_state_count += 1

        else:

            raw_state = state
            raw_state_count = 1


        # -------------------------------------------------------------
        # Promote to persistent state only after N candles.
        # -------------------------------------------------------------

        state_changed = False


        if raw_state_count >= PERSISTENCE_REQUIRED:

            if persistent_state != raw_state:

                previous_persistent = (
                    persistent_state
                )

                persistent_state = raw_state

                state_changed = True

            else:

                previous_persistent = (
                    persistent_state
                )

        else:

            previous_persistent = (
                persistent_state
            )


        # -------------------------------------------------------------
        # MYT timestamp.
        # -------------------------------------------------------------

        timestamp_myt = (
            current_timestamp
            .tz_convert(
                "Asia/Kuala_Lumpur"
            )
        )


        # -------------------------------------------------------------
        # Record.
        # -------------------------------------------------------------

        record = {
            "timestamp": current_timestamp,
            "timestamp_myt": timestamp_myt,
            "raw_state": state,
            "raw_bias": bias,
            "raw_state_count": raw_state_count,
            "persistent_state": persistent_state,
            "persistent_state_changed": state_changed,
        }


        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [record]
                ),
            ],
            ignore_index=True,
        )


        # Keep the file manageable.
        if len(history) > 10000:

            history = history.tail(
                10000
            ).reset_index(
                drop=True
            )


        save_history()


        # -------------------------------------------------------------
        # Console.
        # -------------------------------------------------------------

        print(
            f"[{timestamp_myt:%Y-%m-%d %H:%M:%S} MYT] "
            f"RAW={state:<22} "
            f"({raw_state_count}) "
            f"PERSISTENT={persistent_state}"
        )


        # -------------------------------------------------------------
        # Persistent transition.
        # -------------------------------------------------------------

        if state_changed:

            print()
            print(
                ">>> PERSISTENT STATE CHANGE"
            )

            print(
                f"    "
                f"{previous_persistent}"
                f" -> "
                f"{persistent_state}"
            )

            print(
                f"    Raw state persisted "
                f"for {raw_state_count} candles."
            )

            print()


        time.sleep(
            POLL_SECONDS
        )


    except KeyboardInterrupt:

        print()
        print(
            "KRONOS 1M STATE TRACKER STOPPED."
        )

        break


    except Exception as exc:

        print()
        print(
            f"Tracker error: {exc}"
        )

        time.sleep(
            POLL_SECONDS
        )