from pathlib import Path
import subprocess
import sys
import time
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

LIVE_DATA = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_5m_live.csv"
)

SIGNAL_SCRIPT = (
    BASE_DIR
    / "examples"
    / "live_signal_engine.py"
)

PAPER_SCRIPT = (
    BASE_DIR
    / "examples"
    / "paper_trading_engine.py"
)

POLL_SECONDS = 10


# ============================================================
# DATA
# ============================================================

def get_latest_timestamp():

    if not LIVE_DATA.exists():
        return None

    try:

        df = pd.read_csv(
            LIVE_DATA,
            usecols=["timestamps"]
        )

        if df.empty:
            return None

        timestamps = pd.to_datetime(
            df["timestamps"],
            utc=True,
            errors="coerce"
        ).dropna()

        if timestamps.empty:
            return None

        return timestamps.max()

    except Exception as exc:

        print(
            f"[Monitor] Could not read live data: {exc}"
        )

        return None


# ============================================================
# RUN SCRIPT
# ============================================================

def run_script(script_path):

    print()
    print("-" * 70)
    print(
        f"Running {script_path.name}"
    )
    print("-" * 70)

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(script_path)
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:

            print(
                "Warnings/errors:"
            )

            print(
                result.stderr
            )

        if result.returncode != 0:

            print(
                f"{script_path.name} "
                f"exited with code "
                f"{result.returncode}"
            )

            return False

        return True

    except Exception as exc:

        print(
            f"Failed to run "
            f"{script_path.name}: {exc}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KRONOS CONTINUOUS PAPER TRADING MONITOR")
    print("=" * 70)

    print(
        "PAPER MODE ONLY — NO REAL ORDERS"
    )

    print()

    print(
        f"Live data:       {LIVE_DATA}"
    )

    print(
        f"Signal engine:   {SIGNAL_SCRIPT}"
    )

    print(
        f"Paper engine:    {PAPER_SCRIPT}"
    )

    print(
        f"Poll interval:   {POLL_SECONDS} seconds"
    )

    print()

    print(
        "Waiting for new completed 5-minute candles..."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Remember the current candle.
    #
    # We do NOT process old candles as new events when the
    # monitor starts.
    # --------------------------------------------------------

    last_seen = get_latest_timestamp()

    if last_seen is not None:

        print(
            f"Current latest candle: "
            f"{last_seen}"
        )

    else:

        print(
            "No existing candle detected."
        )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        try:

            current = get_latest_timestamp()

            if current is None:

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # New completed candle.
            # ------------------------------------------------

            if current > last_seen:

                print()
                print()
                print("=" * 70)
                print(
                    "NEW COMPLETED CANDLE"
                )
                print("=" * 70)

                print(
                    f"Previous: "
                    f"{last_seen}"
                )

                print(
                    f"Current:  "
                    f"{current}"
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # First calculate the signal.
                # Then process the paper account using that signal.
                # ------------------------------------------------

                signal_ok = run_script(
                    SIGNAL_SCRIPT
                )

                if signal_ok:

                    run_script(
                        PAPER_SCRIPT
                    )

                else:

                    print(
                        "Signal engine failed."
                    )

                    print(
                        "Paper engine was NOT run."
                    )

                last_seen = current

                print()
                print(
                    "Waiting for next completed candle..."
                )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print("=" * 70)
            print(
                "KRONOS PAPER MONITOR STOPPED"
            )
            print("=" * 70)

            break

        except Exception as exc:

            print()
            print(
                f"[Monitor] Unexpected error: "
                f"{exc}"
            )

            time.sleep(
                POLL_SECONDS
            )


if __name__ == "__main__":
    main()