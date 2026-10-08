from pathlib import Path
import time
import subprocess
import sys
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

LIVE_DATA = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live.csv"
SIGNAL_SCRIPT = BASE_DIR / "examples" / "live_signal_engine.py"

POLL_SECONDS = 10


# ============================================================
# HELPERS
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


def run_signal_engine():

    print()
    print("=" * 70)
    print("NEW COMPLETED CANDLE DETECTED")
    print("=" * 70)

    print(
        f"Running: {SIGNAL_SCRIPT.name}"
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(SIGNAL_SCRIPT)
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(
                "Signal engine warnings/errors:"
            )
            print(result.stderr)

        if result.returncode != 0:

            print(
                f"Signal engine exited with code "
                f"{result.returncode}"
            )

        else:

            print(
                "Signal engine completed successfully."
            )

    except Exception as exc:

        print(
            f"[Monitor] Failed to run signal engine: {exc}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KRONOS LIVE MONITOR")
    print("=" * 70)

    print(f"Live data:     {LIVE_DATA}")
    print(f"Signal engine: {SIGNAL_SCRIPT}")
    print(f"Poll interval: {POLL_SECONDS} seconds")
    print()
    print(
        "Waiting for new completed 5-minute candles..."
    )
    print(
        "Press Ctrl+C to stop."
    )
    print("=" * 70)

    last_seen = get_latest_timestamp()

    if last_seen is not None:

        print(
            f"Current latest candle: {last_seen}"
        )

    else:

        print(
            "No existing completed candle detected."
        )

    while True:

        try:

            current = get_latest_timestamp()

            if current is None:

                time.sleep(POLL_SECONDS)
                continue

            if last_seen is None:

                last_seen = current

                print(
                    f"[Monitor] Initial candle detected: "
                    f"{current}"
                )

            elif current > last_seen:

                print()
                print(
                    f"[Monitor] Candle advanced: "
                    f"{last_seen} → {current}"
                )

                last_seen = current

                run_signal_engine()

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:

            print()
            print("=" * 70)
            print("KRONOS LIVE MONITOR STOPPED")
            print("=" * 70)

            break

        except Exception as exc:

            print(
                f"[Monitor] Unexpected error: {exc}"
            )

            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()