from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

# ============================================================
# KRONOS LIVE EVALUATION MONITOR
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CANDLE_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live.csv"
EVALUATOR = BASE_DIR / "examples" / "live_signal_evaluator.py"

POLL_SECONDS = 30

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("KRONOS LIVE SIGNAL EVALUATION MONITOR")
print("=" * 70)

print(f"Live data:  {CANDLE_FILE}")
print(f"Evaluator:  {EVALUATOR}")
print(f"Poll:       {POLL_SECONDS} seconds")
print()
print("This monitor evaluates completed Kronos signals")
print("as enough future candles become available.")
print("No trading. No orders.")
print()
print("Press Ctrl+C to stop.")
print("=" * 70)

# ============================================================
# HELPERS
# ============================================================

def get_latest_timestamp():

    if not CANDLE_FILE.exists():
        return None

    try:
        df = pd.read_csv(
            CANDLE_FILE,
            usecols=["timestamps"],
        )

        if df.empty:
            return None

        timestamps = pd.to_datetime(
            df["timestamps"],
            utc=True,
        )

        return timestamps.max()

    except Exception as exc:

        print(
            f"[WARNING] Could not read candle timestamp: {exc}"
        )

        return None


def run_evaluator():

    print()
    print("-" * 70)
    print("Running live_signal_evaluator.py")
    print("-" * 70)

    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
        ],
        cwd=str(BASE_DIR),
    )

    if result.returncode != 0:

        print()
        print(
            f"[WARNING] Evaluator exited with code "
            f"{result.returncode}"
        )

    print("-" * 70)


# ============================================================
# INITIAL STATE
# ============================================================

last_seen = get_latest_timestamp()

if last_seen is None:

    print("[WARNING] No live candle data found.")

else:

    print(
        f"Current latest candle: "
        f"{last_seen}"
    )

print()
print("Waiting for new completed candles...")
print("=" * 70)

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        time.sleep(POLL_SECONDS)

        latest = get_latest_timestamp()

        if latest is None:
            continue

        if last_seen is None:

            last_seen = latest
            continue

        if latest > last_seen:

            print()
            print("=" * 70)
            print("NEW COMPLETED CANDLE")
            print("=" * 70)

            print(f"Previous: {last_seen}")
            print(f"Current:  {latest}")

            last_seen = latest

            run_evaluator()

        # No new candle yet.
        # Continue waiting.

except KeyboardInterrupt:

    print()
    print("=" * 70)
    print("KRONOS EVALUATION MONITOR STOPPED")
    print("=" * 70)