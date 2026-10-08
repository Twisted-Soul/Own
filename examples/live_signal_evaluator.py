from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# KRONOS LIVE SIGNAL EVALUATOR
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CANDLE_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live.csv"
SIGNAL_FILE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live_signal.csv"
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_5m_live_signal_evaluation.csv"
)

HORIZONS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
}


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("KRONOS LIVE SIGNAL EVALUATOR")
print("=" * 70)

print(f"Candles: {CANDLE_FILE}")
print(f"Signals: {SIGNAL_FILE}")
print(f"Output:  {OUTPUT_FILE}")
print()

if not CANDLE_FILE.exists():
    raise FileNotFoundError(CANDLE_FILE)

if not SIGNAL_FILE.exists():
    raise FileNotFoundError(SIGNAL_FILE)

candles = pd.read_csv(CANDLE_FILE)
signals = pd.read_csv(SIGNAL_FILE)

candles["timestamps"] = pd.to_datetime(
    candles["timestamps"],
    utc=True,
)

signals["timestamp"] = pd.to_datetime(
    signals["timestamp"],
    utc=True,
)

candles = (
    candles
    .sort_values("timestamps")
    .drop_duplicates("timestamps")
    .reset_index(drop=True)
)

signals = (
    signals
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

print(f"Loaded candles: {len(candles):,}")
print(f"Loaded signals: {len(signals):,}")


# ============================================================
# EXISTING EVALUATIONS
# ============================================================

if OUTPUT_FILE.exists():

    existing = pd.read_csv(OUTPUT_FILE)

    if not existing.empty:

        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"],
            utc=True,
        )

        existing = (
            existing
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

    else:
        existing = pd.DataFrame()

else:

    existing = pd.DataFrame()


# ============================================================
# PRICE LOOKUP
# ============================================================

timestamp_to_index = {
    timestamp: i
    for i, timestamp in enumerate(candles["timestamps"])
}

close_prices = candles["close"].to_numpy(dtype=float)


# ============================================================
# BUILD / UPDATE EVALUATIONS
# ============================================================

rows = []

for _, signal in signals.iterrows():

    timestamp = signal["timestamp"]
    signal_type = str(signal["signal"]).upper()

    if timestamp not in timestamp_to_index:
        continue

    index = timestamp_to_index[timestamp]

    current_price = close_prices[index]

    row = {
        "timestamp": timestamp,
        "signal": signal_type,
        "price": current_price,
        "bullish_score": signal.get(
            "bullish_score",
            np.nan,
        ),
        "bearish_score": signal.get(
            "bearish_score",
            np.nan,
        ),
    }

    # --------------------------------------------------------
    # FUTURE OUTCOMES
    # --------------------------------------------------------

    for horizon_name, bars_forward in HORIZONS.items():

        future_index = index + bars_forward

        return_column = f"future_return_{horizon_name}"
        price_column = f"future_price_{horizon_name}"
        correct_column = f"correct_{horizon_name}"

        if future_index >= len(close_prices):

            row[price_column] = np.nan
            row[return_column] = np.nan
            row[correct_column] = np.nan

            continue

        future_price = close_prices[future_index]

        future_return = (
            future_price / current_price
        ) - 1.0

        row[price_column] = future_price
        row[return_column] = future_return

        if signal_type == "BUY":

            row[correct_column] = future_return > 0

        elif signal_type == "SELL":

            row[correct_column] = future_return < 0

        else:

            row[correct_column] = np.nan

    rows.append(row)


evaluation = pd.DataFrame(rows)


if evaluation.empty:

    print()
    print("No evaluatable signals found.")
    raise SystemExit(0)


# ============================================================
# SAVE
# ============================================================

evaluation = (
    evaluation
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

evaluation.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("-" * 70)
print("SIGNAL EVALUATION")
print("-" * 70)

print(
    f"Total signals: "
    f"{len(evaluation):,}"
)

for signal_type in ["BUY", "SELL", "HOLD"]:

    count = (
        evaluation["signal"] == signal_type
    ).sum()

    print(
        f"{signal_type:5s}: "
        f"{count:,}"
    )


# ============================================================
# COMPLETED OUTCOMES
# ============================================================

print()
print("-" * 70)
print("COMPLETED SIGNAL OUTCOMES")
print("-" * 70)

for horizon_name in HORIZONS:

    return_column = (
        f"future_return_{horizon_name}"
    )

    correct_column = (
        f"correct_{horizon_name}"
    )

    print()
    print(f"{horizon_name}")
    print("-" * 70)

    for signal_type in ["BUY", "SELL"]:

        subset = evaluation[
            evaluation["signal"] == signal_type
        ].copy()

        valid = subset[
            subset[return_column].notna()
            & subset[correct_column].notna()
        ]

        if valid.empty:

            print(
                f"{signal_type:5s}: "
                "waiting for future candles"
            )

            continue

        accuracy = (
            valid[correct_column].mean()
            * 100
        )

        average_return = (
            valid[return_column].mean()
            * 100
        )

        median_return = (
            valid[return_column].median()
            * 100
        )

        print(
            f"{signal_type:5s}: "
            f"n={len(valid):4d} | "
            f"correct={accuracy:6.2f}% | "
            f"avg={average_return:+.4f}% | "
            f"median={median_return:+.4f}%"
        )


# ============================================================
# HOLD OBSERVATION
# ============================================================

holds = evaluation[
    evaluation["signal"] == "HOLD"
]

print()
print("-" * 70)
print("HOLD OBSERVATIONS")
print("-" * 70)

print(
    f"HOLD signals collected: "
    f"{len(holds):,}"
)

for horizon_name in HORIZONS:

    column = (
        f"future_return_{horizon_name}"
    )

    valid = holds[
        holds[column].notna()
    ]

    if valid.empty:
        continue

    avg = valid[column].mean() * 100
    median = valid[column].median() * 100

    print(
        f"{horizon_name:4s}: "
        f"n={len(valid):4d} | "
        f"avg future move={avg:+.4f}% | "
        f"median={median:+.4f}%"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("-" * 70)
print(
    f"Saved: {OUTPUT_FILE}"
)
print("-" * 70)
print("=" * 70)