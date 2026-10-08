import sys
from pathlib import Path

# Make the Kronos root available
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from model import Kronos, KronosTokenizer, KronosPredictor


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "./data/btc/BTCUSDT_5m_clean.csv"

LOOKBACK = 400
PRED_LEN = 120

# Number of historical prediction tests
NUM_TESTS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("Loading BTCUSDT data...")

df = pd.read_csv(DATA_PATH)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Total candles: {len(df)}")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")


# ============================================================
# LOAD KRONOS
# ============================================================

print("\nLoading Kronos tokenizer...")

tokenizer = KronosTokenizer.from_pretrained(
    "NeoQuasar/Kronos-Tokenizer-base"
)

print("Loading Kronos model...")

model = Kronos.from_pretrained(
    "NeoQuasar/Kronos-small"
)

predictor = KronosPredictor(
    model,
    tokenizer,
    max_context=512
)

print("Kronos loaded successfully.")


# ============================================================
# SELECT TEST POINTS
# ============================================================

# We need:
#
# LOOKBACK candles before prediction
# +
# PRED_LEN candles after prediction
#
# Leave enough room at the end.

first_origin = LOOKBACK
last_origin = len(df) - PRED_LEN

origins = np.linspace(
    first_origin,
    last_origin,
    NUM_TESTS,
    dtype=int
)

print("\nTest points:")
for i, origin in enumerate(origins, 1):
    print(
        f"Test {i}: "
        f"{df['timestamps'].iloc[origin]} "
        f"(index {origin})"
    )


# ============================================================
# RUN TESTS
# ============================================================

results = []

for test_number, origin in enumerate(origins, 1):

    print("\n" + "=" * 60)
    print(f"TEST {test_number}/{NUM_TESTS}")
    print("=" * 60)

    # Historical context
    x_df = df.iloc[
        origin - LOOKBACK:origin
    ].copy()

    # Actual future
    y_df = df.iloc[
        origin:origin + PRED_LEN
    ].copy()

    x_timestamp = x_df["timestamps"]
    y_timestamp = y_df["timestamps"]

    print(
        f"Prediction start: {y_timestamp.iloc[0]}"
    )

    print(
        f"Prediction end:   {y_timestamp.iloc[-1]}"
    )

    # --------------------------------------------------------
    # Run Kronos
    # --------------------------------------------------------

    pred_df = predictor.predict(
        df=x_df[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount"
            ]
        ],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    # --------------------------------------------------------
    # Compare CLOSE
    # --------------------------------------------------------

    actual_close = y_df["close"].values
    predicted_close = pred_df["close"].values

    mae = np.mean(
        np.abs(actual_close - predicted_close)
    )

    rmse = np.sqrt(
        np.mean(
            (actual_close - predicted_close) ** 2
        )
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    actual_start = actual_close[0]
    actual_end = actual_close[-1]

    predicted_start = predicted_close[0]
    predicted_end = predicted_close[-1]

    actual_return = (
        (actual_end - actual_start)
        / actual_start
    ) * 100

    predicted_return = (
        (predicted_end - predicted_start)
        / predicted_start
    ) * 100

    actual_direction = (
        "UP" if actual_return > 0 else "DOWN"
    )

    predicted_direction = (
        "UP" if predicted_return > 0 else "DOWN"
    )

    direction_correct = (
        actual_direction == predicted_direction
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "test": test_number,
        "start_time": y_timestamp.iloc[0],
        "end_time": y_timestamp.iloc[-1],
        "MAE": mae,
        "RMSE": rmse,
        "actual_return_%": actual_return,
        "predicted_return_%": predicted_return,
        "actual_direction": actual_direction,
        "predicted_direction": predicted_direction,
        "direction_correct": direction_correct
    })

    print("\nRESULT")
    print(f"MAE:                {mae:.2f} USDT")
    print(f"RMSE:               {rmse:.2f} USDT")
    print(f"Actual return:      {actual_return:.3f}%")
    print(f"Predicted return:   {predicted_return:.3f}%")
    print(f"Actual direction:   {actual_direction}")
    print(f"Predicted direction:{predicted_direction}")
    print(f"Direction correct:  {direction_correct}")


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("KRONOS BTCUSDT EVALUATION SUMMARY")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# OVERALL METRICS
# ============================================================

average_mae = results_df["MAE"].mean()
average_rmse = results_df["RMSE"].mean()

direction_accuracy = (
    results_df["direction_correct"].mean()
    * 100
)

print("\n")
print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"Average MAE:             {average_mae:.2f} USDT")
print(f"Average RMSE:            {average_rmse:.2f} USDT")
print(f"Direction accuracy:      {direction_accuracy:.2f}%")

print("\nEvaluation complete.")