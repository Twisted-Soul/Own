from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_validation_predictions.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_validation_analysis.csv"
)

ROUND_TRIP_COST = 0.0014


# ============================================================
# LOAD
# ============================================================

print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
)

print(f"Loaded {len(df):,} validation predictions.")


# ============================================================
# BASIC CHECKS
# ============================================================

required_columns = {
    "timestamps",
    "future_return_1h",
    "target",
    "prediction",
    "down_probability",
    "neutral_probability",
    "up_probability",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# MODEL CONFIDENCE
# ============================================================

probability_columns = [
    "down_probability",
    "neutral_probability",
    "up_probability",
]

df["confidence"] = df[probability_columns].max(axis=1)

df["predicted_probability"] = df["confidence"]


def probability_to_direction(row):
    probabilities = {
        -1: row["down_probability"],
        0: row["neutral_probability"],
        1: row["up_probability"],
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


df["prediction"] = df.apply(
    probability_to_direction,
    axis=1,
)


# ============================================================
# CORRECTNESS
# ============================================================

df["correct"] = (
    df["prediction"] == df["target"]
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"Rows: {len(df):,}")

accuracy = df["correct"].mean()

print(f"Overall accuracy: {accuracy:.4%}")

print(
    f"Average future 1h return: "
    f"{df['future_return_1h'].mean():.4%}"
)


# ============================================================
# PREDICTED CLASS ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PREDICTED CLASS ANALYSIS")
print("=" * 70)

class_names = {
    -1: "DOWN",
    0: "NEUTRAL",
    1: "UP",
}

for prediction in [-1, 0, 1]:

    subset = df[
        df["prediction"] == prediction
    ]

    if len(subset) == 0:
        continue

    name = class_names[prediction]

    print()
    print(
        f"{name}: {len(subset):,} "
        f"({len(subset) / len(df):.2%})"
    )

    print(
        f"  Accuracy: "
        f"{subset['correct'].mean():.2%}"
    )

    print(
        f"  Avg future 1h return: "
        f"{subset['future_return_1h'].mean():+.4%}"
    )

    print(
        f"  Median future 1h return: "
        f"{subset['future_return_1h'].median():+.4%}"
    )

    print(
        f"  UP outcomes: "
        f"{(subset['target'] == 1).mean():.2%}"
    )

    print(
        f"  DOWN outcomes: "
        f"{(subset['target'] == -1).mean():.2%}"
    )

    print(
        f"  NEUTRAL outcomes: "
        f"{(subset['target'] == 0).mean():.2%}"
    )


# ============================================================
# CONFIDENCE BUCKETS
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

# Confidence buckets are deliberately broad enough to avoid
# over-interpreting tiny samples.

bins = [
    0.0,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    1.01,
]

labels = [
    "<35%",
    "35-40%",
    "40-45%",
    "45-50%",
    "50-55%",
    "55-60%",
    "60-70%",
    "70%+",
]

df["confidence_bucket"] = pd.cut(
    df["confidence"],
    bins=bins,
    labels=labels,
    right=False,
)

confidence_rows = []

for bucket in labels:

    subset = df[
        df["confidence_bucket"] == bucket
    ]

    if len(subset) == 0:
        continue

    accuracy_bucket = subset["correct"].mean()

    avg_return = (
        subset["future_return_1h"].mean()
    )

    median_return = (
        subset["future_return_1h"].median()
    )

    confidence_rows.append(
        {
            "confidence_bucket": bucket,
            "rows": len(subset),
            "percentage_of_validation": (
                len(subset) / len(df)
            ),
            "accuracy": accuracy_bucket,
            "avg_future_return": avg_return,
            "median_future_return": median_return,
            "avg_return_after_cost": (
                avg_return - ROUND_TRIP_COST
            ),
        }
    )

    print()
    print(f"{bucket}")

    print(
        f"  Rows: "
        f"{len(subset):,}"
    )

    print(
        f"  Share: "
        f"{len(subset) / len(df):.2%}"
    )

    print(
        f"  Accuracy: "
        f"{accuracy_bucket:.2%}"
    )

    print(
        f"  Avg future return: "
        f"{avg_return:+.4%}"
    )

    print(
        f"  Median future return: "
        f"{median_return:+.4%}"
    )

    print(
        f"  Avg return after 0.14% cost: "
        f"{avg_return - ROUND_TRIP_COST:+.4%}"
    )


# ============================================================
# DIRECTIONAL TRADING EDGE
# ============================================================

print()
print("=" * 70)
print("DIRECTIONAL EDGE")
print("=" * 70)

# For a directional trade:
#
# predicted UP   -> long
# predicted DOWN -> short
# predicted NEUTRAL -> no trade
#
# We calculate the actual future return in the direction
# predicted by the model.

trade_df = df[
    df["prediction"].isin([-1, 1])
].copy()

if len(trade_df) > 0:

    trade_df["gross_directional_return"] = np.where(
        trade_df["prediction"] == 1,
        trade_df["future_return_1h"],
        -trade_df["future_return_1h"],
    )

    trade_df["net_directional_return"] = (
        trade_df["gross_directional_return"]
        - ROUND_TRIP_COST
    )

    print(
        f"Directional predictions: "
        f"{len(trade_df):,}"
    )

    print(
        f"Average gross directional return: "
        f"{trade_df['gross_directional_return'].mean():+.4%}"
    )

    print(
        f"Median gross directional return: "
        f"{trade_df['gross_directional_return'].median():+.4%}"
    )

    print(
        f"Average net directional return: "
        f"{trade_df['net_directional_return'].mean():+.4%}"
    )

    print(
        f"Win rate before cost: "
        f"{(trade_df['gross_directional_return'] > 0).mean():.2%}"
    )

    print(
        f"Win rate after cost: "
        f"{(trade_df['net_directional_return'] > 0).mean():.2%}"
    )


# ============================================================
# HIGH-CONFIDENCE DIRECTIONAL TRADING
# ============================================================

print()
print("=" * 70)
print("HIGH-CONFIDENCE DIRECTIONAL ANALYSIS")
print("=" * 70)

confidence_thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

high_confidence_rows = []

for threshold in confidence_thresholds:

    subset = df[
        (df["prediction"].isin([-1, 1]))
        & (df["confidence"] >= threshold)
    ].copy()

    if len(subset) == 0:
        continue

    subset["gross_directional_return"] = np.where(
        subset["prediction"] == 1,
        subset["future_return_1h"],
        -subset["future_return_1h"],
    )

    subset["net_directional_return"] = (
        subset["gross_directional_return"]
        - ROUND_TRIP_COST
    )

    row = {
        "confidence_threshold": threshold,
        "trades": len(subset),
        "share_of_validation": (
            len(subset) / len(df)
        ),
        "accuracy": subset["correct"].mean(),
        "avg_gross_return": (
            subset["gross_directional_return"].mean()
        ),
        "median_gross_return": (
            subset["gross_directional_return"].median()
        ),
        "avg_net_return": (
            subset["net_directional_return"].mean()
        ),
        "net_win_rate": (
            subset["net_directional_return"] > 0
        ).mean(),
    }

    high_confidence_rows.append(row)

    print()
    print(
        f"Confidence >= {threshold:.0%}"
    )

    print(
        f"  Trades: "
        f"{len(subset):,}"
    )

    print(
        f"  Share: "
        f"{len(subset) / len(df):.2%}"
    )

    print(
        f"  Accuracy: "
        f"{subset['correct'].mean():.2%}"
    )

    print(
        f"  Avg gross directional return: "
        f"{subset['gross_directional_return'].mean():+.4%}"
    )

    print(
        f"  Avg net return: "
        f"{subset['net_directional_return'].mean():+.4%}"
    )

    print(
        f"  Net win rate: "
        f"{(subset['net_directional_return'] > 0).mean():.2%}"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

analysis_df = pd.DataFrame(
    confidence_rows
)

if not analysis_df.empty:
    analysis_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

print()
print("=" * 70)
print("ML VALIDATION ANALYSIS COMPLETE")
print("=" * 70)

print(f"Saved: {OUTPUT_FILE}")

print()
print(
    "Important: the test set has still NOT been evaluated."
)