from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_dataset.csv"
)

RANDOM_STATE = 42


# ============================================================
# FEATURE SET
# ============================================================
#
# We deliberately avoid raw BTC price levels.
#
# Absolute BTC price (e.g. 85,000 vs 120,000) is not itself
# a useful stationary trading feature.
#
# Instead we use normalized / relative information.

FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "relative_volume",
    "return_5m",
    "return_15m",
    "return_1h",
    "volatility_20",
    "distance_ema_20",
    "distance_ema_50",
    "distance_ema_200",
    "candle_range",
    "candle_body",
    "candle_body_pct",
]


TARGET_COLUMN = "target"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ML dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
)

print(f"Loaded {len(df):,} rows.")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing features:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# SPLITS
# ============================================================

train_df = df[
    df["dataset"] == "train"
].copy()

validation_df = df[
    df["dataset"] == "validation"
].copy()

test_df = df[
    df["dataset"] == "test"
].copy()


print()
print("=" * 70)
print("DATASET SPLITS")
print("=" * 70)

print(
    f"TRAIN:       {len(train_df):,}"
)

print(
    f"VALIDATION:  {len(validation_df):,}"
)

print(
    f"TEST:        {len(test_df):,}"
)


# ============================================================
# FEATURE MATRIX
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]


print()
print("=" * 70)
print("FEATURES")
print("=" * 70)

print(
    f"Using {len(FEATURE_COLUMNS)} normalized/relative features:"
)

for i, feature in enumerate(
    FEATURE_COLUMNS,
    start=1,
):
    print(f"{i:2d}. {feature}")


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=100,
                max_features="sqrt",
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Fitting model...")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# METRICS FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    predictions,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: {balanced_accuracy:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_true,
            predictions,
            labels=[-1, 0, 1],
            target_names=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[-1, 0, 1],
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual DOWN",
            "Actual NEUTRAL",
            "Actual UP",
        ],
        columns=[
            "Pred DOWN",
            "Pred NEUTRAL",
            "Pred UP",
        ],
    )

    print(cm_df)

    return accuracy, balanced_accuracy


# ============================================================
# TRAIN PERFORMANCE
# ============================================================

train_pred = model.predict(
    X_train
)

evaluate_model(
    "RANDOM FOREST — TRAIN",
    y_train,
    train_pred,
)


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

validation_pred = model.predict(
    X_validation
)

evaluate_model(
    "RANDOM FOREST — VALIDATION",
    y_validation,
    validation_pred,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

classifier = (
    model.named_steps["classifier"]
)

importances = classifier.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": importances,
    }
).sort_values(
    "importance",
    ascending=False,
)

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for _, row in importance_df.iterrows():

    print(
        f"{row['feature']:25s} "
        f"{row['importance']:.6f}"
    )


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = (
    model.predict_proba(
        X_validation
    )
)

classes = (
    classifier.classes_
)

print()
print("=" * 70)
print("VALIDATION PROBABILITIES")
print("=" * 70)

print(
    "Classes:",
    classes.tolist()
)

for class_value, column_index in zip(
    classes,
    range(len(classes)),
):

    label = {
        -1: "DOWN",
        0: "NEUTRAL",
        1: "UP",
    }.get(
        class_value,
        str(class_value),
    )

    probabilities = (
        validation_probabilities[
            :,
            column_index,
        ]
    )

    print(
        f"{label:8s}: "
        f"mean={probabilities.mean():.4f}, "
        f"median={np.median(probabilities):.4f}, "
        f"max={probabilities.max():.4f}"
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = validation_df[
    [
        "timestamps",
        "future_return_1h",
        "target",
    ]
].copy()

validation_results[
    "prediction"
] = validation_pred

for class_value, column_index in zip(
    classes,
    range(len(classes)),
):

    label = {
        -1: "down_probability",
        0: "neutral_probability",
        1: "up_probability",
    }[class_value]

    validation_results[label] = (
        validation_probabilities[
            :,
            column_index,
        ]
    )

output_file = Path(
    "data/btc/"
    "BTCUSDT_5m_rf_validation_predictions.csv"
)

validation_results.to_csv(
    output_file,
    index=False,
)


# ============================================================
# TEST REMAINS UNTOUCHED
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST COMPLETE")
print("=" * 70)

print(
    f"Saved validation predictions:"
)

print(
    output_file
)

print()
print(
    "The test set has NOT been evaluated."
)