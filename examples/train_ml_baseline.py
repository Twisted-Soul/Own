from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/btc/BTCUSDT_5m_ml_dataset.csv")

TARGET_COLUMN = "target"

# Columns that must never be used as ML features.
NON_FEATURE_COLUMNS = {
    "timestamps",
    "dataset",
    "future_return_1h",
    "target",
}

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def print_metrics(name, y_true, y_pred):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[-1, 0, 1],
            target_names=["DOWN", "NEUTRAL", "UP"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[-1, 0, 1],
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual DOWN", "Actual NEUTRAL", "Actual UP"],
        columns=["Pred DOWN", "Pred NEUTRAL", "Pred UP"],
    )

    print(cm_df)


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

print(f"Loaded {len(df):,} rows")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CHECK DATASET SPLIT
# ============================================================

required_splits = {"train", "validation", "test"}

if "dataset" not in df.columns:
    raise ValueError("Missing 'dataset' column.")

actual_splits = set(df["dataset"].unique())

missing_splits = required_splits - actual_splits

if missing_splits:
    raise ValueError(
        f"Missing dataset splits: {missing_splits}"
    )


train_df = df[df["dataset"] == "train"].copy()
validation_df = df[df["dataset"] == "validation"].copy()
test_df = df[df["dataset"] == "test"].copy()

print()
print("=" * 70)
print("DATASET SPLITS")
print("=" * 70)

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:
    print(
        f"{name:12s}: "
        f"{len(part):,} rows | "
        f"{part['timestamps'].min()} → "
        f"{part['timestamps'].max()}"
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

if not feature_columns:
    raise ValueError("No feature columns found.")

print()
print("=" * 70)
print("FEATURES")
print("=" * 70)

print(f"Number of features: {len(feature_columns)}")

for i, column in enumerate(feature_columns, start=1):
    print(f"{i:2d}. {column}")


X_train = train_df[feature_columns].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_validation = validation_df[feature_columns].copy()
y_validation = validation_df[TARGET_COLUMN].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df[TARGET_COLUMN].copy()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_validation),
    ("TEST", y_test),
]:
    counts = y.value_counts().sort_index()

    print()
    print(name)

    for target, count in counts.items():
        label = {
            -1: "DOWN",
            0: "NEUTRAL",
            1: "UP",
        }.get(target, str(target))

        pct = count / len(y) * 100

        print(
            f"  {label:8s}: "
            f"{count:7,} "
            f"({pct:6.2f}%)"
        )


# ============================================================
# NAIVE BASELINE 1
# MAJORITY CLASS
# ============================================================

majority_class = y_train.mode()[0]

validation_majority_pred = np.full(
    len(y_validation),
    majority_class,
)

print_metrics(
    f"NAIVE BASELINE — ALWAYS {majority_class:+d}",
    y_validation,
    validation_majority_pred,
)


# ============================================================
# NAIVE BASELINE 2
# PREVIOUS 1H MOMENTUM
# ============================================================

# return_1h is the return observed BEFORE the prediction.
#
# Positive momentum -> UP
# Negative momentum -> DOWN
# Small momentum -> NEUTRAL
#
# We use the same +/- 0.10% threshold as the target.

if "return_1h" not in feature_columns:
    raise ValueError(
        "return_1h feature is required for momentum baseline."
    )

validation_momentum = validation_df["return_1h"].to_numpy()

validation_momentum_pred = np.where(
    validation_momentum >= 0.001,
    1,
    np.where(
        validation_momentum <= -0.001,
        -1,
        0,
    ),
)

print_metrics(
    "NAIVE BASELINE — PREVIOUS 1H MOMENTUM",
    y_validation,
    validation_momentum_pred,
)


# ============================================================
# LOGISTIC REGRESSION PIPELINE
# ============================================================

print()
print("=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

# Pipeline:
#
# 1. Median imputation
# 2. Standardization
# 3. Multiclass logistic regression
#
# Standardization is especially important because the dataset
# contains features on very different scales.

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight=None,
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
# TRAIN PERFORMANCE
# ============================================================

train_pred = model.predict(X_train)

print_metrics(
    "LOGISTIC REGRESSION — TRAIN",
    y_train,
    train_pred,
)


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

validation_pred = model.predict(X_validation)

print_metrics(
    "LOGISTIC REGRESSION — VALIDATION",
    y_validation,
    validation_pred,
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)

classes = model.named_steps["classifier"].classes_

print()
print("=" * 70)
print("VALIDATION PROBABILITY SUMMARY")
print("=" * 70)

print(
    "Model classes:",
    classes.tolist(),
)

for class_value, column_index in zip(
    classes,
    range(len(classes)),
):
    label = {
        -1: "DOWN",
        0: "NEUTRAL",
        1: "UP",
    }.get(class_value, str(class_value))

    probabilities = validation_probabilities[
        :,
        column_index,
    ]

    print(
        f"{label:8s}: "
        f"mean={probabilities.mean():.4f}, "
        f"median={np.median(probabilities):.4f}, "
        f"max={probabilities.max():.4f}"
    )


# ============================================================
# TEST IS NOT USED FOR MODEL SELECTION
# ============================================================

print()
print("=" * 70)
print("TEST SET")
print("=" * 70)

print(
    "The test set has NOT been evaluated yet."
)

print(
    "It will remain untouched until the validation results "
    "are reviewed."
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

validation_results["prediction"] = validation_pred

for class_value, column_index in zip(
    classes,
    range(len(classes)),
):
    label = {
        -1: "down_probability",
        0: "neutral_probability",
        1: "up_probability",
    }[class_value]

    validation_results[label] = validation_probabilities[
        :,
        column_index,
    ]

output_file = Path(
    "data/btc/BTCUSDT_5m_ml_validation_predictions.csv"
)

validation_results.to_csv(
    output_file,
    index=False,
)

print()
print("=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)

print(f"Saved validation predictions:")
print(output_file)

print()
print("Next step:")
print(
    "Review validation performance before touching "
    "the test set."
)