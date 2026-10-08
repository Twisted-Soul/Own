from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_dataset.csv"
)

RANDOM_STATE = 42

TARGET_COLUMN = "future_return_1h"


# ============================================================
# NORMALIZED / RELATIVE FEATURES
# ============================================================

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


# ============================================================
# LOAD
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

missing = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]

if missing:
    raise ValueError(
        "Missing features:\n"
        + "\n".join(missing)
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
# MATRICES
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]


# ============================================================
# TARGET STATISTICS
# ============================================================

print()
print("=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_validation),
    ("TEST", y_test),
]:

    print()
    print(name)

    print(
        f"  Mean:   {y.mean():+.6%}"
    )

    print(
        f"  Median: {y.median():+.6%}"
    )

    print(
        f"  Std:    {y.std():.6%}"
    )

    print(
        f"  MAE from zero: "
        f"{np.abs(y).mean():.6%}"
    )


# ============================================================
# BASELINE: PREDICT ZERO
# ============================================================

print()
print("=" * 70)
print("BASELINE — PREDICT ZERO RETURN")
print("=" * 70)

zero_prediction = np.zeros(
    len(y_validation)
)

zero_mae = mean_absolute_error(
    y_validation,
    zero_prediction,
)

zero_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        zero_prediction,
    )
)

print(
    f"MAE:  {zero_mae:.6%}"
)

print(
    f"RMSE: {zero_rmse:.6%}"
)


# ============================================================
# BASELINE: PREDICT TRAIN MEAN
# ============================================================

train_mean = y_train.mean()

mean_prediction = np.full(
    len(y_validation),
    train_mean,
)

mean_mae = mean_absolute_error(
    y_validation,
    mean_prediction,
)

mean_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        mean_prediction,
    )
)

print()
print("=" * 70)
print("BASELINE — PREDICT TRAIN MEAN")
print("=" * 70)

print(
    f"Train mean: {train_mean:+.6%}"
)

print(
    f"MAE:         {mean_mae:.6%}"
)

print(
    f"RMSE:        {mean_rmse:.6%}"
)


# ============================================================
# RIDGE REGRESSION
# ============================================================

print()
print("=" * 70)
print("TRAINING RIDGE REGRESSION")
print("=" * 70)

ridge_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "regressor",
            Ridge(
                alpha=10.0,
            ),
        ),
    ]
)

print("Fitting Ridge...")

ridge_model.fit(
    X_train,
    y_train,
)

print("Ridge training complete.")


ridge_validation_pred = (
    ridge_model.predict(
        X_validation
    )
)


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST REGRESSOR")
print("=" * 70)

rf_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=100,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Fitting Random Forest...")

rf_model.fit(
    X_train,
    y_train,
)

print("Random Forest training complete.")

rf_validation_pred = (
    rf_model.predict(
        X_validation
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_regression(
    name,
    y_true,
    predictions,
):

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    correlation = np.corrcoef(
        y_true,
        predictions,
    )[0, 1]

    direction_accuracy = (
        np.sign(predictions)
        == np.sign(y_true)
    ).mean()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"MAE:               {mae:.6%}"
    )

    print(
        f"RMSE:              {rmse:.6%}"
    )

    print(
        f"R²:                {r2:.6f}"
    )

    print(
        f"Prediction corr:   {correlation:.6f}"
    )

    print(
        f"Direction accuracy:{direction_accuracy:.4%}"
    )

    print(
        f"Prediction mean:   {predictions.mean():+.6%}"
    )

    print(
        f"Prediction median: {np.median(predictions):+.6%}"
    )

    print(
        f"Prediction std:    {predictions.std():.6%}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": correlation,
        "direction_accuracy": direction_accuracy,
    }


# ============================================================
# EVALUATE
# ============================================================

ridge_metrics = evaluate_regression(
    "RIDGE REGRESSION — VALIDATION",
    y_validation,
    ridge_validation_pred,
)

rf_metrics = evaluate_regression(
    "RANDOM FOREST REGRESSION — VALIDATION",
    y_validation,
    rf_validation_pred,
)


# ============================================================
# DIRECTIONAL TRADING TEST
# ============================================================

ROUND_TRIP_COST = 0.0014

print()
print("=" * 70)
print("DIRECTIONAL TRADING ANALYSIS")
print("=" * 70)


def directional_analysis(
    name,
    predictions,
):

    # Long if predicted return > 0
    # Short if predicted return < 0
    #
    # This is deliberately a simple first test.
    # We will NOT optimize thresholds yet.

    direction = np.sign(
        predictions
    )

    trade_mask = (
        direction != 0
    )

    actual = y_validation.to_numpy()

    gross_returns = (
        actual[trade_mask]
        * direction[trade_mask]
    )

    net_returns = (
        gross_returns
        - ROUND_TRIP_COST
    )

    print()
    print(name)

    print(
        f"Trades: "
        f"{len(net_returns):,}"
    )

    print(
        f"Average gross return: "
        f"{gross_returns.mean():+.6%}"
    )

    print(
        f"Median gross return: "
        f"{np.median(gross_returns):+.6%}"
    )

    print(
        f"Average net return: "
        f"{net_returns.mean():+.6%}"
    )

    print(
        f"Net win rate: "
        f"{(net_returns > 0).mean():.2%}"
    )


directional_analysis(
    "RIDGE",
    ridge_validation_pred,
)

directional_analysis(
    "RANDOM FOREST",
    rf_validation_pred,
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

results = validation_df[
    [
        "timestamps",
        "future_return_1h",
        "target",
    ]
].copy()

results[
    "ridge_prediction"
] = ridge_validation_pred

results[
    "random_forest_prediction"
] = rf_validation_pred


OUTPUT_FILE = Path(
    "data/btc/"
    "BTCUSDT_5m_ml_regression_validation.csv"
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("ML REGRESSION COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print(
    "The test set has NOT been evaluated."
)