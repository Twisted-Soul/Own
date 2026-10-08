from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_ml_dataset.csv"
)

RANDOM_STATE = 42

ROUND_TRIP_COST = 0.0014


# Same normalized feature set used in the successful
# experiments.

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

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True,
)

print(
    f"Loaded {len(df):,} rows."
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


# ============================================================
# MATRICES
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train_class = train_df["target"]
y_train_return = train_df["future_return_1h"]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation_class = validation_df["target"]

X_test = test_df[FEATURE_COLUMNS]
y_test_class = test_df["target"]
y_test_return = test_df["future_return_1h"]


# ============================================================
# TEST PERIOD
# ============================================================

print()
print("=" * 70)
print("UNTOUCHED TEST PERIOD")
print("=" * 70)

print(
    f"Rows: {len(test_df):,}"
)

print(
    f"Start: {test_df['timestamps'].min()}"
)

print(
    f"End:   {test_df['timestamps'].max()}"
)

print()
print("Test target distribution:")

counts = (
    y_test_class
    .value_counts()
    .sort_index()
)

for target, count in counts.items():

    label = {
        -1: "DOWN",
        0: "NEUTRAL",
        1: "UP",
    }[target]

    print(
        f"  {label:8s}: "
        f"{count:7,} "
        f"({count / len(test_df):.2%})"
    )


# ============================================================
# NAIVE TEST BASELINE
# ============================================================

print()
print("=" * 70)
print("NAIVE BASELINES — TEST")
print("=" * 70)

majority_class = y_train_class.mode()[0]

majority_prediction = np.full(
    len(y_test_class),
    majority_class,
)

print(
    f"Always DOWN/majority "
    f"accuracy: "
    f"{accuracy_score(y_test_class, majority_prediction):.4%}"
)

print(
    f"Always DOWN/majority "
    f"balanced accuracy: "
    f"{balanced_accuracy_score(y_test_class, majority_prediction):.4%}"
)


# Previous 1h momentum

momentum = test_df[
    "return_1h"
].to_numpy()

momentum_prediction = np.where(
    momentum >= 0.001,
    1,
    np.where(
        momentum <= -0.001,
        -1,
        0,
    ),
)

print(
    f"Previous 1h momentum "
    f"accuracy: "
    f"{accuracy_score(y_test_class, momentum_prediction):.4%}"
)

print(
    f"Previous 1h momentum "
    f"balanced accuracy: "
    f"{balanced_accuracy_score(y_test_class, momentum_prediction):.4%}"
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print()
print("=" * 70)
print("LOGISTIC REGRESSION — TEST")
print("=" * 70)

logistic_model = Pipeline(
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
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print(
    "Training only on TRAIN data..."
)

logistic_model.fit(
    X_train,
    y_train_class,
)

logistic_test_pred = (
    logistic_model.predict(
        X_test
    )
)

print(
    f"Accuracy: "
    f"{accuracy_score(y_test_class, logistic_test_pred):.4%}"
)

print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test_class, logistic_test_pred):.4%}"
)


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST CLASSIFIER — TEST")
print("=" * 70)

rf_classifier = Pipeline(
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

print(
    "Training only on TRAIN data..."
)

rf_classifier.fit(
    X_train,
    y_train_class,
)

rf_class_test_pred = (
    rf_classifier.predict(
        X_test
    )
)

print(
    f"Accuracy: "
    f"{accuracy_score(y_test_class, rf_class_test_pred):.4%}"
)

print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test_class, rf_class_test_pred):.4%}"
)


# ============================================================
# RIDGE REGRESSION
# ============================================================

print()
print("=" * 70)
print("RIDGE REGRESSION — TEST")
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

print(
    "Training only on TRAIN data..."
)

ridge_model.fit(
    X_train,
    y_train_return,
)

ridge_test_pred = (
    ridge_model.predict(
        X_test
    )
)

ridge_mae = mean_absolute_error(
    y_test_return,
    ridge_test_pred,
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test_return,
        ridge_test_pred,
    )
)

ridge_r2 = r2_score(
    y_test_return,
    ridge_test_pred,
)

ridge_corr = np.corrcoef(
    y_test_return,
    ridge_test_pred,
)[0, 1]

print(
    f"MAE:             {ridge_mae:.6%}"
)

print(
    f"RMSE:            {ridge_rmse:.6%}"
)

print(
    f"R²:              {ridge_r2:.6f}"
)

print(
    f"Correlation:     {ridge_corr:.6f}"
)


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST REGRESSION — TEST")
print("=" * 70)

rf_regressor = Pipeline(
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

print(
    "Training only on TRAIN data..."
)

rf_regressor.fit(
    X_train,
    y_train_return,
)

rf_regression_test_pred = (
    rf_regressor.predict(
        X_test
    )
)

rf_mae = mean_absolute_error(
    y_test_return,
    rf_regression_test_pred,
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test_return,
        rf_regression_test_pred,
    )
)

rf_r2 = r2_score(
    y_test_return,
    rf_regression_test_pred,
)

rf_corr = np.corrcoef(
    y_test_return,
    rf_regression_test_pred,
)[0, 1]

print(
    f"MAE:             {rf_mae:.6%}"
)

print(
    f"RMSE:            {rf_rmse:.6%}"
)

print(
    f"R²:              {rf_r2:.6f}"
)

print(
    f"Correlation:     {rf_corr:.6f}"
)


# ============================================================
# DIRECTIONAL REGRESSION TEST
# ============================================================

print()
print("=" * 70)
print("REGRESSION DIRECTIONAL TEST — TEST")
print("=" * 70)


def directional_test(
    name,
    predictions,
):

    directions = np.sign(
        predictions
    )

    actual = y_test_return.to_numpy()

    mask = directions != 0

    gross_returns = (
        actual[mask]
        * directions[mask]
    )

    net_returns = (
        gross_returns
        - ROUND_TRIP_COST
    )

    print()
    print(name)

    print(
        f"Trades: "
        f"{len(gross_returns):,}"
    )

    print(
        f"Gross avg: "
        f"{gross_returns.mean():+.6%}"
    )

    print(
        f"Gross median: "
        f"{np.median(gross_returns):+.6%}"
    )

    print(
        f"Net avg: "
        f"{net_returns.mean():+.6%}"
    )

    print(
        f"Net win rate: "
        f"{(net_returns > 0).mean():.2%}"
    )


directional_test(
    "RIDGE",
    ridge_test_pred,
)

directional_test(
    "RANDOM FOREST",
    rf_regression_test_pred,
)


# ============================================================
# CLASSIFICATION DIRECTIONAL TEST
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION DIRECTIONAL TEST — TEST")
print("=" * 70)


def classification_directional_test(
    name,
    predictions,
):

    actual = y_test_return.to_numpy()

    mask = predictions != 0

    directional_prediction = (
        predictions[mask]
    )

    gross_returns = (
        actual[mask]
        * directional_prediction
    )

    net_returns = (
        gross_returns
        - ROUND_TRIP_COST
    )

    print()
    print(name)

    print(
        f"Trades: "
        f"{len(gross_returns):,}"
    )

    print(
        f"Gross avg: "
        f"{gross_returns.mean():+.6%}"
    )

    print(
        f"Gross median: "
        f"{np.median(gross_returns):+.6%}"
    )

    print(
        f"Net avg: "
        f"{net_returns.mean():+.6%}"
    )

    print(
        f"Net win rate: "
        f"{(net_returns > 0).mean():.2%}"
    )


classification_directional_test(
    "LOGISTIC REGRESSION",
    logistic_test_pred,
)

classification_directional_test(
    "RANDOM FOREST CLASSIFIER",
    rf_class_test_pred,
)


# ============================================================
# TEST PERIOD MARKET STATISTICS
# ============================================================

print()
print("=" * 70)
print("TEST PERIOD MARKET STATISTICS")
print("=" * 70)

print(
    f"Future 1h mean: "
    f"{y_test_return.mean():+.6%}"
)

print(
    f"Future 1h median: "
    f"{y_test_return.median():+.6%}"
)

print(
    f"Future 1h std: "
    f"{y_test_return.std():.6%}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = test_df[
    [
        "timestamps",
        "future_return_1h",
        "target",
    ]
].copy()

results[
    "logistic_prediction"
] = logistic_test_pred

results[
    "rf_class_prediction"
] = rf_class_test_pred

results[
    "ridge_prediction"
] = ridge_test_pred

results[
    "rf_regression_prediction"
] = rf_regression_test_pred


OUTPUT_FILE = Path(
    "data/btc/"
    "BTCUSDT_5m_ml_test_predictions.csv"
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("UNTOUCHED TEST EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print(
    "This is the first evaluation of the untouched "
    "out-of-sample period."
)