import pandas as pd
import numpy as np


INPUT_FILE = "data/btc/BTCUSDT_5m_1y_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Loaded {len(df):,} candles")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")


# ============================================================
# FUTURE RETURNS
# ============================================================

# Future close-to-close returns.
#
# These are ONLY used for analysis.
# They are not used to generate signals.

df["future_5m"] = df["close"].shift(-1) / df["close"] - 1
df["future_15m"] = df["close"].shift(-3) / df["close"] - 1
df["future_1h"] = df["close"].shift(-12) / df["close"] - 1
df["future_4h"] = df["close"].shift(-48) / df["close"] - 1


# ============================================================
# REGIME DEFINITIONS
# ============================================================

# Trend regime
df["trend"] = np.select(
    [
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"]),

        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"]),
    ],
    [
        "BULL",
        "BEAR",
    ],
    default="MIXED"
)


# Momentum regime
df["momentum"] = np.select(
    [
        df["return_1h"] > 0.002,
        df["return_1h"] < -0.002,
    ],
    [
        "POSITIVE",
        "NEGATIVE",
    ],
    default="NEUTRAL"
)


# Volatility regime
volatility_median = df["volatility_20"].median()

df["volatility_regime"] = np.where(
    df["volatility_20"] >= volatility_median,
    "HIGH",
    "LOW"
)


# Volume regime
df["volume_regime"] = np.select(
    [
        df["relative_volume"] >= 1.5,
        df["relative_volume"] < 0.75,
    ],
    [
        "HIGH",
        "LOW",
    ],
    default="NORMAL"
)


# RSI regime
df["rsi_regime"] = np.select(
    [
        df["rsi_14"] < 30,
        df["rsi_14"] > 70,
    ],
    [
        "OVERSOLD",
        "OVERBOUGHT",
    ],
    default="NORMAL"
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_group(data, name):

    if len(data) < 100:
        return

    print("\n" + "-" * 75)
    print(name)
    print("-" * 75)

    print(f"Samples: {len(data):,}")

    for column in [
        "future_5m",
        "future_15m",
        "future_1h",
        "future_4h",
    ]:

        returns = data[column].dropna()

        if returns.empty:
            continue

        avg = returns.mean()
        median = returns.median()
        win_rate = (returns > 0).mean()

        print(
            f"{column:12s} "
            f"avg={avg * 100:8.4f}%  "
            f"median={median * 100:8.4f}%  "
            f"UP={win_rate * 100:6.2f}%"
        )


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 75)
print("BASELINE")
print("=" * 75)

analyze_group(df, "ALL DATA")


# ============================================================
# INDIVIDUAL REGIMES
# ============================================================

print("\n" + "=" * 75)
print("TREND REGIMES")
print("=" * 75)

for value in ["BULL", "BEAR", "MIXED"]:
    analyze_group(
        df[df["trend"] == value],
        f"Trend = {value}"
    )


print("\n" + "=" * 75)
print("MOMENTUM REGIMES")
print("=" * 75)

for value in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
    analyze_group(
        df[df["momentum"] == value],
        f"Momentum = {value}"
    )


print("\n" + "=" * 75)
print("VOLATILITY REGIMES")
print("=" * 75)

for value in ["HIGH", "LOW"]:
    analyze_group(
        df[df["volatility_regime"] == value],
        f"Volatility = {value}"
    )


print("\n" + "=" * 75)
print("VOLUME REGIMES")
print("=" * 75)

for value in ["HIGH", "NORMAL", "LOW"]:
    analyze_group(
        df[df["volume_regime"] == value],
        f"Volume = {value}"
    )


print("\n" + "=" * 75)
print("RSI REGIMES")
print("=" * 75)

for value in ["OVERSOLD", "NORMAL", "OVERBOUGHT"]:
    analyze_group(
        df[df["rsi_regime"] == value],
        f"RSI = {value}"
    )


# ============================================================
# COMBINED REGIMES
# ============================================================

print("\n" + "=" * 75)
print("COMBINED REGIMES")
print("=" * 75)


combinations = [
    ("BULL + POSITIVE MOMENTUM",
     (df["trend"] == "BULL") &
     (df["momentum"] == "POSITIVE")),

    ("BULL + NEGATIVE MOMENTUM",
     (df["trend"] == "BULL") &
     (df["momentum"] == "NEGATIVE")),

    ("BEAR + POSITIVE MOMENTUM",
     (df["trend"] == "BEAR") &
     (df["momentum"] == "POSITIVE")),

    ("BEAR + NEGATIVE MOMENTUM",
     (df["trend"] == "BEAR") &
     (df["momentum"] == "NEGATIVE")),

    ("BULL + HIGH VOLATILITY",
     (df["trend"] == "BULL") &
     (df["volatility_regime"] == "HIGH")),

    ("BULL + LOW VOLATILITY",
     (df["trend"] == "BULL") &
     (df["volatility_regime"] == "LOW")),

    ("BEAR + HIGH VOLATILITY",
     (df["trend"] == "BEAR") &
     (df["volatility_regime"] == "HIGH")),

    ("BEAR + LOW VOLATILITY",
     (df["trend"] == "BEAR") &
     (df["volatility_regime"] == "LOW")),

    ("HIGH VOLUME + POSITIVE MOMENTUM",
     (df["volume_regime"] == "HIGH") &
     (df["momentum"] == "POSITIVE")),

    ("HIGH VOLUME + NEGATIVE MOMENTUM",
     (df["volume_regime"] == "HIGH") &
     (df["momentum"] == "NEGATIVE")),
]


for name, condition in combinations:

    analyze_group(
        df[condition],
        name
    )


# ============================================================
# TREND + MOMENTUM + VOLATILITY
# ============================================================

print("\n" + "=" * 75)
print("THREE-FACTOR REGIMES")
print("=" * 75)

three_factor_groups = (
    df
    .groupby(
        ["trend", "momentum", "volatility_regime"],
        observed=True
    )
    .agg(
        samples=("close", "size"),
        avg_1h=("future_1h", "mean"),
        median_1h=("future_1h", "median"),
        up_1h=("future_1h", lambda x: (x > 0).mean()),
        avg_4h=("future_4h", "mean"),
        median_4h=("future_4h", "median"),
        up_4h=("future_4h", lambda x: (x > 0).mean()),
    )
    .reset_index()
)

three_factor_groups = three_factor_groups[
    three_factor_groups["samples"] >= 500
]

three_factor_groups = three_factor_groups.sort_values(
    "avg_4h",
    ascending=False
)

for _, row in three_factor_groups.iterrows():

    print(
        f"{row['trend']:5s} | "
        f"{row['momentum']:8s} | "
        f"{row['volatility_regime']:4s} | "
        f"n={int(row['samples']):6,d} | "
        f"1h={row['avg_1h'] * 100:7.4f}% | "
        f"1h UP={row['up_1h'] * 100:5.1f}% | "
        f"4h={row['avg_4h'] * 100:7.4f}% | "
        f"4h UP={row['up_4h'] * 100:5.1f}%"
    )


print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)