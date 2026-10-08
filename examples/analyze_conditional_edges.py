import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/btc/BTCUSDT_5m_1y_features.csv")
OUTPUT_FILE = Path("data/btc/BTCUSDT_5m_conditional_edges.csv")

ROUND_TRIP_COST = 0.0014

# ============================================================
# LOAD
# ============================================================

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

df["timestamps"] = pd.to_datetime(
    df["timestamps"],
    utc=True
)

df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Loaded {len(df):,} candles")
print(f"Start: {df['timestamps'].iloc[0]}")
print(f"End:   {df['timestamps'].iloc[-1]}")

# ============================================================
# FUTURE RETURNS
# ============================================================

HORIZONS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
}

for name, bars in HORIZONS.items():

    df[f"future_{name}"] = (
        df["close"].shift(-bars) / df["close"] - 1
    )

# ============================================================
# MARKET STATES
# ============================================================

# Trend
df["trend"] = np.select(
    [
        (
            (df["ema_20"] > df["ema_50"])
            & (df["ema_50"] > df["ema_200"])
        ),
        (
            (df["ema_20"] < df["ema_50"])
            & (df["ema_50"] < df["ema_200"])
        ),
    ],
    [
        "BULL",
        "BEAR",
    ],
    default="MIXED",
)

# Momentum
df["momentum"] = np.select(
    [
        df["return_1h"] >= 0.002,
        df["return_1h"] <= -0.002,
    ],
    [
        "POSITIVE",
        "NEGATIVE",
    ],
    default="NEUTRAL",
)

# RSI
df["rsi_state"] = np.select(
    [
        df["rsi_14"] < 30,
        df["rsi_14"] > 70,
    ],
    [
        "OVERSOLD",
        "OVERBOUGHT",
    ],
    default="NORMAL",
)

# Volume
df["volume_state"] = np.select(
    [
        df["relative_volume"] >= 1.5,
        df["relative_volume"] < 0.75,
    ],
    [
        "HIGH",
        "LOW",
    ],
    default="NORMAL",
)

# Volatility
vol_threshold = df["volatility_20"].median()

df["volatility_state"] = np.where(
    df["volatility_20"] >= vol_threshold,
    "HIGH",
    "LOW",
)

# ============================================================
# ANALYSIS FUNCTION
# ============================================================

results = []


def analyze_group(name, mask):

    subset = df.loc[mask].copy()

    if len(subset) < 100:
        return

    row = {
        "condition": name,
        "samples": len(subset),
    }

    for horizon in HORIZONS:

        values = subset[f"future_{horizon}"].dropna()

        if len(values) == 0:
            continue

        gross_mean = values.mean()
        median = values.median()

        up_rate = (values > 0).mean()

        # A hypothetical LONG
        long_net = gross_mean - ROUND_TRIP_COST

        # A hypothetical SHORT
        short_net = -gross_mean - ROUND_TRIP_COST

        row[f"{horizon}_gross"] = gross_mean
        row[f"{horizon}_median"] = median
        row[f"{horizon}_up_rate"] = up_rate
        row[f"{horizon}_long_net"] = long_net
        row[f"{horizon}_short_net"] = short_net

    results.append(row)


# ============================================================
# INDIVIDUAL STATES
# ============================================================

print()
print("Analyzing individual states...")

for value in ["BULL", "BEAR", "MIXED"]:
    analyze_group(
        f"TREND={value}",
        df["trend"] == value
    )

for value in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
    analyze_group(
        f"MOMENTUM={value}",
        df["momentum"] == value
    )

for value in ["OVERSOLD", "OVERBOUGHT", "NORMAL"]:
    analyze_group(
        f"RSI={value}",
        df["rsi_state"] == value
    )

for value in ["HIGH", "NORMAL", "LOW"]:
    analyze_group(
        f"VOLUME={value}",
        df["volume_state"] == value
    )

for value in ["HIGH", "LOW"]:
    analyze_group(
        f"VOLATILITY={value}",
        df["volatility_state"] == value
    )

# ============================================================
# TWO-FACTOR STATES
# ============================================================

print("Analyzing two-factor states...")

for trend in ["BULL", "BEAR", "MIXED"]:

    for momentum in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:

        mask = (
            (df["trend"] == trend)
            & (df["momentum"] == momentum)
        )

        analyze_group(
            f"{trend}|{momentum}",
            mask
        )


for trend in ["BULL", "BEAR", "MIXED"]:

    for volatility in ["HIGH", "LOW"]:

        mask = (
            (df["trend"] == trend)
            & (df["volatility_state"] == volatility)
        )

        analyze_group(
            f"{trend}|VOL={volatility}",
            mask
        )


for momentum in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:

    for volatility in ["HIGH", "LOW"]:

        mask = (
            (df["momentum"] == momentum)
            & (df["volatility_state"] == volatility)
        )

        analyze_group(
            f"{momentum}|VOL={volatility}",
            mask
        )


for rsi in ["OVERSOLD", "OVERBOUGHT", "NORMAL"]:

    for volume in ["HIGH", "NORMAL", "LOW"]:

        mask = (
            (df["rsi_state"] == rsi)
            & (df["volume_state"] == volume)
        )

        analyze_group(
            f"RSI={rsi}|VOLUME={volume}",
            mask
        )

# ============================================================
# THREE-FACTOR STATES
# ============================================================

print("Analyzing three-factor states...")

for trend in ["BULL", "BEAR", "MIXED"]:

    for momentum in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:

        for volatility in ["HIGH", "LOW"]:

            mask = (
                (df["trend"] == trend)
                & (df["momentum"] == momentum)
                & (df["volatility_state"] == volatility)
            )

            analyze_group(
                f"{trend}|{momentum}|VOL={volatility}",
                mask
            )

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# DISPLAY IMPORTANT RESULTS
# ============================================================

print()
print("=" * 90)
print("CONDITIONAL EDGE ANALYSIS")
print("=" * 90)

print()
print("Interpretation:")
print("long_net  = average future return - 0.14% cost")
print("short_net = -average future return - 0.14% cost")
print()

# Show states with the largest absolute 1h gross movement
ranked = results_df.sort_values(
    "1h_gross",
    key=lambda x: x.abs(),
    ascending=False
)

print("TOP CONDITIONS BY ABSOLUTE 1H MOVEMENT")
print("-" * 90)

for _, row in ranked.head(20).iterrows():

    print(
        f"{row['condition']:<35} "
        f"N={int(row['samples']):>6} "
        f"1h gross={row['1h_gross']:>8.4%} "
        f"UP={row['1h_up_rate']:>6.2%} "
        f"LONG net={row['1h_long_net']:>8.4%} "
        f"SHORT net={row['1h_short_net']:>8.4%}"
    )

# ============================================================
# BEST POTENTIAL LONG CONDITIONS
# ============================================================

print()
print("LARGEST POSITIVE 1H GROSS RETURNS")
print("-" * 90)

best_long = results_df.sort_values(
    "1h_gross",
    ascending=False
)

for _, row in best_long.head(15).iterrows():

    print(
        f"{row['condition']:<35} "
        f"N={int(row['samples']):>6} "
        f"gross={row['1h_gross']:>8.4%} "
        f"UP={row['1h_up_rate']:>6.2%} "
        f"net={row['1h_long_net']:>8.4%}"
    )

# ============================================================
# BEST POTENTIAL SHORT CONDITIONS
# ============================================================

print()
print("LARGEST NEGATIVE 1H GROSS RETURNS")
print("-" * 90)

best_short = results_df.sort_values(
    "1h_gross",
    ascending=True
)

for _, row in best_short.head(15).iterrows():

    print(
        f"{row['condition']:<35} "
        f"N={int(row['samples']):>6} "
        f"gross={row['1h_gross']:>8.4%} "
        f"UP={row['1h_up_rate']:>6.2%} "
        f"short net={row['1h_short_net']:>8.4%}"
    )

print()
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print(f"Saved: {OUTPUT_FILE}")
print(f"Conditions analyzed: {len(results_df):,}")