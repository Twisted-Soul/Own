from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_live_features.csv"
)


print("=" * 80)
print("KRONOS 1M REACTION STATE ANALYZER")
print("=" * 80)


df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("Feature file is empty.")


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


if len(df) < 20:
    raise ValueError(
        "Not enough 1m candles for reaction analysis."
    )


# ---------------------------------------------------------------------
# Latest state
# ---------------------------------------------------------------------

row = df.iloc[-1]

previous = df.iloc[-2]

close = row["close"]


# ---------------------------------------------------------------------
# Helper values
# ---------------------------------------------------------------------

r1 = row["return_1m"]
r5 = row["return_5m"]
r15 = row["return_15m"]
r30 = row["return_30m"]
r1h = row["return_1h"]

rsi = row["rsi_14"]

relative_volume = row["relative_volume"]

volatility_expansion = row[
    "volatility_expansion"
]

range_expansion = row[
    "range_expansion"
]

recovery_15 = row[
    "recovery_from_15m_low"
]

recovery_30 = row[
    "recovery_from_30m_low"
]

distance_15_high = row[
    "distance_from_15m_high"
]

distance_30_high = row[
    "distance_from_30m_high"
]


# ---------------------------------------------------------------------
# Recent candle behavior
# ---------------------------------------------------------------------

recent = df.tail(10)

recent_returns = recent[
    "return_1m"
].dropna()

positive_minutes = (
    recent_returns > 0
).sum()

negative_minutes = (
    recent_returns < 0
).sum()


# ---------------------------------------------------------------------
# Local low / high behavior
# ---------------------------------------------------------------------

recent_low = recent["low"].min()
recent_high = recent["high"].max()

distance_recent_low = (
    close / recent_low - 1
)

distance_recent_high = (
    close / recent_high - 1
)


# ---------------------------------------------------------------------
# Determine whether downside shock exists
# ---------------------------------------------------------------------

shock_score = 0

if r5 <= -0.001:
    shock_score += 1

if r15 <= -0.002:
    shock_score += 1

if r1h <= -0.004:
    shock_score += 1

if rsi < 30:
    shock_score += 1

if range_expansion >= 1.5:
    shock_score += 1

if relative_volume >= 1.5:
    shock_score += 1


shock_detected = (
    shock_score >= 3
)


# ---------------------------------------------------------------------
# Stabilization detection
# ---------------------------------------------------------------------

stabilization_score = 0

if r1 > -0.0005:
    stabilization_score += 1

if abs(r1) < abs(r5) / 3:
    stabilization_score += 1

if volatility_expansion < 1.0:
    stabilization_score += 1

if negative_minutes <= 6:
    stabilization_score += 1

if distance_recent_low >= 0:
    stabilization_score += 1


stabilizing = (
    stabilization_score >= 3
)


# ---------------------------------------------------------------------
# Recovery detection
# ---------------------------------------------------------------------

recovery_score = 0

if recovery_15 >= 0.0005:
    recovery_score += 1

if recovery_30 >= 0.0005:
    recovery_score += 1

if r1 > 0:
    recovery_score += 1

if r5 > -0.0005:
    recovery_score += 1

if positive_minutes >= 4:
    recovery_score += 1


recovery_detected = (
    recovery_score >= 3
)


# ---------------------------------------------------------------------
# Strong recovery confirmation
# ---------------------------------------------------------------------

confirmation_score = 0

if r1 > 0:
    confirmation_score += 1

if r5 > 0:
    confirmation_score += 1

if r15 > -0.0005:
    confirmation_score += 1

if recovery_15 >= 0.001:
    confirmation_score += 1

if recovery_30 >= 0.001:
    confirmation_score += 1

if positive_minutes >= 5:
    confirmation_score += 1

if close > row["ema_20"]:
    confirmation_score += 1


confirmed_recovery = (
    confirmation_score >= 4
)


# ---------------------------------------------------------------------
# Downside continuation
# ---------------------------------------------------------------------

continuation_score = 0

if r1 < 0:
    continuation_score += 1

if r5 < -0.0005:
    continuation_score += 1

if r15 < -0.001:
    continuation_score += 1

if close < row["ema_20"]:
    continuation_score += 1

if negative_minutes >= 5:
    continuation_score += 1

if recovery_15 < 0.0005:
    continuation_score += 1


downside_continuation = (
    continuation_score >= 4
)


# ---------------------------------------------------------------------
# Final state
# ---------------------------------------------------------------------

if confirmed_recovery:

    state = "CONFIRMED_RECOVERY"

elif recovery_detected:

    state = "RECOVERY"

elif stabilizing:

    state = "STABILIZING"

elif shock_detected:

    state = "SHOCK"

elif downside_continuation:

    state = "DOWNSIDE_CONTINUATION"

else:

    state = "NEUTRAL"


# ---------------------------------------------------------------------
# Directional bias
# ---------------------------------------------------------------------

if state in [
    "RECOVERY",
    "CONFIRMED_RECOVERY",
]:

    bias = "POTENTIAL_LONG"

elif state == "DOWNSIDE_CONTINUATION":

    bias = "POTENTIAL_SHORT"

else:

    bias = "WAIT"


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

print()

print(
    f"UTC: {row['timestamp']}"
)

if "timestamp_myt" in row:

    print(
        f"MYT: {row['timestamp_myt']}"
    )

print(
    f"Close: {close:.2f}"
)

print()

print("-" * 80)
print("MARKET STATE")
print("-" * 80)

print(
    f"1m return:   {r1 * 100:+.4f}%"
)

print(
    f"5m return:   {r5 * 100:+.4f}%"
)

print(
    f"15m return:  {r15 * 100:+.4f}%"
)

print(
    f"30m return:  {r30 * 100:+.4f}%"
)

print(
    f"1h return:   {r1h * 100:+.4f}%"
)

print(
    f"RSI:         {rsi:.2f}"
)

print(
    f"Relative vol:{relative_volume:.2f}"
)

print(
    f"Vol expansion:{volatility_expansion:.2f}"
)

print(
    f"Range expansion:{range_expansion:.2f}"
)

print()

print("-" * 80)
print("REACTION ANALYSIS")
print("-" * 80)

print(
    f"Shock score: "
    f"{shock_score}/6"
)

print(
    f"Stabilization score: "
    f"{stabilization_score}/5"
)

print(
    f"Recovery score: "
    f"{recovery_score}/5"
)

print(
    f"Confirmation score: "
    f"{confirmation_score}/7"
)

print(
    f"Continuation score: "
    f"{continuation_score}/6"
)

print()

print(
    f"Recent positive minutes: "
    f"{positive_minutes}/10"
)

print(
    f"Recent negative minutes: "
    f"{negative_minutes}/10"
)

print(
    f"Recovery from 15m low: "
    f"{recovery_15 * 100:+.4f}%"
)

print(
    f"Recovery from 30m low: "
    f"{recovery_30 * 100:+.4f}%"
)

print()

print("=" * 80)
print("KRONOS REACTION STATE")
print("=" * 80)

print(
    f"STATE: {state}"
)

print(
    f"BIAS:  {bias}"
)

print("=" * 80)