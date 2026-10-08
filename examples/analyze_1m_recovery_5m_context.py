from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "btc"

EVENT_FILE = DATA_DIR / "BTCUSDT_1m_event_reactions.csv"
FEATURE_FILE = DATA_DIR / "BTCUSDT_5m_1y_features.csv"

OUTPUT_FILE = DATA_DIR / "BTCUSDT_1m_recovery_5m_context.csv"
SUMMARY_FILE = DATA_DIR / "BTCUSDT_1m_recovery_5m_context_summary.csv"

ROUND_TRIP_COST = 0.0014  # 0.14% reference cost


# ============================================================
# LOAD
# ============================================================

events = pd.read_csv(EVENT_FILE)
features = pd.read_csv(FEATURE_FILE)

events["event_timestamp"] = pd.to_datetime(
    events["event_timestamp"], utc=True
)

features["timestamps"] = pd.to_datetime(
    features["timestamps"], utc=True
)

events = events.sort_values("event_timestamp").reset_index(drop=True)
features = features.sort_values("timestamps").reset_index(drop=True)


print("=" * 70)
print("1M EVENT REACTION + 5M MARKET CONTEXT ANALYSIS")
print("=" * 70)

print(f"1m events: {len(events):,}")
print(
    f"5m feature rows: {len(features):,}"
)

print(
    f"5m data range: "
    f"{features['timestamps'].min()} -> "
    f"{features['timestamps'].max()}"
)


# ============================================================
# MATCH EACH EVENT TO THE LATEST COMPLETED 5M CANDLE
# ============================================================

context_cols = [
    "timestamps",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "relative_volume",
    "return_5m",
    "return_15m",
    "return_1h",
    "volatility_20",
    "distance_ema_20",
    "distance_ema_50",
    "distance_ema_200",
]

features_context = features[context_cols].copy()

# Important:
# direction="backward" means we use the most recent completed
# 5m candle at or before the event timestamp.
#
# This avoids using the 5m candle that is still forming after
# the economic release.

merged = pd.merge_asof(
    events.sort_values("event_timestamp"),
    features_context.sort_values("timestamps"),
    left_on="event_timestamp",
    right_on="timestamps",
    direction="backward",
    tolerance=pd.Timedelta("10min"),
)

matched = merged["timestamps"].notna()

print()
print("5M CONTEXT MATCHING")
print("-" * 70)
print(f"Matched events   : {matched.sum():,}")
print(f"Unmatched events : {(~matched).sum():,}")


# ============================================================
# 5M MARKET CONTEXT
# ============================================================

def classify_context(row):
    ema20 = row["ema_20"]
    ema50 = row["ema_50"]
    ret15 = row["return_15m"]

    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ret15):
        return "UNKNOWN"

    if ema20 > ema50 and ret15 > 0:
        return "BULLISH"

    if ema20 < ema50 and ret15 < 0:
        return "BEARISH"

    return "NEUTRAL"


merged["context_5m"] = merged.apply(
    classify_context,
    axis=1,
)


# ============================================================
# EVENT SHOCK CLASSIFICATION
# ============================================================

# We use the first 5-minute reaction as the initial shock.
#
# This is NOT yet a trading signal.
# It is simply the event reaction state.

SHOCK_THRESHOLD = 0.10

merged["shock_type"] = np.select(
    [
        merged["return_5m_pct"] >= SHOCK_THRESHOLD,
        merged["return_5m_pct"] <= -SHOCK_THRESHOLD,
    ],
    [
        "UPSIDE_SHOCK",
        "DOWNSIDE_SHOCK",
    ],
    default="SMALL_MOVE",
)


# ============================================================
# RECOVERY / REJECTION CLASSIFICATION
# ============================================================

# Downside shock:
# We are interested in whether BTC recovers after the shock.
#
# Upside shock:
# We are interested in whether BTC rejects/reverses after
# the initial upside move.

RECOVERY_THRESHOLD = 0.20

merged["recovery_candidate"] = "NO"

downside_recovery = (
    (merged["shock_type"] == "DOWNSIDE_SHOCK")
    & (merged["recovery_5m_to_15m_pct"] >= RECOVERY_THRESHOLD)
)

upside_rejection = (
    (merged["shock_type"] == "UPSIDE_SHOCK")
    & (merged["recovery_5m_to_15m_pct"] <= -RECOVERY_THRESHOLD)
)

merged.loc[downside_recovery, "recovery_candidate"] = "LONG_RECOVERY"

merged.loc[upside_rejection, "recovery_candidate"] = "SHORT_REJECTION"


# ============================================================
# DIRECTIONAL OUTCOME
# ============================================================

# LONG:
# positive future return = favorable
#
# SHORT:
# negative future return = favorable

def directional_return(row, horizon):
    value = row[f"return_{horizon}m_pct"]

    if row["recovery_candidate"] == "LONG_RECOVERY":
        return value

    if row["recovery_candidate"] == "SHORT_REJECTION":
        return -value

    return np.nan


for horizon in [15, 30, 45, 60]:
    merged[f"directional_{horizon}m_pct"] = merged.apply(
        lambda row: directional_return(row, horizon),
        axis=1,
    )

    merged[f"net_{horizon}m_pct"] = (
        merged[f"directional_{horizon}m_pct"]
        - ROUND_TRIP_COST * 100
    )


# ============================================================
# SAVE EVENT-LEVEL DATA
# ============================================================

merged.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved event-level analysis:")
print(OUTPUT_FILE)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(df, label):
    rows = []

    for horizon in [15, 30, 45, 60]:

        col = f"directional_{horizon}m_pct"
        net_col = f"net_{horizon}m_pct"

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna()

        net_values = pd.to_numeric(
            df[net_col],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        winners = values[values > 0]
        losers = values[values < 0]

        gross_profit = winners.sum()
        gross_loss = abs(losers.sum())

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = np.inf

        rows.append(
            {
                "group": label,
                "horizon_min": horizon,
                "observations": len(values),
                "win_rate_pct": (values > 0).mean() * 100,
                "avg_gross_return_pct": values.mean(),
                "median_gross_return_pct": values.median(),
                "avg_net_return_pct": net_values.mean(),
                "median_net_return_pct": net_values.median(),
                "profit_factor": profit_factor,
                "best_trade_pct": values.max(),
                "worst_trade_pct": values.min(),
            }
        )

    return rows


# ============================================================
# ANALYSIS 1:
# ALL EVENT SHOCKS BY 5M CONTEXT
# ============================================================

summary_rows = []

for context in ["BULLISH", "NEUTRAL", "BEARISH"]:

    subset = merged[
        merged["context_5m"] == context
    ]

    print()
    print("=" * 70)
    print(f"5M CONTEXT: {context}")
    print("=" * 70)

    print(f"Events: {len(subset)}")

    if len(subset) > 0:
        print(
            subset[
                [
                    "event_timestamp",
                    "event_type",
                    "shock_type",
                    "return_5m_pct",
                    "return_15m_pct",
                    "return_30m_pct",
                    "return_45m_pct",
                    "return_60m_pct",
                ]
            ].to_string(index=False)
        )

    # Only actual recovery/rejection candidates
    candidates = subset[
        subset["recovery_candidate"] != "NO"
    ]

    print()
    print(
        f"Recovery/rejection candidates: "
        f"{len(candidates)}"
    )

    summary_rows.extend(
        summarize(
            candidates,
            f"{context}_5M",
        )
    )


# ============================================================
# ANALYSIS 2:
# LONG RECOVERY BY 5M CONTEXT
# ============================================================

for context in ["BULLISH", "NEUTRAL", "BEARISH"]:

    subset = merged[
        (merged["context_5m"] == context)
        & (merged["recovery_candidate"] == "LONG_RECOVERY")
    ]

    summary_rows.extend(
        summarize(
            subset,
            f"LONG_RECOVERY_{context}_5M",
        )
    )


# ============================================================
# ANALYSIS 3:
# SHORT REJECTION BY 5M CONTEXT
# ============================================================

for context in ["BULLISH", "NEUTRAL", "BEARISH"]:

    subset = merged[
        (merged["context_5m"] == context)
        & (merged["recovery_candidate"] == "SHORT_REJECTION")
    ]

    summary_rows.extend(
        summarize(
            subset,
            f"SHORT_REJECTION_{context}_5M",
        )
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# PRINT IMPORTANT RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

if len(summary) == 0:
    print("No qualifying recovery/rejection candidates found.")
else:
    print(
        summary.to_string(index=False)
    )

print()
print(f"Summary saved to:")
print(SUMMARY_FILE)