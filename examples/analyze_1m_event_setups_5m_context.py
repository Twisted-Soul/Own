
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA = Path("data/btc")

SETUP_FILE = DATA / "BTCUSDT_1m_event_trade_setups.csv"
MATRIX_FILE = DATA / "BTCUSDT_1m_event_tp_sl_matrix.csv"
FEATURE_FILE = DATA / "BTCUSDT_5m_full_features.csv"

ENRICHED_FILE = DATA / "BTCUSDT_1m_event_tp_sl_with_5m.csv"
BUCKET_FILE = DATA / "BTCUSDT_5m_feature_bucket_results.csv"
BASELINE_FILE = DATA / "BTCUSDT_5m_context_baseline_comparison.csv"

PRIMARY_TP = 0.30
PRIMARY_SL = 0.30

REQUIRED_FEATURES = [
    "ema_20", "ema_50", "ema_200",
    "rsi_14",
    "return_5m", "return_15m", "return_1h",
    "relative_volume", "atr_14", "volatility_20",
    "high_20", "low_20", "high_50", "low_50",
]


# ============================================================
# HELPERS
# ============================================================

def parse_utc(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(
            df[col], utc=True, errors="coerce"
        )
    return df


def profit_factor(returns):
    returns = pd.Series(returns).dropna()
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        return np.inf if gains > 0 else np.nan

    return gains / losses


def summarize(group):
    returns = group["net_return_pct"].dropna()

    if returns.empty:
        return pd.Series({
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "time_exits": 0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "avg_net_return_pct": np.nan,
            "total_net_return_pct": np.nan,
        })

    return pd.Series({
        "trades": len(returns),
        "wins": int((returns > 0).sum()),
        "losses": int((returns < 0).sum()),
        "time_exits": int(
            group.loc[returns.index, "exit_reason"]
            .eq("TIME")
            .sum()
        ),
        "win_rate_pct": 100 * (returns > 0).mean(),
        "profit_factor": profit_factor(returns),
        "avg_net_return_pct": returns.mean(),
        "total_net_return_pct": returns.sum(),
    })


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 72)
print("FULL 5m CONTEXT BACKTEST")
print("=" * 72)

setups = pd.read_csv(SETUP_FILE)
matrix = pd.read_csv(MATRIX_FILE)
features = pd.read_csv(FEATURE_FILE)

setups = parse_utc(
    setups,
    ["event_timestamp", "entry_timestamp"],
)

matrix = parse_utc(
    matrix,
    ["event_timestamp", "entry_timestamp", "exit_timestamp"],
)

features = parse_utc(features, ["timestamps"])

features = (
    features
    .dropna(subset=["timestamps"])
    .sort_values("timestamps")
    .reset_index(drop=True)
)

# Candles are timestamped by their OPEN time.
# A candle is usable only after its 5-minute close.
features["candle_close_time"] = (
    features["timestamps"] + pd.to_timedelta(5, unit="min")
)

print(f"1m setups: {len(setups)}")
print(f"TP/SL outcome rows: {len(matrix):,}")
print(f"5m feature candles: {len(features):,}")


# ============================================================
# ATTACH LAST COMPLETED 5m CANDLE TO EACH SETUP
# ============================================================

context_rows = []

for _, setup in setups.iterrows():
    entry_time = setup["entry_timestamp"]

    if pd.isna(entry_time):
        context_rows.append({
            "event_timestamp": setup["event_timestamp"],
            "entry_timestamp": entry_time,
            "direction": setup["direction"],
            "context_available": False,
        })
        continue

    # Strict no-lookahead rule.
    available = features[
        features["candle_close_time"] <= entry_time
    ]

    if available.empty:
        context_rows.append({
            "event_timestamp": setup["event_timestamp"],
            "entry_timestamp": entry_time,
            "direction": setup["direction"],
            "context_available": False,
        })
        continue

    candle = available.iloc[-1]

    row = {
        "event_timestamp": setup["event_timestamp"],
        "entry_timestamp": entry_time,
        "direction": setup["direction"],
        "context_available": True,
        "latest_5m_open": candle["timestamps"],
        "latest_5m_close": candle["candle_close_time"],
    }

    for col in REQUIRED_FEATURES:
        row[col] = candle[col] if col in candle.index else np.nan

    context_rows.append(row)

context = pd.DataFrame(context_rows)

keys = ["event_timestamp", "entry_timestamp", "direction"]

# One context record per setup.
assert not context.duplicated(keys).any(), (
    "Duplicate setup keys found in context."
)

# Attach the same entry-time context to every TP/SL simulation.
enriched = matrix.merge(
    context,
    on=keys,
    how="left",
    validate="many_to_one",
)

enriched["context_available"] = (
    enriched["context_available"].fillna(False).astype(bool)
)

valid_context = enriched[
    enriched["context_available"]
    & enriched[REQUIRED_FEATURES].notna().all(axis=1)
]

print()
print(f"Setups with complete context: "
      f"{context['context_available'].sum()} / {len(context)}")

print(f"TP/SL rows with complete context: "
      f"{len(valid_context):,} / {len(enriched):,}")

# Verify that selected 5m candles were actually completed by entry.
check = enriched[
    enriched["context_available"]
    & enriched["latest_5m_close"].notna()
]

assert (
    check["latest_5m_close"] <= check["entry_timestamp"]
).all(), "Lookahead detected: a selected 5m candle was unfinished."

# Save the enriched trade matrix.
enriched.to_csv(ENRICHED_FILE, index=False)


# ============================================================
# CREATE EXPLORATORY FEATURE CATEGORIES
# ============================================================

enriched["ema_structure"] = np.select(
    [
        (
            (enriched["ema_20"] > enriched["ema_50"])
            & (enriched["ema_50"] > enriched["ema_200"])
        ),
        (
            (enriched["ema_20"] < enriched["ema_50"])
            & (enriched["ema_50"] < enriched["ema_200"])
        ),
    ],
    ["BULLISH_STACK", "BEARISH_STACK"],
    default="MIXED",
)

for col in ["return_5m", "return_15m", "return_1h"]:
    enriched[f"{col}_state"] = np.select(
        [
            enriched[col] > 0,
            enriched[col] < 0,
        ],
        ["POSITIVE", "NEGATIVE"],
        default="FLAT",
    )

enriched["ema20_vs_50"] = np.select(
    [
        enriched["ema_20"] > enriched["ema_50"],
        enriched["ema_20"] < enriched["ema_50"],
    ],
    ["BULLISH", "BEARISH"],
    default="EQUAL",
)

enriched["ema50_vs_200"] = np.select(
    [
        enriched["ema_50"] > enriched["ema_200"],
        enriched["ema_50"] < enriched["ema_200"],
    ],
    ["BULLISH", "BEARISH"],
    default="EQUAL",
)

enriched["rsi_bucket"] = pd.cut(
    enriched["rsi_14"],
    bins=[-np.inf, 30, 40, 50, 60, 70, np.inf],
    labels=[
        "BELOW_30",
        "30_TO_40",
        "40_TO_50",
        "50_TO_60",
        "60_TO_70",
        "ABOVE_70",
    ],
    right=False,
).astype("string")

enriched["relative_volume_state"] = np.select(
    [
        enriched["relative_volume"] < 1,
        enriched["relative_volume"] >= 1,
    ],
    ["BELOW_AVERAGE", "AT_OR_ABOVE_AVERAGE"],
    default="UNKNOWN",
)

# Simple combined bias, retained only as an exploratory comparison.
votes = pd.DataFrame({
    "ema20_50": np.where(
        enriched["ema_20"] > enriched["ema_50"], 1,
        np.where(enriched["ema_20"] < enriched["ema_50"], -1, 0)
    ),
    "ema50_200": np.where(
        enriched["ema_50"] > enriched["ema_200"], 1,
        np.where(enriched["ema_50"] < enriched["ema_200"], -1, 0)
    ),
    "return15": np.sign(enriched["return_15m"]),
    "return1h": np.sign(enriched["return_1h"]),
    "rsi": np.where(
        enriched["rsi_14"] > 50, 1,
        np.where(enriched["rsi_14"] < 50, -1, 0)
    ),
})

enriched["context_score"] = votes.sum(axis=1)

enriched["context_bias"] = np.select(
    [
        enriched["context_score"] >= 4,
        enriched["context_score"] <= -4,
    ],
    ["BULLISH", "BEARISH"],
    default="NEUTRAL",
)

enriched["direction_alignment"] = np.select(
    [
        (
            (enriched["direction"] == "LONG")
            & (enriched["context_bias"] == "BULLISH")
        ),
        (
            (enriched["direction"] == "SHORT")
            & (enriched["context_bias"] == "BEARISH")
        ),
        (
            (enriched["direction"] == "LONG")
            & (enriched["context_bias"] == "BEARISH")
        ),
        (
            (enriched["direction"] == "SHORT")
            & (enriched["context_bias"] == "BULLISH")
        ),
    ],
    ["ALIGNED", "ALIGNED", "CONTRADICTED", "CONTRADICTED"],
    default="NEUTRAL",
)


# ============================================================
# BASELINE VS CONTEXT: PRIMARY TP/SL
# ============================================================

primary = enriched[
    np.isclose(enriched["tp_pct"], PRIMARY_TP)
    & np.isclose(enriched["sl_pct"], PRIMARY_SL)
].copy()

if primary.empty:
    raise ValueError(
        f"No rows found for TP={PRIMARY_TP}% / SL={PRIMARY_SL}%."
    )

baseline_all = summarize(primary)
baseline_complete = summarize(
    primary[primary["context_available"]]
)

comparison = pd.DataFrame([
    {
        "comparison": "BASELINE_ALL_SETUPS",
        **baseline_all.to_dict(),
    },
    {
        "comparison": "SAME_SETUPS_WITH_COMPLETE_CONTEXT",
        **baseline_complete.to_dict(),
    },
])

comparison.to_csv(BASELINE_FILE, index=False)

print()
print("=" * 72)
print(f"BASELINE COMPARISON | TP {PRIMARY_TP}% / SL {PRIMARY_SL}%")
print("=" * 72)
print(comparison.to_string(index=False))


# ============================================================
# FEATURE BUCKET SUMMARIES
# ============================================================

feature_groups = {
    "EMA structure": "ema_structure",
    "EMA20 vs EMA50": "ema20_vs_50",
    "EMA50 vs EMA200": "ema50_vs_200",
    "5m return": "return_5m_state",
    "15m return": "return_15m_state",
    "1h return": "return_1h_state",
    "RSI bucket": "rsi_bucket",
    "Relative volume": "relative_volume_state",
    "Combined context bias": "context_bias",
    "Setup alignment": "direction_alignment",
}

bucket_results = []

for feature_name, column in feature_groups.items():
    usable = enriched[
        enriched["context_available"]
        & enriched[column].notna()
    ].copy()

    for (tp, sl, bucket), group in usable.groupby(
        ["tp_pct", "sl_pct", column],
        observed=True,
        dropna=True,
    ):
        metrics = summarize(group)

        bucket_results.append({
            "feature": feature_name,
            "bucket": str(bucket),
            "tp_pct": tp,
            "sl_pct": sl,
            **metrics.to_dict(),
        })

bucket_df = pd.DataFrame(bucket_results)
bucket_df.to_csv(BUCKET_FILE, index=False)


# ============================================================
# PRINT PRIMARY CONFIGURATION RESULTS
# ============================================================

primary_buckets = bucket_df[
    np.isclose(bucket_df["tp_pct"], PRIMARY_TP)
    & np.isclose(bucket_df["sl_pct"], PRIMARY_SL)
].copy()

print()
print("=" * 72)
print("FEATURE RESULTS | PRIMARY TP/SL")
print("=" * 72)

if not primary_buckets.empty:
    for feature_name in feature_groups:
        subset = primary_buckets[
            primary_buckets["feature"] == feature_name
        ].sort_values("bucket")

        print()
        print(f"--- {feature_name} ---")
        print(
            subset[
                [
                    "bucket",
                    "trades",
                    "wins",
                    "losses",
                    "win_rate_pct",
                    "profit_factor",
                    "avg_net_return_pct",
                    "total_net_return_pct",
                ]
            ].to_string(index=False)
        )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 72)
print("BACKTEST COMPLETE")
print("=" * 72)
print(f"Enriched outcomes: {ENRICHED_FILE}")
print(f"Feature buckets:   {BUCKET_FILE}")
print(f"Baseline check:    {BASELINE_FILE}")
print()
print("Reminder: only 39 event setups. Results are exploratory,")
print("not proof of a durable edge or a basis for live leverage.")