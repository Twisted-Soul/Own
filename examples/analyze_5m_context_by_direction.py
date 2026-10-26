
from pathlib import Path

import numpy as np
import pandas as pd

FILE = Path("data/btc/BTCUSDT_1m_event_tp_sl_with_5m.csv")
OUTPUT = Path("data/btc/BTCUSDT_5m_context_by_direction.csv")

TP = 0.30
SL = 0.30


def profit_factor(values):
    values = pd.Series(values).dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()

    if losses == 0:
        return np.inf if gains > 0 else np.nan

    return gains / losses


def metrics(group):
    r = group["net_return_pct"].dropna()

    return pd.Series({
        "trades": len(r),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate_pct": 100 * (r > 0).mean() if len(r) else np.nan,
        "profit_factor": profit_factor(r),
        "avg_net_return_pct": r.mean() if len(r) else np.nan,
        "total_net_return_pct": r.sum() if len(r) else np.nan,
    })


df = pd.read_csv(FILE)

df = df[
    np.isclose(df["tp_pct"], TP)
    & np.isclose(df["sl_pct"], SL)
].copy()

# Keep rows with valid 5m context.
df = df[
    df["context_available"].astype(str).str.lower().eq("true")
].copy()

# Recreate derived categories from the raw saved features.
required_raw = [
    "ema_20", "ema_50", "ema_200",
    "return_15m", "return_1h", "rsi_14",
    "net_return_pct", "direction",
]

missing = [c for c in required_raw if c not in df.columns]
if missing:
    raise ValueError(f"Missing required raw columns: {missing}")

df["ema_structure"] = np.select(
    [
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"]),

        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"]),
    ],
    ["BULLISH_STACK", "BEARISH_STACK"],
    default="MIXED",
)

df["ema50_vs_200"] = np.select(
    [
        df["ema_50"] > df["ema_200"],
        df["ema_50"] < df["ema_200"],
    ],
    ["BULLISH", "BEARISH"],
    default="EQUAL",
)

df["return_15m_state"] = np.select(
    [
        df["return_15m"] > 0,
        df["return_15m"] < 0,
    ],
    ["POSITIVE", "NEGATIVE"],
    default="FLAT",
)

df["rsi_bucket"] = pd.cut(
    df["rsi_14"],
    bins=[-np.inf, 30, 40, 50, 60, 70, np.inf],
    labels=[
        "BELOW_30", "30_TO_40", "40_TO_50",
        "50_TO_60", "60_TO_70", "ABOVE_70",
    ],
    right=False,
).astype("string")

votes = pd.DataFrame({
    "ema20_50": np.where(
        df["ema_20"] > df["ema_50"], 1,
        np.where(df["ema_20"] < df["ema_50"], -1, 0)
    ),
    "ema50_200": np.where(
        df["ema_50"] > df["ema_200"], 1,
        np.where(df["ema_50"] < df["ema_200"], -1, 0)
    ),
    "return15": np.sign(df["return_15m"]),
    "return1h": np.sign(df["return_1h"]),
    "rsi": np.where(
        df["rsi_14"] > 50, 1,
        np.where(df["rsi_14"] < 50, -1, 0)
    ),
})

df["context_score"] = votes.sum(axis=1)

df["context_bias"] = np.select(
    [
        df["context_score"] >= 4,
        df["context_score"] <= -4,
    ],
    ["BULLISH", "BEARISH"],
    default="NEUTRAL",
)

df["direction_alignment"] = np.select(
    [
        (df["direction"] == "LONG") &
        (df["context_bias"] == "BULLISH"),

        (df["direction"] == "SHORT") &
        (df["context_bias"] == "BEARISH"),

        (df["direction"] == "LONG") &
        (df["context_bias"] == "BEARISH"),

        (df["direction"] == "SHORT") &
        (df["context_bias"] == "BULLISH"),
    ],
    ["ALIGNED", "ALIGNED", "CONTRADICTED", "CONTRADICTED"],
    default="NEUTRAL",
)

feature_columns = {
    "EMA structure": "ema_structure",
    "EMA50 vs EMA200": "ema50_vs_200",
    "15m momentum": "return_15m_state",
    "RSI bucket": "rsi_bucket",
    "Combined bias": "context_bias",
    "Setup alignment": "direction_alignment",
}

rows = []

# Baseline results split by direction.
for direction, group in df.groupby("direction"):
    rows.append({
        "feature": "BASELINE",
        "bucket": "ALL",
        "direction": direction,
        **metrics(group).to_dict(),
    })

# Feature buckets split by direction.
for feature_name, column in feature_columns.items():
    for (direction, bucket), group in df.groupby(
        ["direction", column], observed=True, dropna=True
    ):
        rows.append({
            "feature": feature_name,
            "bucket": str(bucket),
            "direction": direction,
            **metrics(group).to_dict(),
        })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print("=" * 72)
print("5m CONTEXT RESULTS BY TRADE DIRECTION")
print(f"TP={TP}% | SL={SL}%")
print("=" * 72)

for feature_name in ["BASELINE", *feature_columns.keys()]:
    subset = result[result["feature"] == feature_name]

    print(f"\n--- {feature_name} ---")
    print(
        subset[
            [
                "bucket", "direction", "trades", "wins", "losses",
                "win_rate_pct", "profit_factor",
                "avg_net_return_pct", "total_net_return_pct",
            ]
        ].to_string(index=False)
    )

print(f"\nSaved: {OUTPUT}")
print("Small groups are exploratory; this is not a live trading rule.")