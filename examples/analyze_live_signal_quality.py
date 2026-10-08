from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# KRONOS LIVE SIGNAL QUALITY ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_5m_live_signal_evaluation.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_5m_live_signal_quality.csv"
)

HORIZONS = ["5m", "15m", "30m", "1h", "2h"]

print("=" * 70)
print("KRONOS LIVE SIGNAL QUALITY ANALYSIS")
print("=" * 70)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

df = (
    df
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

print(f"Signals loaded: {len(df):,}")


# ============================================================
# BASIC SIGNAL DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("SIGNAL DISTRIBUTION")
print("-" * 70)

for signal in ["BUY", "SELL", "HOLD"]:

    subset = df[df["signal"] == signal]

    pct = (
        len(subset) / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{signal:5s}: "
        f"{len(subset):4d} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("SIGNAL SCORE DISTRIBUTION")
print("-" * 70)

for signal in ["BUY", "SELL"]:

    subset = df[df["signal"] == signal].copy()

    if subset.empty:
        continue

    if signal == "BUY":
        subset["signal_strength"] = (
            subset["bullish_score"]
            - subset["bearish_score"]
        )
    else:
        subset["signal_strength"] = (
            subset["bearish_score"]
            - subset["bullish_score"]
        )

    print()
    print(signal)

    print(
        subset["signal_strength"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# DIRECTIONAL ANALYSIS
# ============================================================

results = []

print()
print("=" * 70)
print("DIRECTIONAL PERFORMANCE")
print("=" * 70)

for horizon in HORIZONS:

    return_col = f"future_return_{horizon}"
    correct_col = f"correct_{horizon}"

    print()
    print(f"{horizon} HORIZON")
    print("-" * 70)

    for signal in ["BUY", "SELL"]:

        subset = df[
            df["signal"] == signal
        ].copy()

        subset = subset[
            subset[return_col].notna()
        ]

        if subset.empty:

            print(
                f"{signal:5s}: "
                "no completed observations"
            )

            continue

        # ----------------------------------------------------
        # Directional correctness
        # ----------------------------------------------------

        correct = (
            subset[correct_col]
            .mean()
            * 100
        )

        # ----------------------------------------------------
        # Returns
        # ----------------------------------------------------

        avg_return = (
            subset[return_col]
            .mean()
            * 100
        )

        median_return = (
            subset[return_col]
            .median()
            * 100
        )

        best = (
            subset[return_col]
            .max()
            * 100
        )

        worst = (
            subset[return_col]
            .min()
            * 100
        )

        # ----------------------------------------------------
        # Directional return
        #
        # BUY  -> future return
        # SELL -> negative future return
        # ----------------------------------------------------

        if signal == "BUY":

            directional_returns = (
                subset[return_col]
            )

        else:

            directional_returns = (
                -subset[return_col]
            )

        avg_directional = (
            directional_returns
            .mean()
            * 100
        )

        median_directional = (
            directional_returns
            .median()
            * 100
        )

        # ----------------------------------------------------
        # Win / loss magnitude
        # ----------------------------------------------------

        winners = directional_returns[
            directional_returns > 0
        ]

        losers = directional_returns[
            directional_returns <= 0
        ]

        if not winners.empty:

            avg_winner = (
                winners.mean()
                * 100
            )

        else:

            avg_winner = np.nan

        if not losers.empty:

            avg_loser = (
                losers.mean()
                * 100
            )

        else:

            avg_loser = np.nan

        # ----------------------------------------------------
        # Profit factor
        # ----------------------------------------------------

        gross_profit = (
            winners.sum()
        )

        gross_loss = (
            -losers.sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = np.inf

        print(
            f"{signal:5s}: "
            f"n={len(subset):4d} | "
            f"correct={correct:6.2f}% | "
            f"directional={avg_directional:+.4f}% | "
            f"median={median_directional:+.4f}% | "
            f"PF={profit_factor:.3f}"
        )

        results.append({
            "signal": signal,
            "horizon": horizon,
            "observations": len(subset),
            "accuracy_pct": correct,
            "avg_market_return_pct": avg_return,
            "median_market_return_pct": median_return,
            "avg_directional_return_pct": avg_directional,
            "median_directional_return_pct": median_directional,
            "avg_winner_pct": avg_winner,
            "avg_loser_pct": avg_loser,
            "best_market_move_pct": best,
            "worst_market_move_pct": worst,
            "profit_factor": profit_factor,
        })


# ============================================================
# HOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("HOLD BEHAVIOR")
print("=" * 70)

for horizon in HORIZONS:

    return_col = f"future_return_{horizon}"

    subset = df[
        (df["signal"] == "HOLD")
        & df[return_col].notna()
    ]

    if subset.empty:
        continue

    avg = (
        subset[return_col]
        .mean()
        * 100
    )

    median = (
        subset[return_col]
        .median()
        * 100
    )

    std = (
        subset[return_col]
        .std()
        * 100
    )

    print(
        f"{horizon:4s}: "
        f"n={len(subset):4d} | "
        f"avg={avg:+.4f}% | "
        f"median={median:+.4f}% | "
        f"std={std:.4f}%"
    )


# ============================================================
# SCORE-STRENGTH ANALYSIS
# ============================================================

print()
print("=" * 70)
print("SIGNAL STRENGTH ANALYSIS")
print("=" * 70)

strength_rows = []

for signal in ["BUY", "SELL"]:

    subset = df[
        df["signal"] == signal
    ].copy()

    if subset.empty:
        continue

    if signal == "BUY":

        subset["strength"] = (
            subset["bullish_score"]
            - subset["bearish_score"]
        )

    else:

        subset["strength"] = (
            subset["bearish_score"]
            - subset["bullish_score"]
        )

    for strength in sorted(
        subset["strength"].dropna().unique()
    ):

        group = subset[
            subset["strength"] == strength
        ]

        print()
        print(
            f"{signal} strength +{int(strength)} "
            f"(n={len(group)})"
        )

        for horizon in HORIZONS:

            return_col = (
                f"future_return_{horizon}"
            )

            correct_col = (
                f"correct_{horizon}"
            )

            valid = group[
                group[return_col].notna()
            ]

            if valid.empty:
                continue

            accuracy = (
                valid[correct_col]
                .mean()
                * 100
            )

            if signal == "BUY":

                directional = (
                    valid[return_col]
                    .mean()
                    * 100
                )

            else:

                directional = (
                    -valid[return_col]
                    .mean()
                    * 100
                )

            print(
                f"  {horizon:4s}: "
                f"n={len(valid):3d} | "
                f"correct={accuracy:6.2f}% | "
                f"directional={directional:+.4f}%"
            )

            strength_rows.append({
                "signal": signal,
                "strength": strength,
                "horizon": horizon,
                "observations": len(valid),
                "accuracy_pct": accuracy,
                "avg_directional_return_pct": directional,
            })


# ============================================================
# SAVE RESULTS
# ============================================================

quality = pd.DataFrame(results)

strength_df = pd.DataFrame(strength_rows)

if not strength_df.empty:

    strength_df.to_csv(
        OUTPUT_FILE.with_name(
            "BTCUSDT_5m_live_signal_strength.csv"
        ),
        index=False,
    )

if not quality.empty:

    quality.to_csv(
        OUTPUT_FILE,
        index=False,
    )

print()
print("-" * 70)
print(f"Saved quality report: {OUTPUT_FILE}")

if not strength_df.empty:

    print(
        "Saved strength report: "
        f"{OUTPUT_FILE.with_name('BTCUSDT_5m_live_signal_strength.csv')}"
    )

print("-" * 70)
print("=" * 70)