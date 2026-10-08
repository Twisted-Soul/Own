from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_event_reactions.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "btc"
    / "BTCUSDT_1m_event_entry_analysis.csv"
)

ROUND_TRIP_COST_PCT = 0.14


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("KRONOS 1M EVENT NO-LOOKAHEAD ENTRY ANALYSIS")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"],
    utc=True,
)

print(f"Events: {len(df)}")
print()


# =============================================================================
# HELPERS
# =============================================================================

def summarize(values, label):

    values = pd.Series(values).dropna()

    if values.empty:
        return {
            "setup": label,
            "n": 0,
            "avg_gross_pct": np.nan,
            "median_gross_pct": np.nan,
            "win_rate_gross_pct": np.nan,
            "avg_net_pct": np.nan,
            "win_rate_net_pct": np.nan,
            "profit_factor_net": np.nan,
        }

    wins = values[values > ROUND_TRIP_COST_PCT]
    losses = values[values <= ROUND_TRIP_COST_PCT]

    net_values = values - ROUND_TRIP_COST_PCT

    net_wins = net_values[net_values > 0]
    net_losses = net_values[net_values <= 0]

    gross_profit = net_wins.sum()
    gross_loss = abs(net_losses.sum())

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = np.inf

    return {
        "setup": label,
        "n": len(values),
        "avg_gross_pct": values.mean(),
        "median_gross_pct": values.median(),
        "win_rate_gross_pct": (values > 0).mean() * 100,
        "avg_net_pct": net_values.mean(),
        "win_rate_net_pct": (net_values > 0).mean() * 100,
        "profit_factor_net": pf,
    }


def print_results(title, rows):

    print()
    print("-" * 100)
    print(title)
    print("-" * 100)

    for row in rows:

        pf = row["profit_factor_net"]

        if np.isinf(pf):
            pf_text = "INF"
        elif pd.isna(pf):
            pf_text = "N/A"
        else:
            pf_text = f"{pf:.3f}"

        print(
            f"{row['setup']:<55} "
            f"n={row['n']:>3} | "
            f"gross={row['avg_gross_pct']:>8.4f}% | "
            f"net={row['avg_net_pct']:>8.4f}% | "
            f"net win={row['win_rate_net_pct']:>6.2f}% | "
            f"PF={pf_text}"
        )


# =============================================================================
# IMPORTANT:
#
# A signal at 5m can only use information available at 5m.
#
# Therefore:
#
# ENTRY = 5m price
# OUTCOME = price at 20/30/45/60m
#
# We do NOT measure from the original release price.
# =============================================================================


# =============================================================================
# 5-MINUTE SHOCK -> FOLLOW-THROUGH
# =============================================================================

results = []

for shock in [0.10, 0.20, 0.30, 0.50]:

    mask = (
        df["return_5m_pct"] <= -shock
    )

    subset = df[mask]

    for horizon in [15, 30, 45, 60]:

        # Entry is at the 5-minute checkpoint.
        #
        # return_Xm is release -> X minutes.
        # We therefore need to reconstruct:
        #
        # 5m -> Xm
        #
        # from the checkpoint prices.

        entry_price = subset["price_5m"]

        exit_price = subset[f"price_{horizon}m"]

        short_return = (
            (entry_price / exit_price - 1.0)
            * 100.0
        )

        results.append(
            summarize(
                short_return,
                (
                    f"SHORT after {shock:.2f}% "
                    f"5m down shock -> {horizon}m"
                ),
            )
        )


print_results(
    "NO-LOOKAHEAD SHORT FOLLOW-THROUGH",
    results,
)


# =============================================================================
# 5m -> 15m RECOVERY
#
# Signal is only allowed after 15m.
# Entry = 15m price.
# Exit = 30/45/60m.
# =============================================================================

recovery_results = []

for recovery in [0.02, 0.05, 0.10, 0.15, 0.20]:

    mask = (
        df["recovery_5m_to_15m_pct"]
        >= recovery
    )

    subset = df[mask]

    for horizon in [30, 45, 60]:

        entry_price = subset["price_15m"]

        exit_price = subset[f"price_{horizon}m"]

        long_return = (
            (exit_price / entry_price - 1.0)
            * 100.0
        )

        recovery_results.append(
            summarize(
                long_return,
                (
                    f"LONG after 5->15 recovery "
                    f">={recovery:.2f}% -> {horizon}m"
                ),
            )
        )


print_results(
    "NO-LOOKAHEAD LONG RECOVERY",
    recovery_results,
)


# =============================================================================
# 15m -> 30m RECOVERY
#
# Signal is only allowed after 30m.
# Entry = 30m.
# Exit = 45/60m.
# =============================================================================

confirmation_results = []

for recovery in [0.02, 0.05, 0.10, 0.15, 0.20]:

    mask = (
        df["recovery_15m_to_30m_pct"]
        >= recovery
    )

    subset = df[mask]

    for horizon in [45, 60]:

        entry_price = subset["price_30m"]

        exit_price = subset[f"price_{horizon}m"]

        long_return = (
            (exit_price / entry_price - 1.0)
            * 100.0
        )

        confirmation_results.append(
            summarize(
                long_return,
                (
                    f"LONG after 15->30 recovery "
                    f">={recovery:.2f}% -> {horizon}m"
                ),
            )
        )


print_results(
    "NO-LOOKAHEAD CONFIRMED RECOVERY",
    confirmation_results,
)


# =============================================================================
# EVENT TYPE
# =============================================================================

type_results = []

for event_type in sorted(
    df["event_type"].dropna().unique()
):

    subset = df[
        df["event_type"] == event_type
    ]

    # CPI / NFP baseline using 5m -> 60m
    entry = subset["price_5m"]
    exit_price = subset["price_60m"]

    long_return = (
        (exit_price / entry - 1.0)
        * 100.0
    )

    type_results.append(
        summarize(
            long_return,
            f"{event_type} 5m->60m LONG",
        )
    )


print_results(
    "EVENT TYPE — 5m ENTRY -> 60m",
    type_results,
)


# =============================================================================
# SAVE
# =============================================================================

all_results = (
    results
    + recovery_results
    + confirmation_results
    + type_results
)

output = pd.DataFrame(all_results)

output.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 80)
print("NO-LOOKAHEAD ANALYSIS COMPLETE")
print("=" * 80)
print(f"Rows: {len(output)}")
print(f"Reference cost: {ROUND_TRIP_COST_PCT:.2f}%")
print(f"Output: {OUTPUT_FILE}")