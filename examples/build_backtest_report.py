from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/btc")

OUTPUT_FILE = (
    DATA_DIR
    / "Kronos_research_summary.csv"
)


# ============================================================
# HELPER
# ============================================================

def add_result(
    rows,
    strategy,
    horizon,
    trades,
    win_rate,
    avg_gross,
    avg_net,
    final_capital,
    total_return,
    max_drawdown,
    source,
):
    rows.append(
        {
            "strategy": strategy,
            "horizon": horizon,
            "trades": trades,
            "win_rate": win_rate,
            "avg_gross": avg_gross,
            "avg_net": avg_net,
            "final_capital": final_capital,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "source": source,
        }
    )


# ============================================================
# LOAD ML TEST RESULTS
# ============================================================

rows = []

ml_file = (
    DATA_DIR
    / "BTCUSDT_5m_ml_test_backtest.csv"
)

if ml_file.exists():

    ml = pd.read_csv(
        ml_file
    )

    print(
        f"Loaded ML backtest: "
        f"{len(ml):,} trades"
    )

    for strategy, group in ml.groupby(
        "strategy"
    ):

        trades = len(group)

        wins = (
            group["net_return"] > 0
        ).sum()

        win_rate = (
            wins / trades
            if trades
            else 0
        )

        avg_gross = (
            group["gross_return"].mean()
        )

        avg_net = (
            group["net_return"].mean()
        )

        final_capital = (
            group["capital_after"].iloc[-1]
        )

        total_return = (
            final_capital / 10_000
            - 1
        )

        equity = (
            group["capital_after"]
        )

        running_max = (
            equity.cummax()
        )

        drawdown = (
            equity / running_max - 1
        )

        max_drawdown = (
            drawdown.min()
        )

        add_result(
            rows=rows,
            strategy=f"ML {strategy}",
            horizon="1h",
            trades=trades,
            win_rate=win_rate,
            avg_gross=avg_gross,
            avg_net=avg_net,
            final_capital=final_capital,
            total_return=total_return,
            max_drawdown=max_drawdown,
            source=ml_file.name,
        )

else:

    print(
        f"WARNING: {ml_file} not found"
    )


# ============================================================
# LOAD VOLATILITY BREAKOUT RESULTS
# ============================================================

breakout_file = (
    DATA_DIR
    / "BTCUSDT_5m_volatility_breakout.csv"
)

if breakout_file.exists():

    breakout = pd.read_csv(
        breakout_file
    )

    print(
        f"Loaded breakout results: "
        f"{len(breakout):,} rows"
    )

    # The file contains one row per trade/horizon.
    # Detect the horizon column automatically.

    possible_horizon_columns = [
        "horizon",
        "horizon_minutes",
        "holding_minutes",
    ]

    horizon_column = None

    for column in possible_horizon_columns:

        if column in breakout.columns:

            horizon_column = column
            break

    if horizon_column:

        print(
            f"Breakout horizon column: "
            f"{horizon_column}"
        )

else:

    print(
        f"WARNING: {breakout_file} not found"
    )


# ============================================================
# LOAD CONDITIONAL EDGE ANALYSIS
# ============================================================

conditional_file = (
    DATA_DIR
    / "BTCUSDT_5m_conditional_edges.csv"
)

if conditional_file.exists():

    conditional = pd.read_csv(
        conditional_file
    )

    print(
        f"Loaded conditional edge analysis: "
        f"{len(conditional):,} conditions"
    )

else:

    print(
        f"WARNING: "
        f"{conditional_file} not found"
    )


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    rows
)

if summary.empty:

    raise RuntimeError(
        "No research results were found."
    )


# Sort best to worst by total return.

summary = summary.sort_values(
    "total_return",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("KRONOS RESEARCH SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

best = summary.iloc[0]

print(
    f"Best recorded ML strategy: "
    f"{best['strategy']}"
)

print(
    f"Return: "
    f"{best['total_return']:+.2%}"
)

print(
    f"Average net trade: "
    f"{best['avg_net']:+.4%}"
)

print(
    f"Maximum drawdown: "
    f"{best['max_drawdown']:.2%}"
)

print()
print(
    "This report is descriptive only."
)

print(
    "It does not perform parameter optimization "
    "or alter any historical results."
)