import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_volatility_breakout.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_breakout_gross_analysis.csv"
)

STARTING_CAPITAL = 10_000.0

HORIZONS = [
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
]

# ============================================================
# LOAD
# ============================================================

print("Loading breakout trades...")

df = pd.read_csv(INPUT_FILE)

df["signal_timestamp"] = pd.to_datetime(
    df["signal_timestamp"],
    utc=True
)

df["entry_timestamp"] = pd.to_datetime(
    df["entry_timestamp"],
    utc=True
)

df["exit_timestamp"] = pd.to_datetime(
    df["exit_timestamp"],
    utc=True
)

print(f"Loaded {len(df):,} trades")

# ============================================================
# ANALYSIS
# ============================================================

results = []

print()
print("=" * 85)
print("BREAKOUT GROSS EDGE ANALYSIS")
print("=" * 85)

for horizon in HORIZONS:

    trades = df[
        df["horizon"] == horizon
    ].copy()

    if trades.empty:
        continue

    # --------------------------------------------------------
    # Gross statistics
    # --------------------------------------------------------

    gross = trades["gross_return"]

    wins = gross[gross > 0]
    losses = gross[gross < 0]

    win_rate = (
        (gross > 0).mean()
    )

    avg_gross = gross.mean()
    median_gross = gross.median()

    avg_winner = (
        wins.mean()
        if len(wins)
        else np.nan
    )

    avg_loser = (
        losses.mean()
        if len(losses)
        else np.nan
    )

    gross_profit = (
        wins.sum()
    )

    gross_loss = (
        abs(losses.sum())
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    # --------------------------------------------------------
    # Gross equity curve
    # --------------------------------------------------------

    capital = STARTING_CAPITAL

    equity = []

    for r in gross:

        capital *= (1 + r)

        equity.append(capital)

    equity = pd.Series(equity)

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1

    max_drawdown = drawdown.min()

    final_capital = capital

    total_return = (
        final_capital
        / STARTING_CAPITAL
    ) - 1

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    longs = trades[
        trades["direction"] == "LONG"
    ]

    shorts = trades[
        trades["direction"] == "SHORT"
    ]

    long_avg = (
        longs["gross_return"].mean()
        if len(longs)
        else np.nan
    )

    short_avg = (
        shorts["gross_return"].mean()
        if len(shorts)
        else np.nan
    )

    long_win = (
        (longs["gross_return"] > 0).mean()
        if len(longs)
        else np.nan
    )

    short_win = (
        (shorts["gross_return"] > 0).mean()
        if len(shorts)
        else np.nan
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append({
        "horizon": horizon,
        "trades": len(trades),
        "win_rate": win_rate,
        "average_gross": avg_gross,
        "median_gross": median_gross,
        "average_winner": avg_winner,
        "average_loser": avg_loser,
        "profit_factor": profit_factor,
        "final_capital": final_capital,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "long_trades": len(longs),
        "long_win_rate": long_win,
        "long_average": long_avg,
        "short_trades": len(shorts),
        "short_win_rate": short_win,
        "short_average": short_avg,
    })

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("-" * 85)
    print(horizon)
    print("-" * 85)

    print(
        f"Trades:              {len(trades):,}"
    )

    print(
        f"Win rate:            {win_rate:.2%}"
    )

    print(
        f"Average gross:       {avg_gross:.4%}"
    )

    print(
        f"Median gross:        {median_gross:.4%}"
    )

    print(
        f"Average winner:      {avg_winner:.4%}"
    )

    print(
        f"Average loser:       {avg_loser:.4%}"
    )

    print(
        f"Profit factor:       {profit_factor:.3f}"
    )

    print(
        f"Final gross capital:  ${final_capital:,.2f}"
    )

    print(
        f"Gross total return:   {total_return:.2%}"
    )

    print(
        f"Gross max drawdown:   {max_drawdown:.2%}"
    )

    print()
    print("LONG")

    print(
        f"Trades:              {len(longs):,}"
    )

    print(
        f"Win rate:            {long_win:.2%}"
    )

    print(
        f"Average gross:       {long_avg:.4%}"
    )

    print()
    print("SHORT")

    print(
        f"Trades:              {len(shorts):,}"
    )

    print(
        f"Win rate:            {short_win:.2%}"
    )

    print(
        f"Average gross:       {short_avg:.4%}"
    )

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 85)
print("ANALYSIS COMPLETE")
print("=" * 85)

print(
    f"Saved: {OUTPUT_FILE}"
)