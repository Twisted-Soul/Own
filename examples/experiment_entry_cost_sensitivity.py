import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/btc/BTCUSDT_entry_quality_events.csv")
OUTPUT = Path("data/btc/BTCUSDT_entry_cost_sensitivity.csv")

# Illustrative costs per side, in percentage points.
# 0.02 means 0.02% per side, or 0.04% round trip.
COSTS_PER_SIDE_PCT = [0.0, 0.02, 0.05, 0.07, 0.10]

HORIZONS = ["5m", "15m", "30m", "60m"]


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        return np.nan
    return gains / losses


def main():
    df = pd.read_csv(INPUT)
    results = []

    for horizon in HORIZONS:
        col = f"signal_return_{horizon}_pct"

        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

        for direction in ["ALL", "BUY", "SELL"]:
            if direction == "ALL":
                sample = df[col].dropna()
            else:
                sample = df.loc[df["signal"] == direction, col].dropna()

            for cost in COSTS_PER_SIDE_PCT:
                # Returns and costs are both expressed in percent.
                net = sample - 2 * cost

                results.append({
                    "horizon": horizon,
                    "signal": direction,
                    "cost_per_side_pct": cost,
                    "samples": len(net),
                    "net_win_rate_pct": (
                        (net > 0).mean() * 100
                        if len(net) else np.nan
                    ),
                    "mean_net_return_pct": (
                        net.mean() if len(net) else np.nan
                    ),
                    "median_net_return_pct": (
                        net.median() if len(net) else np.nan
                    ),
                    "net_profit_factor": profit_factor(net),
                })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)

    print("\n=== ENTRY COST SENSITIVITY ===")
    print(out.round(4).to_string(index=False))
    print("\nSaved:", OUTPUT)
    print("This is a diagnostic, not a full strategy backtest.")
    print("Costs are illustrative, not venue-verified.")


if __name__ == "__main__":
    main()
