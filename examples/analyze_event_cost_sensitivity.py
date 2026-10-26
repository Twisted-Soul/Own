
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/btc/BTCUSDT_event_oos_split.csv")
OUTPUT = Path("data/btc/BTCUSDT_event_cost_sensitivity.csv")

# Total round-trip cost assumptions, in percentage points.
COST_SCENARIOS = [0.14, 0.18, 0.22, 0.30]

df = pd.read_csv(INPUT)

required = [
    "period", "direction", "event_type",
    "gross_return_pct", "entry_timestamp",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["gross_return_pct"] = pd.to_numeric(
    df["gross_return_pct"], errors="coerce"
)
df = df.dropna(subset=["gross_return_pct"]).copy()

# The original net return should approximately equal
# gross return minus the assumed 0.14% round-trip cost.
if "net_return_pct" in df.columns:
    check = (
        df["gross_return_pct"] - df["net_return_pct"]
    ).dropna()

    print("Original cost consistency check:")
    print(f"  Median implied cost: {check.median():.6f}%")
    print(f"  Minimum implied cost: {check.min():.6f}%")
    print(f"  Maximum implied cost: {check.max():.6f}%")

rows = []

for cost in COST_SCENARIOS:
    df["scenario_net_return_pct"] = (
        df["gross_return_pct"] - cost
    )

    for (period, direction), group in df.groupby(
        ["period", "direction"]
    ):
        r = group["scenario_net_return_pct"]

        positive = r[r > 0].sum()
        negative = -r[r < 0].sum()

        pf = (
            positive / negative
            if negative > 0
            else (np.inf if positive > 0 else np.nan)
        )

        rows.append({
            "round_trip_cost_pct": cost,
            "period": period,
            "direction": direction,
            "trades": len(r),
            "wins": int((r > 0).sum()),
            "losses": int((r < 0).sum()),
            "win_rate_pct": 100 * (r > 0).mean(),
            "profit_factor": pf,
            "avg_net_return_pct": r.mean(),
            "total_net_return_pct": r.sum(),
        })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print("\n" + "=" * 90)
print("EVENT STRATEGY: COST SENSITIVITY")
print("=" * 90)
print(result.to_string(index=False))
print(f"\nSaved: {OUTPUT}")
print(
    "\nThese scenarios subtract a fixed assumed round-trip cost "
    "from gross returns. They do not model price-dependent slippage "
    "or changes to stop execution."
)