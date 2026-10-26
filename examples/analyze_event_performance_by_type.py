
from pathlib import Path

import numpy as np
import pandas as pd

FILE = Path("data/btc/BTCUSDT_1m_event_tp_sl_matrix.csv")
OUTPUT = Path("data/btc/BTCUSDT_event_performance_by_type.csv")

TP = 0.30
SL = 0.30

df = pd.read_csv(FILE)

df = df[
    np.isclose(df["tp_pct"], TP)
    & np.isclose(df["sl_pct"], SL)
].copy()

def summarize(group):
    r = group["net_return_pct"].dropna()

    wins = r[r > 0]
    losses = r[r < 0]

    gross_wins = wins.sum()
    gross_losses = -losses.sum()

    pf = (
        gross_wins / gross_losses
        if gross_losses > 0
        else (np.inf if gross_wins > 0 else np.nan)
    )

    return pd.Series({
        "trades": len(r),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate_pct": 100 * (r > 0).mean() if len(r) else np.nan,
        "profit_factor": pf,
        "avg_net_return_pct": r.mean() if len(r) else np.nan,
        "total_net_return_pct": r.sum() if len(r) else np.nan,
        "tp_exits": int((group["exit_reason"] == "TP").sum()),
        "sl_exits": int((group["exit_reason"] == "SL").sum()),
        "time_exits": int((group["exit_reason"] == "TIME").sum()),
    })

result = (
    df.groupby(["event_type", "direction"], dropna=False)
    .apply(summarize, include_groups=False)
    .reset_index()
)

result.to_csv(OUTPUT, index=False)

print("=" * 72)
print("EVENT PERFORMANCE BY TYPE AND DIRECTION")
print("=" * 72)
print(result.to_string(index=False))
print(f"\nSaved: {OUTPUT}")

print("\nOVERALL CHECK")
print(summarize(df).to_string())