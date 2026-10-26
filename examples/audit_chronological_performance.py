import pandas as pd
import numpy as np
from pathlib import Path

SOURCE = Path("data/btc/BTCUSDT_chronological_trades.csv")

df = pd.read_csv(SOURCE)
df["entry_time"] = pd.to_datetime(
    df["entry_time"], utc=True, errors="coerce"
)

df = df[
    (df["tp_pct"] == 0.20)
    & (df["sl_pct"] == 0.20)
].copy()

df["month"] = df["entry_time"].dt.strftime("%Y-%m")
df["net_return_pct"] = df["gross_return_pct"] - 0.04

def profit_factor(values):
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses == 0:
        return np.nan
    return gains / losses

def summarize(group):
    gross = group["gross_return_pct"]
    net = group["net_return_pct"]

    return pd.Series({
        "trades": len(group),
        "gross_win_rate_pct": (gross > 0).mean() * 100,
        "mean_gross_pct": gross.mean(),
        "mean_net_pct": net.mean(),
        "net_profit_factor": profit_factor(net),
        "tp_exits": group["exit_reason"].isin(
            ["TAKE_PROFIT", "TARGET_GAP"]
        ).sum(),
        "stop_exits": group["exit_reason"].isin(
            ["STOP_LOSS", "STOP_GAP", "BOTH_HIT_STOP_FIRST"]
        ).sum(),
        "time_exits": (group["exit_reason"] == "MAX_HOLD_TIME").sum(),
    })

print("=== MONTHLY PERFORMANCE: TP 0.20% / SL 0.20% ===")
monthly = (
    df.groupby(["split", "month"], sort=True)
      .apply(summarize, include_groups=False)
)
print(monthly.round(4).to_string())

print("\n=== PERFORMANCE BY DIRECTION ===")
directional = (
    df.groupby(["split", "signal"], sort=True)
      .apply(summarize, include_groups=False)
)
print(directional.round(4).to_string())

print("\n=== EXIT REASONS ===")
print(
    df.groupby(["split", "exit_reason"])
      .size()
      .rename("trades")
      .to_string()
)
