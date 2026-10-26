
from huggingface_hub import snapshot_download

repo_id = "ibrahimdaud/binance-btcusdt"
local_dir = "data/btc/smilyface_1m"

print("Downloading only raw BTCUSDT 1-minute candles...")

path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["raw/klines_1m/BTCUSDT/*.parquet"],
)

print(f"\nDownload complete: {path}")
print("Downloaded files are kept separate from existing project data.")