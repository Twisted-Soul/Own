from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

LIVE_DATA = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live.csv"
LIVE_FEATURES = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live_features.csv"
LIVE_SIGNAL = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live_signal.csv"


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def build_features(df):
    df = df.copy()

    df["timestamps"] = pd.to_datetime(
        df["timestamps"],
        utc=True,
        errors="coerce"
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["timestamps"] + numeric_cols)
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["ema_20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema_50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema_200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema_12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = df["close"].shift(1)

    tr_1 = df["high"] - df["low"]
    tr_2 = (df["high"] - previous_close).abs()
    tr_3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr_1, tr_2, tr_3],
        axis=1
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_ma_20"] = df["volume"].rolling(20).mean()

    df["relative_volume"] = (
        df["volume"] / df["volume_ma_20"]
    )

    # --------------------------------------------------------
    # Price ranges
    # --------------------------------------------------------

    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()

    df["high_50"] = df["high"].rolling(50).max()
    df["low_50"] = df["low"].rolling(50).min()

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_5m"] = (
        df["close"] / df["close"].shift(1) - 1
    )

    df["return_15m"] = (
        df["close"] / df["close"].shift(3) - 1
    )

    df["return_1h"] = (
        df["close"] / df["close"].shift(12) - 1
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_20"] = (
        df["return_5m"].rolling(20).std()
    )

    # --------------------------------------------------------
    # EMA distances
    # --------------------------------------------------------

    df["distance_ema_20"] = (
        df["close"] / df["ema_20"] - 1
    )

    df["distance_ema_50"] = (
        df["close"] / df["ema_50"] - 1
    )

    df["distance_ema_200"] = (
        df["close"] / df["ema_200"] - 1
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"] - df["low"]
    )

    df["candle_body"] = (
        df["close"] - df["open"]
    )

    df["candle_body_pct"] = (
        df["candle_body"] / df["open"]
    )

    return df


# ============================================================
# SIGNAL LOGIC
# ============================================================

def generate_signal(row):

    bullish_score = 0
    bearish_score = 0

    reasons_bullish = []
    reasons_bearish = []

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if row["rsi_14"] < 30:
        bullish_score += 1
        reasons_bullish.append("RSI oversold")

    elif row["rsi_14"] > 70:
        bearish_score += 1
        reasons_bearish.append("RSI overbought")

    # --------------------------------------------------------
    # EMA trend
    # --------------------------------------------------------

    if row["ema_20"] > row["ema_50"]:
        bullish_score += 1
        reasons_bullish.append("EMA20 > EMA50")

    elif row["ema_20"] < row["ema_50"]:
        bearish_score += 1
        reasons_bearish.append("EMA20 < EMA50")

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if row["macd"] > row["macd_signal"]:
        bullish_score += 1
        reasons_bullish.append("MACD bullish")

    elif row["macd"] < row["macd_signal"]:
        bearish_score += 1
        reasons_bearish.append("MACD bearish")

    # --------------------------------------------------------
    # 1-hour momentum
    # --------------------------------------------------------

    if row["return_1h"] > 0:
        bullish_score += 1
        reasons_bullish.append("1h momentum positive")

    elif row["return_1h"] < 0:
        bearish_score += 1
        reasons_bearish.append("1h momentum negative")

    # --------------------------------------------------------
    # Relative volume adjustment
    #
    # Same logic used by the original signal generator:
    # strong volume doubles the momentum direction.
    # --------------------------------------------------------

    if row["relative_volume"] > 1.5:

        if row["return_1h"] > 0:
            bullish_score += 1
            reasons_bullish.append("High volume + positive momentum")

        elif row["return_1h"] < 0:
            bearish_score += 1
            reasons_bearish.append("High volume + negative momentum")

    # --------------------------------------------------------
    # Final signal
    # --------------------------------------------------------

    if bullish_score >= 3 and bullish_score > bearish_score:
        signal = "BUY"

    elif bearish_score >= 3 and bearish_score > bullish_score:
        signal = "SELL"

    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "bullish_reasons": " | ".join(reasons_bullish),
        "bearish_reasons": " | ".join(reasons_bearish),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KRONOS LIVE SIGNAL ENGINE")
    print("=" * 70)

    if not LIVE_DATA.exists():
        raise FileNotFoundError(
            f"Live data file not found: {LIVE_DATA}"
        )

    print(f"Input:    {LIVE_DATA}")
    print(f"Features: {LIVE_FEATURES}")
    print(f"Signal:   {LIVE_SIGNAL}")
    print()

    # --------------------------------------------------------
    # Load live candles
    # --------------------------------------------------------

    df = pd.read_csv(LIVE_DATA)

    print(f"Loaded candles: {len(df):,}")

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = build_features(df)

    if len(features) == 0:
        raise RuntimeError("No usable candle data.")

    # Save complete live feature dataset
    features.to_csv(
        LIVE_FEATURES,
        index=False
    )

    latest = features.iloc[-1]

    # --------------------------------------------------------
    # Generate signal
    # --------------------------------------------------------

    result = generate_signal(latest)

    timestamp = latest["timestamps"]

    print()
    print("-" * 70)
    print("LATEST COMPLETED CANDLE")
    print("-" * 70)

    print(f"Timestamp:       {timestamp}")
    print(f"Open:            {latest['open']:.2f}")
    print(f"High:            {latest['high']:.2f}")
    print(f"Low:             {latest['low']:.2f}")
    print(f"Close:           {latest['close']:.2f}")
    print()

    print("TECHNICAL STATE")
    print("-" * 70)

    print(f"RSI 14:          {latest['rsi_14']:.2f}")
    print(f"EMA 20:          {latest['ema_20']:.2f}")
    print(f"EMA 50:          {latest['ema_50']:.2f}")
    print(f"EMA 200:         {latest['ema_200']:.2f}")
    print(f"MACD:            {latest['macd']:.4f}")
    print(f"MACD Signal:     {latest['macd_signal']:.4f}")
    print(f"Relative Volume: {latest['relative_volume']:.2f}")
    print(f"Return 5m:       {latest['return_5m'] * 100:.4f}%")
    print(f"Return 15m:      {latest['return_15m'] * 100:.4f}%")
    print(f"Return 1h:       {latest['return_1h'] * 100:.4f}%")
    print(f"Volatility 20:   {latest['volatility_20']:.6f}")

    print()
    print("SIGNAL")
    print("-" * 70)

    print(f"Signal:          {result['signal']}")
    print(f"Bullish score:   {result['bullish_score']}")
    print(f"Bearish score:   {result['bearish_score']}")

    print()
    print("Bullish reasons:")
    print(
        result["bullish_reasons"]
        if result["bullish_reasons"]
        else "None"
    )

    print()
    print("Bearish reasons:")
    print(
        result["bearish_reasons"]
        if result["bearish_reasons"]
        else "None"
    )

    # --------------------------------------------------------
    # Save signal
    # --------------------------------------------------------

    signal_row = {
        "timestamp": timestamp,
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "close": latest["close"],
        "rsi_14": latest["rsi_14"],
        "ema_20": latest["ema_20"],
        "ema_50": latest["ema_50"],
        "ema_200": latest["ema_200"],
        "macd": latest["macd"],
        "macd_signal": latest["macd_signal"],
        "relative_volume": latest["relative_volume"],
        "return_5m": latest["return_5m"],
        "return_15m": latest["return_15m"],
        "return_1h": latest["return_1h"],
        "volatility_20": latest["volatility_20"],
        "signal": result["signal"],
        "bullish_score": result["bullish_score"],
        "bearish_score": result["bearish_score"],
        "bullish_reasons": result["bullish_reasons"],
        "bearish_reasons": result["bearish_reasons"],
    }

    signal_df = pd.DataFrame([signal_row])

    # --------------------------------------------------------
    # Prevent duplicate candle records
    # --------------------------------------------------------

    if LIVE_SIGNAL.exists():

        existing = pd.read_csv(LIVE_SIGNAL)

        if "timestamp" in existing.columns:

            if timestamp in existing["timestamp"].astype(str).values:
                print()
                print(
                    "Signal already exists for this candle. "
                    "No duplicate written."
                )
                return

            signal_df = pd.concat(
                [existing, signal_df],
                ignore_index=True
            )

    signal_df.to_csv(
        LIVE_SIGNAL,
        index=False
    )

    print()
    print("-" * 70)
    print(f"Signal saved: {LIVE_SIGNAL}")
    print(f"Signal rows:  {len(signal_df):,}")
    print("-" * 70)


if __name__ == "__main__":
    main()