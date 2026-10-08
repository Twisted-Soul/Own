from pathlib import Path
import json
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

LIVE_DATA = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live.csv"
SIGNAL_DATA = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_live_signal.csv"

PAPER_TRADES = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_paper_trades.csv"
PAPER_STATE = BASE_DIR / "data" / "btc" / "BTCUSDT_5m_paper_state.json"

STARTING_CAPITAL = 10_000.0

# Estimated round-trip fee + slippage.
ROUND_TRIP_COST = 0.0014

# 12 x 5-minute candles = 1 hour.
HOLDING_BARS = 12


# ============================================================
# STATE
# ============================================================

def default_state():

    return {
        "capital": STARTING_CAPITAL,
        "position": None,
        "pending_signal": None,
        "last_processed_signal_timestamp": None,
        "last_processed_candle_timestamp": None,
    }


def load_state():

    if not PAPER_STATE.exists():
        return default_state()

    try:

        with open(
            PAPER_STATE,
            "r",
            encoding="utf-8"
        ) as f:

            state = json.load(f)

        defaults = default_state()

        for key, value in defaults.items():

            if key not in state:
                state[key] = value

        return state

    except Exception as exc:

        print(
            f"Could not load paper state: {exc}"
        )

        return default_state()


def save_state(state):

    with open(
        PAPER_STATE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


# ============================================================
# DATA
# ============================================================

def load_data():

    if not LIVE_DATA.exists():

        raise FileNotFoundError(
            f"Live data not found: {LIVE_DATA}"
        )

    if not SIGNAL_DATA.exists():

        raise FileNotFoundError(
            f"Signal data not found: {SIGNAL_DATA}"
        )

    candles = pd.read_csv(
        LIVE_DATA
    )

    signals = pd.read_csv(
        SIGNAL_DATA
    )

    candles["timestamps"] = pd.to_datetime(
        candles["timestamps"],
        utc=True,
        errors="coerce"
    )

    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"],
        utc=True,
        errors="coerce"
    )

    candles = (
        candles
        .dropna(subset=["timestamps"])
        .sort_values("timestamps")
        .drop_duplicates("timestamps")
        .reset_index(drop=True)
    )

    signals = (
        signals
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return candles, signals


# ============================================================
# TRADES
# ============================================================

def empty_trades():

    return pd.DataFrame(
        columns=[
            "signal_timestamp",
            "entry_timestamp",
            "exit_timestamp",
            "side",
            "signal",
            "entry_price",
            "exit_price",
            "gross_return",
            "cost",
            "net_return",
            "capital_before",
            "capital_after",
            "exit_reason",
        ]
    )


def load_trades():

    if not PAPER_TRADES.exists():
        return empty_trades()

    return pd.read_csv(
        PAPER_TRADES
    )


def save_trades(trades):

    trades.to_csv(
        PAPER_TRADES,
        index=False
    )


# ============================================================
# RETURNS
# ============================================================

def calculate_return(
    side,
    entry_price,
    exit_price
):

    if side == "LONG":

        return (
            exit_price / entry_price
            - 1
        )

    if side == "SHORT":

        return (
            entry_price / exit_price
            - 1
        )

    raise ValueError(
        f"Unknown side: {side}"
    )


# ============================================================
# POSITION EXIT
# ============================================================

def process_open_position(
    state,
    candles,
    trades
):

    position = state.get("position")

    if position is None:
        return trades

    entry_timestamp = pd.Timestamp(
        position["entry_timestamp"]
    )

    entry_rows = candles.index[
        candles["timestamps"] == entry_timestamp
    ].tolist()

    if not entry_rows:

        print(
            "Open position entry candle is not "
            "currently available."
        )

        return trades

    entry_index = entry_rows[0]

    current_index = len(candles) - 1

    bars_held = (
        current_index - entry_index
    )

    current_price = float(
        candles.iloc[current_index]["close"]
    )

    gross_return = calculate_return(
        position["side"],
        float(position["entry_price"]),
        current_price
    )

    print()
    print("-" * 70)
    print("OPEN PAPER POSITION")
    print("-" * 70)

    print(
        f"Side:             {position['side']}"
    )

    print(
        f"Entry time:       {position['entry_timestamp']}"
    )

    print(
        f"Entry price:      ${float(position['entry_price']):,.2f}"
    )

    print(
        f"Current time:     {candles.iloc[current_index]['timestamps']}"
    )

    print(
        f"Current price:    ${current_price:,.2f}"
    )

    print(
        f"Bars held:        {bars_held}"
    )

    print(
        f"Unrealized gross: {gross_return * 100:+.4f}%"
    )

    # --------------------------------------------------------
    # Exit after 12 completed candles after entry.
    # --------------------------------------------------------

    if bars_held < HOLDING_BARS:

        print(
            f"Status:           HOLDING "
            f"({HOLDING_BARS - bars_held} bars remaining)"
        )

        return trades

    # --------------------------------------------------------
    # Close position.
    # --------------------------------------------------------

    net_return = (
        gross_return
        - ROUND_TRIP_COST
    )

    capital_before = float(
        state["capital"]
    )

    capital_after = (
        capital_before
        * (1 + net_return)
    )

    exit_timestamp = (
        candles.iloc[current_index]["timestamps"]
    )

    trade = {

        "signal_timestamp":
            position["signal_timestamp"],

        "entry_timestamp":
            position["entry_timestamp"],

        "exit_timestamp":
            str(exit_timestamp),

        "side":
            position["side"],

        "signal":
            position["signal"],

        "entry_price":
            float(position["entry_price"]),

        "exit_price":
            current_price,

        "gross_return":
            gross_return,

        "cost":
            ROUND_TRIP_COST,

        "net_return":
            net_return,

        "capital_before":
            capital_before,

        "capital_after":
            capital_after,

        "exit_reason":
            "FIXED_1H",
    }

    trades = pd.concat(
        [
            trades,
            pd.DataFrame([trade])
        ],
        ignore_index=True
    )

    state["capital"] = capital_after
    state["position"] = None

    print()
    print("POSITION CLOSED")
    print("-" * 70)

    print(
        f"Exit time:        {exit_timestamp}"
    )

    print(
        f"Exit price:       ${current_price:,.2f}"
    )

    print(
        f"Gross return:     {gross_return * 100:+.4f}%"
    )

    print(
        f"Trading cost:     {ROUND_TRIP_COST * 100:.4f}%"
    )

    print(
        f"Net return:       {net_return * 100:+.4f}%"
    )

    print(
        f"Capital after:    ${capital_after:,.2f}"
    )

    return trades


# ============================================================
# PENDING SIGNAL
# ============================================================

def process_pending_signal(
    state,
    candles
):

    pending = state.get("pending_signal")

    if pending is None:
        return

    signal_timestamp = pd.Timestamp(
        pending["signal_timestamp"]
    )

    signal_rows = candles.index[
        candles["timestamps"] == signal_timestamp
    ].tolist()

    if not signal_rows:

        print(
            "Pending signal candle not found."
        )

        return

    signal_index = signal_rows[0]

    next_index = signal_index + 1

    # --------------------------------------------------------
    # The next candle does not exist yet.
    # --------------------------------------------------------

    if next_index >= len(candles):

        print()
        print(
            "Pending actionable signal:"
        )

        print(
            f"Signal:     {pending['signal']}"
        )

        print(
            f"Signal time: {signal_timestamp}"
        )

        print(
            "Waiting for next candle OPEN..."
        )

        return

    # --------------------------------------------------------
    # Enter at next candle OPEN.
    # --------------------------------------------------------

    entry_candle = candles.iloc[next_index]

    entry_timestamp = (
        entry_candle["timestamps"]
    )

    entry_price = float(
        entry_candle["open"]
    )

    signal = pending["signal"]

    side = (
        "LONG"
        if signal == "BUY"
        else "SHORT"
    )

    state["position"] = {

        "signal_timestamp":
            str(signal_timestamp),

        "entry_timestamp":
            str(entry_timestamp),

        "entry_price":
            entry_price,

        "side":
            side,

        "signal":
            signal,
    }

    state["pending_signal"] = None

    print()
    print("=" * 70)
    print("PAPER POSITION OPENED")
    print("=" * 70)

    print(
        f"Signal:           {signal}"
    )

    print(
        f"Side:             {side}"
    )

    print(
        f"Signal candle:    {signal_timestamp}"
    )

    print(
        f"Entry candle:     {entry_timestamp}"
    )

    print(
        f"Entry price:      ${entry_price:,.2f}"
    )


# ============================================================
# NEW SIGNAL
# ============================================================

def process_new_signal(
    state,
    candles,
    signals
):

    if signals.empty:
        return

    latest_signal = signals.iloc[-1]

    signal_timestamp = latest_signal["timestamp"]

    signal = str(
        latest_signal["signal"]
    ).upper()

    # --------------------------------------------------------
    # Don't process the same signal repeatedly.
    # --------------------------------------------------------

    last_processed = (
        state.get(
            "last_processed_signal_timestamp"
        )
    )

    if last_processed == str(signal_timestamp):
        return

    print()
    print("-" * 70)
    print("NEW SIGNAL")
    print("-" * 70)

    print(
        f"Signal timestamp: {signal_timestamp}"
    )

    print(
        f"Signal:           {signal}"
    )

    # --------------------------------------------------------
    # HOLD does not require an entry.
    # --------------------------------------------------------

    if signal == "HOLD":

        print(
            "HOLD signal — no paper position."
        )

        state[
            "last_processed_signal_timestamp"
        ] = str(signal_timestamp)

        return

    # --------------------------------------------------------
    # Only BUY and SELL are actionable.
    # --------------------------------------------------------

    if signal not in ("BUY", "SELL"):

        print(
            f"Unknown signal '{signal}' — ignored."
        )

        state[
            "last_processed_signal_timestamp"
        ] = str(signal_timestamp)

        return

    # --------------------------------------------------------
    # Store the signal as pending.
    #
    # We DO NOT mark it processed yet.
    # This is important because the next candle may not
    # exist yet.
    # --------------------------------------------------------

    state["pending_signal"] = {

        "signal_timestamp":
            str(signal_timestamp),

        "signal":
            signal,
    }

    state[
        "last_processed_signal_timestamp"
    ] = str(signal_timestamp)

    print(
        "Actionable signal stored as PENDING."
    )

    print(
        "Entry will occur at the next candle OPEN."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KRONOS PAPER TRADING ENGINE")
    print("=" * 70)

    print(
        "PAPER MODE ONLY — NO REAL ORDERS"
    )

    print(
        f"Starting capital: ${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Round-trip cost:  "
        f"{ROUND_TRIP_COST * 100:.3f}%"
    )

    print(
        f"Holding period:   "
        f"{HOLDING_BARS} candles "
        f"({HOLDING_BARS * 5} minutes)"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    candles, signals = load_data()

    state = load_state()
    trades = load_trades()

    print(
        f"Live candles:     {len(candles):,}"
    )

    print(
        f"Signals:          {len(signals):,}"
    )

    print(
        f"Completed trades: {len(trades):,}"
    )

    # --------------------------------------------------------
    # Current candle
    # --------------------------------------------------------

    latest_timestamp = (
        candles.iloc[-1]["timestamps"]
    )

    latest_close = float(
        candles.iloc[-1]["close"]
    )

    # --------------------------------------------------------
    # 1. Manage existing position.
    # --------------------------------------------------------

    trades = process_open_position(
        state,
        candles,
        trades
    )

    # --------------------------------------------------------
    # 2. If an actionable signal was pending,
    #    attempt entry at next candle open.
    # --------------------------------------------------------

    if state.get("position") is None:

        process_pending_signal(
            state,
            candles
        )

    # --------------------------------------------------------
    # 3. If no position and no pending signal,
    #    process latest signal.
    # --------------------------------------------------------

    if (
        state.get("position") is None
        and state.get("pending_signal") is None
    ):

        process_new_signal(
            state,
            candles,
            signals
        )

    # --------------------------------------------------------
    # Account summary.
    # --------------------------------------------------------

    state[
        "last_processed_candle_timestamp"
    ] = str(latest_timestamp)

    save_trades(trades)
    save_state(state)

    print()
    print("=" * 70)
    print("PAPER ACCOUNT")
    print("=" * 70)

    print(
        f"Capital:          "
        f"${state['capital']:,.2f}"
    )

    print(
        f"Closed trades:    "
        f"{len(trades):,}"
    )

    if state.get("position") is not None:

        position = state["position"]

        print(
            f"Open position:    "
            f"{position['side']}"
        )

        print(
            f"Entry price:      "
            f"${float(position['entry_price']):,.2f}"
        )

    else:

        print(
            "Open position:    NONE"
        )

    if state.get("pending_signal") is not None:

        print(
            f"Pending signal:   "
            f"{state['pending_signal']['signal']}"
        )

    else:

        print(
            "Pending signal:   NONE"
        )

    print()
    print(
        f"Trades file:      "
        f"{PAPER_TRADES}"
    )

    print(
        f"State file:       "
        f"{PAPER_STATE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()