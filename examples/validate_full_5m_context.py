from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/btc/BTCUSDT_5m_full_features.csv"
)

EVENT_FILE = Path(
    "data/btc/BTCUSDT_1m_event_reactions.csv"
)

SETUP_FILE = Path(
    "data/btc/BTCUSDT_1m_event_trade_setups.csv"
)

OUTPUT_FILE = Path(
    "data/btc/BTCUSDT_5m_context_validation.csv"
)


# ============================================================
# LOAD 5m FEATURES
# ============================================================

print("=" * 70)
print("VALIDATE FULL 5m CONTEXT COVERAGE")
print("=" * 70)

print()
print("Loading 5m features...")

features = pd.read_csv(
    FEATURE_FILE
)

features["timestamps"] = pd.to_datetime(
    features["timestamps"],
    utc=True,
    errors="coerce",
)

# A 5m candle beginning at 14:20 closes at 14:25.
# Therefore it is only usable at 14:25 or later.
features["candle_close_time"] = (
    features["timestamps"]
    + pd.Timedelta("5min")
)

features = (
    features
    .sort_values("timestamps")
    .reset_index(drop=True)
)


print(
    f"5m candles: {len(features):,}"
)

print(
    f"5m range: "
    f"{features['timestamps'].min()} → "
    f"{features['timestamps'].max()}"
)


# ============================================================
# LOAD EVENT DATA
# ============================================================

print()
print("Loading events...")

events = pd.read_csv(
    EVENT_FILE
)

events["event_timestamp"] = pd.to_datetime(
    events["event_timestamp"],
    utc=True,
    errors="coerce",
)

events = (
    events[
        [
            "event_timestamp",
            "event_type",
            "importance",
        ]
    ]
    .drop_duplicates()
    .sort_values("event_timestamp")
    .reset_index(drop=True)
)


print(
    f"Events: {len(events)}"
)


# ============================================================
# LOAD 1m TRADE SETUPS
# ============================================================

print()
print("Loading 1m trade setups...")

setups = pd.read_csv(
    SETUP_FILE
)

setups["entry_timestamp"] = pd.to_datetime(
    setups["entry_timestamp"],
    utc=True,
    errors="coerce",
)

setups["event_timestamp"] = pd.to_datetime(
    setups["event_timestamp"],
    utc=True,
    errors="coerce",
)

print(
    f"1m setups: {len(setups):,}"
)


# ============================================================
# REQUIRED 5m FEATURES
# ============================================================

required_columns = [
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "return_5m",
    "return_15m",
    "return_1h",
    "relative_volume",
    "atr_14",
    "volatility_20",
    "high_20",
    "low_20",
    "high_50",
    "low_50",
]


# ============================================================
# CHECK EACH SETUP
# ============================================================

print()
print("Checking each 1m setup...")


results = []


for _, setup in setups.iterrows():

    entry_time = setup["entry_timestamp"]

    if pd.isna(entry_time):

        results.append(
            {
                "event_timestamp": setup[
                    "event_timestamp"
                ],
                "event_type": setup[
                    "event_type"
                ],
                "direction": setup[
                    "direction"
                ],
                "entry_timestamp": entry_time,
                "context_available": False,
                "reason": "invalid_entry_timestamp",
            }
        )

        continue


    # --------------------------------------------------------
    # NO LOOKAHEAD
    # --------------------------------------------------------
    #
    # Only a completed 5m candle can be used.
    #
    # Example:
    #
    # 5m candle open 14:20
    # candle closes 14:25
    #
    # Entry at 14:24:
    #   cannot use 14:20 candle
    #
    # Entry at 14:25:
    #   can use 14:20 candle
    #

    candidates = features[
        features["candle_close_time"]
        <= entry_time
    ]


    if candidates.empty:

        results.append(
            {
                "event_timestamp": setup[
                    "event_timestamp"
                ],
                "event_type": setup[
                    "event_type"
                ],
                "direction": setup[
                    "direction"
                ],
                "entry_timestamp": entry_time,
                "context_available": False,
                "reason": "no_completed_5m_candle",
            }
        )

        continue


    latest = candidates.iloc[-1]


    # --------------------------------------------------------
    # CHECK REQUIRED FEATURES
    # --------------------------------------------------------

    missing = [
        column
        for column in required_columns
        if pd.isna(latest[column])
    ]


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "event_timestamp": setup[
                "event_timestamp"
            ],

            "event_type": setup[
                "event_type"
            ],

            "direction": setup[
                "direction"
            ],

            "entry_timestamp": entry_time,

            "context_available": (
                len(missing) == 0
            ),

            "reason": (
                "complete"
                if len(missing) == 0
                else ",".join(missing)
            ),

            "latest_5m_open": latest[
                "timestamps"
            ],

            "latest_5m_close": latest[
                "candle_close_time"
            ],
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# SETUP COVERAGE SUMMARY
# ============================================================

print()
print("=" * 70)
print("SETUP CONTEXT COVERAGE")
print("=" * 70)


coverage = (
    results_df[
        "context_available"
    ]
    .value_counts(
        dropna=False
    )
)


print(coverage)


available = results_df[
    results_df["context_available"]
]

unavailable = results_df[
    ~results_df["context_available"]
]


print()
print(
    f"Complete 5m context: "
    f"{len(available)} / {len(results_df)}"
)

print(
    f"Unavailable/incomplete: "
    f"{len(unavailable)}"
)


# ============================================================
# SHOW UNAVAILABLE SETUPS
# ============================================================

if not unavailable.empty:

    print()
    print("=" * 70)
    print("UNAVAILABLE / INCOMPLETE SETUPS")
    print("=" * 70)

    print(
        unavailable[
            [
                "event_timestamp",
                "event_type",
                "direction",
                "entry_timestamp",
                "reason",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# EVENT COVERAGE
# ============================================================

print()
print("=" * 70)
print("EVENT COVERAGE")
print("=" * 70)


event_results = []


for _, event in events.iterrows():

    event_time = event[
        "event_timestamp"
    ]


    candidates = features[
        features["candle_close_time"]
        <= event_time
    ]


    if candidates.empty:

        status = "NO_CONTEXT"
        latest_time = None

    else:

        latest = candidates.iloc[-1]

        missing = [
            column
            for column in required_columns
            if pd.isna(latest[column])
        ]

        if missing:

            status = "INCOMPLETE"

        else:

            status = "OK"

        latest_time = latest[
            "timestamps"
        ]


    event_results.append(
        {
            "event_timestamp": event_time,
            "event_type": event[
                "event_type"
            ],
            "status": status,
            "latest_5m_open": latest_time,
        }
    )


event_results_df = pd.DataFrame(
    event_results
)


print(
    event_results_df.to_string(
        index=False
    )
)


# ============================================================
# EVENT SUMMARY
# ============================================================

print()
print("=" * 70)
print("EVENT COVERAGE SUMMARY")
print("=" * 70)


print(
    event_results_df[
        "status"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)