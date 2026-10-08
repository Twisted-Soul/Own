import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# =============================================================================
# KRONOS WORLD NEWS INTELLIGENCE
# =============================================================================
#
# This module defines the structured format for world-news intelligence.
#
# IMPORTANT:
# The news layer provides MARKET CONTEXT only.
# It does NOT generate BUY/SELL orders.
#
# Later this module will connect to an external AI/web-search provider.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "news"
OUTPUT_FILE = OUTPUT_DIR / "kronos_news_intelligence.json"

MYT = ZoneInfo("Asia/Kuala_Lumpur")


def utc_now():
    return datetime.now(timezone.utc)


def to_myt(dt):
    return dt.astimezone(MYT)


def create_news_item(
    headline,
    source,
    published_time,
    category,
    market_relevance,
    btc_relevance,
    risk_level,
    directional_bias,
    confidence,
    reason,
):
    """
    Create one normalized Kronos news intelligence record.
    """

    if published_time.tzinfo is None:
        published_time = published_time.replace(tzinfo=timezone.utc)

    return {
        "headline": headline,
        "source": source,

        "published_time_utc": published_time.astimezone(
            timezone.utc
        ).isoformat(),

        "published_time_myt": to_myt(published_time).isoformat(),

        "category": category,

        "market_relevance": market_relevance,
        "btc_relevance": btc_relevance,

        "risk_level": risk_level,

        "directional_bias": directional_bias,

        "confidence": round(float(confidence), 3),

        "reason": reason,
    }


def validate_news_item(item):
    """
    Validate the minimum structure expected by Kronos.
    """

    required_fields = [
        "headline",
        "source",
        "published_time_utc",
        "published_time_myt",
        "category",
        "market_relevance",
        "btc_relevance",
        "risk_level",
        "directional_bias",
        "confidence",
        "reason",
    ]

    missing = [
        field
        for field in required_fields
        if field not in item
    ]

    if missing:
        raise ValueError(
            f"Missing required fields: {missing}"
        )

    if not 0.0 <= float(item["confidence"]) <= 1.0:
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    allowed_biases = {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
        "MIXED",
        "UNKNOWN",
    }

    if item["directional_bias"] not in allowed_biases:
        raise ValueError(
            f"Invalid directional_bias: "
            f"{item['directional_bias']}"
        )

    return True


def save_news_items(items):
    """
    Save normalized news intelligence.
    """

    for item in items:
        validate_news_item(item)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "generated_at_utc": utc_now().isoformat(),
        "generated_at_myt": to_myt(utc_now()).isoformat(),
        "count": len(items),
        "items": items,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved {len(items)} news items"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


def main():

    print("=" * 80)
    print("KRONOS WORLD NEWS INTELLIGENCE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Temporary test record.
    #
    # This is NOT real market news.
    # It only verifies that the schema and MYT conversion work.
    # -------------------------------------------------------------------------

    now = utc_now()

    test_item = create_news_item(
        headline="Kronos news intelligence schema test",
        source="TEST",
        published_time=now,
        category="SYSTEM_TEST",
        market_relevance="LOW",
        btc_relevance="LOW",
        risk_level="LOW",
        directional_bias="UNKNOWN",
        confidence=0.0,
        reason="Schema validation test only.",
    )

    save_news_items([test_item])

    print()
    print("TEST RECORD")
    print("-" * 80)

    print(
        json.dumps(
            test_item,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 80)
    print("NEWS INTELLIGENCE SCHEMA READY")
    print("=" * 80)


if __name__ == "__main__":
    main()