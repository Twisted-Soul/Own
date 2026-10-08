import json
import os
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from openai import OpenAI


# =============================================================================
# KRONOS WORLD NEWS AI
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "news"
OUTPUT_FILE = OUTPUT_DIR / "kronos_news_intelligence.json"

MYT = ZoneInfo("Asia/Kuala_Lumpur")

MODEL = "gpt-6-luna"


def utc_now():
    return datetime.now(timezone.utc)


def to_myt(dt):
    return dt.astimezone(MYT)


def build_prompt():
    return """
Search current news from the last 24 hours.

Find only major events that could materially affect:
- Bitcoin
- crypto markets
- global risk sentiment
- US financial markets
- interest rates
- USD
- liquidity
- geopolitics
- crypto regulation

Prioritize Reuters, Bloomberg, CNBC, Financial Times, WSJ,
BBC, AP, official government agencies and central banks.

Ignore celebrity, sports, entertainment, SEO spam and generic
Bitcoin price articles.

Return at most 5 important events.

For each event return:

headline
source
published_time
category
market_relevance
btc_relevance
risk_level
directional_bias
confidence
reason

Allowed values:

market_relevance:
LOW, MEDIUM, HIGH, CRITICAL

btc_relevance:
LOW, MEDIUM, HIGH, CRITICAL

risk_level:
LOW, MEDIUM, HIGH, CRITICAL

directional_bias:
BULLISH, BEARISH, NEUTRAL, MIXED, UNKNOWN

confidence:
0.0 to 1.0

IMPORTANT:
directional_bias describes expected market impact only.
Do NOT give trading instructions.
Do NOT say BUY, SELL, LONG or SHORT.

Return ONLY valid JSON:

{
  "items": [
    {
      "headline": "...",
      "source": "...",
      "published_time": "...",
      "category": "...",
      "market_relevance": "...",
      "btc_relevance": "...",
      "risk_level": "...",
      "directional_bias": "...",
      "confidence": 0.0,
      "reason": "..."
    }
  ]
}
"""


def validate_item(item):

    required = [
        "headline",
        "source",
        "published_time",
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
        for field in required
        if field not in item
    ]

    if missing:
        raise ValueError(
            f"Missing fields: {missing}"
        )

    allowed = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    for field in [
        "market_relevance",
        "btc_relevance",
        "risk_level",
    ]:
        if item[field] not in allowed:
            raise ValueError(
                f"Invalid {field}: {item[field]}"
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

    confidence = float(
        item["confidence"]
    )

    if not 0 <= confidence <= 1:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )


def normalize_item(item):

    published = item["published_time"]

    try:
        dt = datetime.fromisoformat(
            published.replace(
                "Z",
                "+00:00"
            )
        )
    except Exception:
        dt = utc_now()

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    dt_utc = dt.astimezone(
        timezone.utc
    )

    dt_myt = to_myt(
        dt_utc
    )

    normalized = {
        "headline": str(
            item["headline"]
        ).strip(),

        "source": str(
            item["source"]
        ).strip(),

        "published_time_utc":
            dt_utc.isoformat(),

        "published_time_myt":
            dt_myt.isoformat(),

        "category": str(
            item["category"]
        ).strip(),

        "market_relevance":
            str(
                item["market_relevance"]
            ).upper(),

        "btc_relevance":
            str(
                item["btc_relevance"]
            ).upper(),

        "risk_level":
            str(
                item["risk_level"]
            ).upper(),

        "directional_bias":
            str(
                item["directional_bias"]
            ).upper(),

        "confidence":
            round(
                float(
                    item["confidence"]
                ),
                3
            ),

        "reason": str(
            item["reason"]
        ).strip(),
    }

    validate_item(
        normalized
    )

    return normalized


def search_news():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    client = OpenAI(
        api_key=api_key
    )

    print()
    print(
        "Searching current world news..."
    )

    response = client.responses.create(
        model=MODEL,

        tools=[
            {
                "type": "web_search"
            }
        ],

        input=build_prompt(),

        max_output_tokens=3000,
    )

    text = response.output_text.strip()

    if not text:
        raise RuntimeError(
            "Empty AI response."
        )

    try:
        payload = json.loads(
            text
        )

    except json.JSONDecodeError:

        print()
        print(
            "RAW AI RESPONSE"
        )
        print("-" * 80)
        print(text)
        print("-" * 80)

        raise RuntimeError(
            "AI response was not valid JSON."
        )

    if "items" not in payload:
        raise RuntimeError(
            "Missing 'items' in AI response."
        )

    return payload["items"]


def save_results(items):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    now = utc_now()

    payload = {
        "generated_at_utc":
            now.isoformat(),

        "generated_at_myt":
            to_myt(
                now
            ).isoformat(),

        "count":
            len(items),

        "items":
            items,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False
        )


def print_results(items):

    print()
    print("=" * 100)
    print(
        "KRONOS WORLD NEWS INTELLIGENCE"
    )
    print("=" * 100)

    for i, item in enumerate(
        items,
        1
    ):

        print()
        print(
            f"[{i}] {item['headline']}"
        )

        print(
            f"Source: {item['source']}"
        )

        print(
            f"Published MYT: "
            f"{item['published_time_myt']}"
        )

        print(
            f"Category: {item['category']}"
        )

        print(
            f"Market relevance: "
            f"{item['market_relevance']}"
        )

        print(
            f"BTC relevance: "
            f"{item['btc_relevance']}"
        )

        print(
            f"Risk: {item['risk_level']}"
        )

        print(
            f"Market bias: "
            f"{item['directional_bias']}"
        )

        print(
            f"Confidence: "
            f"{item['confidence']:.3f}"
        )

        print(
            f"Reason: {item['reason']}"
        )

        print("-" * 100)


def main():

    print("=" * 100)
    print(
        "KRONOS WORLD NEWS AI"
    )
    print("=" * 100)

    print(
        f"Model: {MODEL}"
    )

    now = utc_now()

    print(
        f"Time: "
        f"{to_myt(now).strftime('%Y-%m-%d %H:%M:%S')} MYT"
    )

    try:

        items = search_news()

    except Exception as exc:

        print()
        print(
            f"NEWS SEARCH FAILED: "
            f"{type(exc).__name__}"
        )

        print(
            str(exc)
        )

        raise

    normalized = []

    for item in items:

        try:

            normalized.append(
                normalize_item(item)
            )

        except Exception as exc:

            print(
                "WARNING: skipped invalid "
                f"item: {exc}"
            )

    if not normalized:
        raise RuntimeError(
            "No valid news items."
        )

    save_results(
        normalized
    )

    print_results(
        normalized
    )

    print()
    print("=" * 100)
    print(
        "NEWS INTELLIGENCE COMPLETE"
    )
    print("=" * 100)

    print(
        f"Items saved: "
        f"{len(normalized)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()