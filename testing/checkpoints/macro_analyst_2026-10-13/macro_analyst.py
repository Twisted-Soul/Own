"""
macro_analyst.py - multi-report macro data retriever + analyst (NFP, CPI, PPI, retail sales).

Each report is just a config entry in REPORTS: which series to fetch, how to
transform them, and how much each metric matters. Adding a new report means
adding one dict entry, not new code.

Usage
  pip install requests
  set BLS_API_KEY=...    (free, bls.gov; strongly recommended)
  set FRED_API_KEY=...   (needed for retail sales, or any --source fred run)

  python macro_analyst.py --report nfp
  python macro_analyst.py --report cpi --watch     # poll for the release, then analyse
  python macro_analyst.py --report ppi
  python macro_analyst.py --report retail
  python macro_analyst.py --summary                # combine the latest reading of every report

consensus.json (fill in the day before each release; only keys you provide are scored):
  {
    "nfp":    {"payrolls_k": 150, "unemployment": 4.3, "ahe_mom": 0.3},
    "cpi":    {"cpi_mom": 0.3, "core_cpi_mom": 0.3, "cpi_yoy": 3.0, "core_cpi_yoy": 3.1},
    "ppi":    {"ppi_mom": 0.2, "core_ppi_mom": 0.2},
    "retail": {"retail_sales_mom": 0.4}
  }

Import use:
  from macro_analyst import analyze_release, macro_summary
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

BLS_V1 = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SNAPSHOT_FILE = Path("macro_snapshots.json")
RELEASE_CALENDAR_FILE = Path("release_calendar.json")
TONE_THRESHOLD = 0.4
SUMMARY_MAX_AGE_DAYS = 35

# Transform kinds: level = latest value, diff = change vs prior month,
# mom = % change vs prior month, yoy = % change vs 12 months ago.
# Each metric: name -> (series_name, transform, weight, scale)
#   weight: + means "above consensus = hawkish"; - means the opposite.
#   scale : size of a 1-sigma surprise. THESE ARE STARTING GUESSES, calibrate them.
# Series ids: (bls_id, fred_id); use None if the source doesn't carry it.
REPORTS = {
    "nfp": {
        "series": {
            "payrolls":     ("CES0000000001", "PAYEMS"),
            "unemployment": ("LNS14000000",   "UNRATE"),
            "earnings":     ("CES0500000003", "CES0500000003"),
        },
        "metrics": {
            "payrolls_k":   ("payrolls",     "diff",  0.4,  75.0),
            "unemployment": ("unemployment", "level", -0.2, 0.1),
            "ahe_mom":      ("earnings",     "mom",   0.3,  0.1),
        },
        "revision": ("payrolls_k", 0.1, 50.0),  # net revision to prior 2 months
    },

    "cpi": {
        "series": {
            # Seasonally adjusted: use for monthly inflation.
            "headline_sa": ("CUSR0000SA0", None),
            "core_sa": ("CUSR0000SA0L1E", None),

            # Not seasonally adjusted: use for annual inflation.
            "headline_nsa": ("CUUR0000SA0", None),
            "core_nsa": ("CUUR0000SA0L1E", None),
        },
        "metrics": {
            "core_cpi_mom": ("core_sa", "mom", 0.4, 0.1),
            "cpi_mom": ("headline_sa", "mom", 0.3, 0.1),
            "core_cpi_yoy": ("core_nsa", "yoy", 0.2, 0.1),
            "cpi_yoy": ("headline_nsa", "yoy", 0.1, 0.1),
        },
    },

    "ppi": {
        "series": {
            "headline": ("WPSFD4",     "PPIFIS"),
            "core":     ("WPSFD49104", "PPIFES"),
        },
        "metrics": {
            "core_ppi_mom": ("core",     "mom", 0.5, 0.2),
            "ppi_mom":      ("headline", "mom", 0.5, 0.2),
        },
    },
    "retail": {  # Census data, so FRED only
        "series": {"sales": (None, "RSAFS")},
        "metrics": {"retail_sales_mom": ("sales", "mom", 1.0, 0.5)},
    },
}

# ---------------------------------------------------------------- helper

def validate_release_metadata(report, period, released_at_utc, source_url):
    """Validate release metadata, not the contents of the source page."""
    if report not in REPORTS:
        return False

    if not isinstance(period, str) or len(period) != 7:
        return False

    try:
        year, month = period.split("-")
        if len(year) != 4 or not year.isdigit():
            return False
        if not month.isdigit() or not 1 <= int(month) <= 12:
            return False
    except (ValueError, AttributeError):
        return False

    if not isinstance(released_at_utc, str) or not source_url:
        return False

    try:
        release_dt = datetime.fromisoformat(released_at_utc)
        if release_dt.tzinfo is None:
            return False
        if release_dt > datetime.now(timezone.utc):
            return False

        parsed = urlparse(source_url)
        if parsed.scheme != "https":
            return False
        if parsed.hostname not in {"bls.gov", "www.bls.gov"}:
            return False
        if not parsed.path.startswith("/news.release/"):
            return False

    except (TypeError, ValueError):
        return False

    return True

def classify_catalyst(released_at_utc, release_verified):
    """
    Classify release freshness for testing purposes.
    This is not a BTC trade signal.
    """
    if not release_verified or not released_at_utc:
        return {
            "catalyst_status": "UNKNOWN",
            "catalyst_age_hours": None,
        }

    try:
        release_dt = datetime.fromisoformat(released_at_utc)

        if release_dt.tzinfo is None:
            return {
                "catalyst_status": "UNKNOWN",
                "catalyst_age_hours": None,
            }

        age_hours = (
            datetime.now(timezone.utc) - release_dt
        ).total_seconds() / 3600

        if age_hours < 0:
            status = "FUTURE"
        elif age_hours <= 6:
            status = "FRESH"
        elif age_hours <= 24:
            status = "RECENT"
        else:
            status = "STALE"

        return {
            "catalyst_status": status,
            "catalyst_age_hours": round(age_hours, 2),
        }

    except (TypeError, ValueError):
        return {
            "catalyst_status": "UNKNOWN",
            "catalyst_age_hours": None,
        }


def is_usable_snapshot(
    retrieval_recent,
    release_verified,
    release_age_days,
    composite,
    max_age_days=SUMMARY_MAX_AGE_DAYS,
):
    """Determine whether a report can contribute to macro bias."""
    return (
        retrieval_recent
        and release_verified
        and release_age_days is not None
        and 0 <= release_age_days <= max_age_days
        and composite is not None
    )


# ---------------------------------------------------------------- retrieval

def fetch_bls(ids, years=3):
    key = os.getenv("BLS_API_KEY")
    now = datetime.now(timezone.utc).year

    body = {
        "seriesid": ids,
        "startyear": str(now - years + 1),
        "endyear": str(now),
    }
    if key:
        body["registrationkey"] = key

    response = requests.post(
        BLS_V2 if key else BLS_V1,
        json=body,
        timeout=15,
    )
    response.raise_for_status()
    js = response.json()

    if js.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {js.get('message')}")

    out = {}
    for series in js["Results"]["series"]:
        rows = []

        for item in series["data"]:
            period = item["period"]
            value = item["value"]

            if not period.startswith("M") or period == "M13":
                continue
            if value == "-":
                continue

            month = int(period[1:])
            if not 1 <= month <= 12:
                continue

            # Always use zero-padded YYYY-MM.
            date_key = f"{int(item['year']):04d}-{month:02d}"
            rows.append((date_key, float(value)))

        # Sort chronologically, newest first.
        rows.sort(key=lambda row: row[0], reverse=True)

        # Reject duplicate months rather than silently using
        # ambiguous observations.
        dates = [row[0] for row in rows]
        if len(dates) != len(set(dates)):
            raise RuntimeError(
                f"Duplicate monthly observations in {series['seriesID']}"
            )

        out[series["seriesID"]] = rows

    return out


def fetch_fred(series_id, n=40):
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set")
    r = requests.get(
        FRED_URL,
        params={"series_id": series_id, "api_key": key, "file_type": "json",
                "sort_order": "desc", "limit": n},
        timeout=10,
    )
    r.raise_for_status()
    return [(o["date"][:7], float(o["value"]))
            for o in r.json()["observations"] if o["value"] != "."]


def pick_source(report, source):
    series = REPORTS[report]["series"]
    if source == "auto":
        return "bls" if all(b for b, _ in series.values()) else "fred"
    if source == "bls" and not all(b for b, _ in series.values()):
        raise RuntimeError(f"'{report}' is not available from BLS; use --source fred")
    return source


def fetch_report(report, source="auto"):
    """Return (source_used, {series_name: [(YYYY-MM, value), ...] newest first})."""
    source = pick_source(report, source)
    series = REPORTS[report]["series"]
    data = {}
    if source == "bls":
        got = fetch_bls([b for b, _ in series.values()])
        for name, (bls_id, _) in series.items():
            if not got.get(bls_id):
                raise RuntimeError(f"BLS returned no data for {bls_id} ({name}); check the series id")
            data[name] = got[bls_id]
    else:
        for name, (_, fred_id) in series.items():
            data[name] = fetch_fred(fred_id)
    return source, data


def wait_for_new_release(report, source, interval=3.0, timeout=900):
    """Poll until the newest month of the report changes."""
    first = next(iter(REPORTS[report]["series"]))
    used, base = fetch_report(report, source)
    baseline = base[first][0][0]
    print(f"Watching {report} via {used}; latest month = {baseline}", file=sys.stderr)
    if used == "bls" and not os.getenv("BLS_API_KEY"):
        print("WARNING: no BLS_API_KEY; the 25 requests/day limit will run out fast.",
              file=sys.stderr)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(interval)
        try:
            used, data = fetch_report(report, source)
        except Exception as exc:  # network blips are normal at release time
            print(f"fetch failed ({exc}); retrying", file=sys.stderr)
            continue
        if data[first][0][0] != baseline:
            return used, data
    raise TimeoutError("No new release detected before timeout")


# ----------------------------------------------------------------- analysis
def transform(rows, kind, shift=0):
    v = [x[1] for x in rows]
    try:
        a = v[shift]
        if kind == "level":
            return a
        if kind == "diff":
            return a - v[shift + 1]
        if kind == "mom":
            return (a / v[shift + 1] - 1) * 100
        if kind == "yoy":
            current_period, current_value = rows[shift]

            year, month = current_period.split("-")
            year_ago_period = f"{int(year) - 1}-{month}"

            values_by_period = {
                period: value for period, value in rows
            }

            year_ago_value = values_by_period.get(year_ago_period)

            if year_ago_value is None or year_ago_value == 0:
                return None

            return (current_value / year_ago_value - 1) * 100
    except IndexError:
        return None
    raise ValueError(kind)

def rnd(x):
    return None if x is None else round(x, 3)

def build_metrics(report, data):
    cfg = REPORTS[report]
    first = next(iter(cfg["series"]))
    out = {"period": data[first][0][0], "prior_period": data[first][1][0], "values": {}}
    for name, (series, kind, _, _) in cfg["metrics"].items():
        rows = data[series]
        out["values"][name] = {
            "value": rnd(transform(rows, kind, 0)),
            "prev1": rnd(transform(rows, kind, 1)),
            "prev2": rnd(transform(rows, kind, 2)),
        }
    return out

def load_json(path, default):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default

def compute_revision(report, metrics, history):
    """Net change to the previous two readings vs what we stored last release."""
    rev = REPORTS[report].get("revision")
    if not rev:
        return None
    name = rev[0]
    old = history.get(report, {}).get(metrics["prior_period"])
    if not old:
        return None
    o, n = old["metrics"]["values"][name], metrics["values"][name]
    if None in (o["value"], o["prev1"], n["prev1"], n["prev2"]):
        return None
    return round((n["prev1"] + n["prev2"]) - (o["value"] + o["prev1"]), 3)

def score(report, metrics, revision, consensus, released_at_utc=None):
    """Score only when consensus provenance and timing are validated."""
    cfg = REPORTS[report]
    meta = consensus.get("_meta", {}) if isinstance(consensus, dict) else {}

    # Consensus must be explicitly verified, sourced, and timestamped
    # before the economic release. Otherwise, no surprise score.
    if meta.get("status") != "verified":
        return {
            "composite": None,
            "tone": "unknown (consensus unverified)",
            "components": {},
            "consensus_status": "unverified",
        }

    source_url = meta.get("source_url")
    captured_at = meta.get("captured_at_utc")

    try:
        source = urlparse(source_url or "")
        captured_dt = datetime.fromisoformat(captured_at)
        release_dt = datetime.fromisoformat(released_at_utc or "")

        if (
            source.scheme != "https"
            or not source.hostname
            or captured_dt.tzinfo is None
            or release_dt.tzinfo is None
            or captured_dt >= release_dt
        ):
            raise ValueError("Consensus provenance or timing is invalid")

    except (TypeError, ValueError):
        return {
            "composite": None,
            "tone": "unknown (consensus provenance invalid)",
            "components": {},
            "consensus_status": "invalid",
        }

    items = []
    for name, (_, _, weight, scale) in cfg["metrics"].items():
        value = consensus.get(name)
        actual = metrics["values"].get(name, {}).get("value")

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and actual is not None
        ):
            items.append((name, actual - value, weight, scale))

    if cfg.get("revision") and revision is not None:
        _, weight, scale = cfg["revision"]
        items.append(("revision", revision, weight, scale))

    total = used = 0.0
    detail = {}

    for name, surprise, weight, scale in items:
        z = max(-3.0, min(3.0, surprise / scale))
        detail[name] = {
            "surprise": round(surprise, 3),
            "z": round(z, 2),
        }
        total += weight * z
        used += abs(weight)

    raw_composite = total / used if used else None

    if raw_composite is None:
        return {
            "composite": None,
            "tone": "unknown (no valid consensus values)",
            "components": detail,
            "consensus_status": "verified",
        }

    composite = round(raw_composite, 3)

    if raw_composite > TONE_THRESHOLD:
        tone = "hawkish (hot)"
    elif raw_composite < -TONE_THRESHOLD:
        tone = "dovish (cold)"
    else:
        tone = "mixed"

    return {
        "composite": composite,
        "tone": tone,
        "components": detail,
        "consensus_status": "verified",
    }

def analyze_release(report, source="auto", consensus_path="consensus.json", fetched=None):
    if report not in REPORTS:
        raise ValueError(f"Unknown report '{report}'. Options: {', '.join(REPORTS)}")

    used, data = fetched or fetch_report(report, source)
    metrics = build_metrics(report, data)

    calendar = load_json(RELEASE_CALENDAR_FILE, {})
    release = calendar.get(report, {}).get(metrics["period"], {})

    released_at_utc = release.get("released_at_utc")
    release_source_url = release.get("source_url")

    release_metadata_valid = validate_release_metadata(
        report=report,
        period=metrics["period"],
        released_at_utc=released_at_utc,
        source_url=release_source_url,
    )

    # Metadata validity alone does not prove the page contents were checked.
    source_checked = release.get("source_checked") is True
    release_verified = release_metadata_valid and source_checked

    history = load_json(SNAPSHOT_FILE, {})
    revision = compute_revision(report, metrics, history)

    result = {
        "report": report,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": used,
        "period": metrics["period"],
        "metrics": metrics["values"],
        "revision": revision,
        "analysis": score(
            report,
            metrics,
            revision,
            load_json(consensus_path, {}).get(report, {}),
            released_at_utc,
        ),
        "released_at_utc": released_at_utc,
        "release_source_url": release_source_url,
        "release_metadata_valid": release_metadata_valid,
        "source_checked": source_checked,
        "release_verified": release_verified,
    }

    history.setdefault(report, {})[metrics["period"]] = {
        "metrics": metrics,
        "analysis": result["analysis"],
        "retrieved_utc": result["retrieved_utc"],
        "released_at_utc": released_at_utc,
        "release_source_url": release_source_url,
        "release_metadata_valid": release_metadata_valid,
        "source_checked": source_checked,
        "release_verified": release_verified,
    }

    SNAPSHOT_FILE.write_text(json.dumps(history, indent=2))
    return result


def macro_summary():
    """Summarize the latest stored readings without confusing retrieval
    freshness with economic release freshness.
    """
    history = load_json(SNAPSHOT_FILE, {})
    rows, scores = {}, []

    for report, periods in history.items():
        if not periods:
            continue

        period = max(periods)
        latest = periods[period]
        comp = latest.get("analysis", {}).get("composite")

        # This only checks whether the snapshot was retrieved recently.
        # It does NOT confirm that the underlying release is recent.
        retrieved = latest.get("retrieved_utc")
        retrieval_age_days = None

        if retrieved:
            try:
                retrieved_dt = datetime.fromisoformat(retrieved)
                if retrieved_dt.tzinfo is None:
                    retrieved_dt = retrieved_dt.replace(
                        tzinfo=timezone.utc
                    )
                retrieval_age_days = (
                    datetime.now(timezone.utc) - retrieved_dt
                ).total_seconds() / 86400
            except (TypeError, ValueError):
                pass

        retrieval_recent = (
            retrieval_age_days is not None
            and 0 <= retrieval_age_days <= SUMMARY_MAX_AGE_DAYS
        )

        # Until actual release timestamps are stored and validated,
        # don't use these scores in the aggregate macro bias.
        released_at = latest.get("released_at_utc")
        release_verified = (
            latest.get("release_verified", False)
            and latest.get("source_checked", False)
            and latest.get("release_metadata_valid", False)
        )
        release_age_days = None

        if released_at and release_verified:
            try:
                release_dt = datetime.fromisoformat(released_at)

                if release_dt.tzinfo is not None:
                    release_age_days = (
                        datetime.now(timezone.utc) - release_dt
                    ).total_seconds() / 86400
            except (TypeError, ValueError):
                pass

        catalyst = classify_catalyst(
            released_at,
            release_verified,
        )


        usable = is_usable_snapshot(
            retrieval_recent=retrieval_recent,
            release_verified=release_verified,
            release_age_days=release_age_days,
            composite=comp,
        )


        rows[report] = {
            "period": period,
            "composite": comp,
            "tone": latest.get("analysis", {}).get(
                "tone", "unknown"
            ),
            "retrieval_age_days": (
                round(retrieval_age_days, 2)
                if retrieval_age_days is not None else None
            ),
            "retrieval_recent": retrieval_recent,
            "release_verified": release_verified,
            "used": usable,
            "release_age_days": (
                round(release_age_days, 2)
                if release_age_days is not None else None
            ),
            "catalyst_status": catalyst["catalyst_status"],
            "catalyst_age_hours": catalyst["catalyst_age_hours"],

        }

        if usable:
            scores.append(comp)

    avg = round(sum(scores) / len(scores), 2) if scores else None

    if avg is None:
        bias = "unknown (no release-verified reports)"
    elif avg > TONE_THRESHOLD:
        bias = (
            "hawkish (rates stay higher; typically a headwind "
            "for risk assets)"
        )
    elif avg < -TONE_THRESHOLD:
        bias = (
            "dovish (rate-cut hopes up; typically supportive, "
            "unless it signals recession)"
        )
    else:
        bias = "mixed"

    return {
        "reports": rows,
        "average_composite": avg,
        "macro_bias": bias,
    }


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Macro report retriever and analyst")
    ap.add_argument("--report", choices=list(REPORTS), default="nfp")
    ap.add_argument("--source", choices=["auto", "bls", "fred"], default="auto")
    ap.add_argument("--consensus", default="consensus.json")
    ap.add_argument("--watch", action="store_true", help="wait for a new release")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--summary", action="store_true", help="combine stored readings")
    args = ap.parse_args()

    if args.summary:
        print(json.dumps(macro_summary(), indent=2))
        return
    fetched = wait_for_new_release(args.report, args.source, args.interval, args.timeout) if args.watch else None
    print(json.dumps(analyze_release(args.report, args.source, args.consensus, fetched), indent=2))


# debugess
def debug_cpi_series():
    _, data = fetch_report("cpi", "bls")

    for name, rows in data.items():
        print(f"\n{name}:")
        for period, value in rows[:15]:
            print(f"  {period}: {value}")

        print("  12-month comparison:")
        if len(rows) > 12:
            print(f"  Latest: {rows[0]}")
            print(f"  Year ago: {rows[12]}")
            print(
                "  Calculated YoY:",
                round((rows[0][1] / rows[12][1] - 1) * 100, 3),
            )


if __name__ == "__main__":
    if "--debug-cpi" in sys.argv:
        debug_cpi_series()
    else:
        main()