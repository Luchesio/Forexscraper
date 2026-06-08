"""
DIY Currency Strength Meter
============================
Builds an 8-major currency strength meter from free ECB exchange rates.

Data source : https://api.frankfurter.dev   (no API key, no quotas)
Method       : average % change of each currency vs the other 7 over a
               lookback window -- exactly the "compare all 28 crosses" idea.

This is wired into the analysis pipeline as an ADDITIONAL data source. Instead
of printing to a terminal, `get_strength_report()` returns a compact text block
that is fed to Gemini alongside the scraped news, so the model can cross-check
its qualitative relative-strength read against an objective, price-derived one.

SINGLE-CALL DESIGN
------------------
Both the "now" and "prev" snapshots are pulled in ONE request using
Frankfurter's date-range endpoint:
    /v1/{start}..{end}?base=USD
This returns a `rates` object keyed by working-day date. We take the latest
date in the window as "now" and the most recent earlier date as "prev", so we
never have to guess which prior calendar day actually had an ECB fixing.

NOTE ON DATA FREQUENCY
----------------------
Frankfurter publishes ONE rate set per working day (ECB, ~16:00 CET). So the
"change" here is day-over-day, not live intraday behaviour. The maths is
identical -- feed it intraday snapshots instead of daily ones for true intraday.
"""

import asyncio
import json
import urllib.request
from datetime import date, timedelta
from typing import Optional

API = "https://api.frankfurter.dev/v1"
MAJORS = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"]

# Network timeout for the Frankfurter call (seconds). Kept short: this is a
# best-effort enrichment, never allowed to hold up the whole analysis.
HTTP_TIMEOUT = 10

# Frankfurter sits behind a CDN that rejects the default Python-urllib
# User-Agent with HTTP 403. Send a normal browser-like UA so the request is
# treated like any ordinary client.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# How many extra calendar days to pull behind the lookback target. This pads
# the window so weekends/holidays (no fixing) never leave us without a usable
# "prev" snapshot in the single response.
WINDOW_PADDING_DAYS = 5


def _fetch_window(start: str, end: str, base: str = "USD") -> dict[str, dict]:
    """Fetch all daily fixings between `start` and `end` in ONE request.

    Returns Frankfurter's `rates` mapping: {date_string: {currency: rate}}.
    Rates are quoted as 'units per 1 base'. The base itself is not included by
    the API, so callers must add it (= 1.0) before computing crosses.
    """
    symbols = ",".join(c for c in MAJORS if c != base)
    url = f"{API}/{start}..{end}?base={base}&symbols={symbols}"
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = json.load(resp)
    return data.get("rates", {})


def strength(now, prev):
    """Average % change of each currency against the other seven.

    Rates are quoted as 'X per 1 USD', so the cross 'units of D per 1 C'
    is rate[D] / rate[C]. A rising cross means C strengthened against D.
    """
    scores = {}
    for c in MAJORS:
        changes = []
        for d in MAJORS:
            if c == d:
                continue
            cross_now = now[d] / now[c]
            cross_prev = prev[d] / prev[c]
            changes.append((cross_now / cross_prev - 1) * 100)
        scores[c] = sum(changes) / len(changes)
    return scores


def _compute(lookback_days: int = 1, base: str = "USD"):
    """Returns (scores_dict, d_now, d_prev) from a SINGLE date-range request.

    Raises on network/data failure or if the window has fewer than two fixings.
    """
    today = date.today()
    start = (today - timedelta(days=lookback_days + WINDOW_PADDING_DAYS)).isoformat()
    end = today.isoformat()

    rates = _fetch_window(start, end, base=base)

    # Frankfurter omits the base currency from each row; add it back as 1.0
    # and keep only the eight majors we score on.
    snapshots: dict[str, dict] = {}
    for d, row in rates.items():
        row = dict(row)
        row[base] = 1.0
        if all(c in row for c in MAJORS):
            snapshots[d] = {c: row[c] for c in MAJORS}

    dates = sorted(snapshots)
    if len(dates) < 2:
        raise ValueError(
            f"Not enough fixings in window {start}..{end} "
            f"({len(dates)} usable day(s)); cannot compute a change."
        )

    d_now = dates[-1]
    # "prev" = the fixing closest to `lookback_days` before d_now, falling back
    # to the immediately preceding fixing if the exact target isn't available.
    now_date = date.fromisoformat(d_now)
    target = (now_date - timedelta(days=lookback_days)).isoformat()
    earlier = [d for d in dates if d < d_now]
    d_prev = max((d for d in earlier if d <= target), default=earlier[-1])

    return strength(snapshots[d_now], snapshots[d_prev]), d_now, d_prev


def _format_report(scores: dict, d_now: str, d_prev: str) -> str:
    """Render the scores as a plain-text block suitable for the LLM prompt."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    lines = [
        "OBJECTIVE CURRENCY STRENGTH METER (ECB daily fixings via Frankfurter)",
        f"Window: {d_prev}  ->  {d_now}",
        "Method: average % change of each currency vs the other 7 majors.",
        "Positive = strengthened over the window, negative = weakened.",
        "Ranking (strongest to weakest):",
    ]
    for i, (ccy, s) in enumerate(ranked, start=1):
        direction = "up" if s > 0 else ("down" if s < 0 else "flat")
        lines.append(f"  {i}. {ccy}  {s:+.2f}%  ({direction})")

    lines.append(f"Strongest: {ranked[0][0]}    Weakest: {ranked[-1][0]}")
    lines.append(f"Trend pair to watch: {ranked[0][0]}/{ranked[-1][0]}")
    return "\n".join(lines)


async def get_strength_report(lookback_days: int = 1) -> Optional[str]:
    """Async wrapper: compute the meter off-thread and return it as text.

    Returns None on any failure so the caller can degrade gracefully — this is
    enrichment, never a hard dependency for the analysis to run.
    `lookback_days`: 1 = daily strength, 7 = weekly, 30 = monthly.
    """
    loop = asyncio.get_event_loop()

    def _do() -> Optional[str]:
        try:
            scores, d_now, d_prev = _compute(lookback_days)
            return _format_report(scores, d_now, d_prev)
        except Exception as e:
            print(f"[CurrencyStrength] Failed to build meter: {type(e).__name__}: {e}")
            return None

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _do),
            timeout=HTTP_TIMEOUT + 5,
        )
    except asyncio.TimeoutError:
        print("[CurrencyStrength] Timed out building meter — skipping enrichment")
        return None
    except Exception as e:
        print(f"[CurrencyStrength] Unexpected error: {type(e).__name__}: {e}")
        return None