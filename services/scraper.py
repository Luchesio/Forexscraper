import http.client
import json
import asyncio
from typing import Optional
from config import SERPER_API_KEY
from services.currency_strength import get_strength_report

# ── Timeout tuning ────────────────────────────────────────────────────────────
#
# REQUEST_TIMEOUT_SECONDS — hard socket-level timeout on the HTTPS connection.
# This fires *inside* the worker thread, so it genuinely cancels the blocking
# I/O (unlike asyncio.wait_for, which can't interrupt a running thread).
#
# Set generously so that legitimately slow sites (heavy news pages, Serper
# queue latency, large responses) are never dropped accidentally.
# Only truly dead or unreachable endpoints — ones that would never return
# anything useful — will hit this limit.
#
REQUEST_TIMEOUT_SECONDS = 25

# SCRAPER_TIMEOUT_SECONDS — asyncio.wait_for wrapper around each scrape_url
# call. Set a few seconds above REQUEST_TIMEOUT_SECONDS so the thread always
# gets a chance to exhaust its own socket timeout and return cleanly before the
# asyncio layer steps in. This is purely a safety net for edge cases where the
# socket timeout somehow doesn't fire (e.g. a very large partial read).
#
# Because all three scrapers run concurrently via asyncio.gather, the total
# wall-clock time for scrape_all_sources() == time of the SLOWEST single
# source, not the sum of all three. So even 25 s per source = max ~25 s total.
#
SCRAPER_TIMEOUT_SECONDS = 30


async def scrape_url(url: str) -> Optional[str]:
    """
    Scrapes a single URL via the Serper scrape API.
    Returns the extracted text, or None only if the request genuinely fails
    (network error, bad API key, HTTP error, or truly no content returned).

    Two-layer timeout — only fires for dead/unreachable sources:
      Layer 1: REQUEST_TIMEOUT_SECONDS on HTTPSConnection itself.
               Kills hanging TCP connections inside the worker thread.
      Layer 2: SCRAPER_TIMEOUT_SECONDS via asyncio.wait_for.
               Safety net so asyncio.gather never waits beyond this point.
    """
    loop = asyncio.get_event_loop()

    def _do_request() -> Optional[str]:
        try:
            conn = http.client.HTTPSConnection(
                "scrape.serper.dev",
                timeout=REQUEST_TIMEOUT_SECONDS,   # Layer 1
            )
            payload = json.dumps({"url": url})
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            }
            conn.request("POST", "/", payload, headers)
            res = conn.getresponse()

            # Non-2xx from Serper means the target couldn't be reached/scraped
            if res.status >= 400:
                print(f"[Scraper] HTTP {res.status} from Serper for {url}")
                conn.close()
                return None

            raw  = res.read()
            conn.close()
            data = json.loads(raw.decode("utf-8"))
            text = data.get("text", "").strip()

            # Return None only when there is genuinely no content
            return text if text else None

        except TimeoutError:
            print(f"[Scraper] Socket timeout ({REQUEST_TIMEOUT_SECONDS}s) for {url} — source unreachable")
            return None
        except Exception as e:
            print(f"[Scraper] Request failed for {url}: {type(e).__name__}: {e}")
            return None

    try:
        # Layer 2: async safety net
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=SCRAPER_TIMEOUT_SECONDS,
        )
        return result

    except asyncio.TimeoutError:
        # Only reaches here if the thread is STILL running after 30 s,
        # meaning the socket timeout (25 s) itself didn't fire — extremely rare.
        print(
            f"[Scraper] Async safety-net timeout ({SCRAPER_TIMEOUT_SECONDS}s) for {url} "
            "— source never responded even after socket timeout"
        )
        return None
    except Exception as e:
        print(f"[Scraper] Unexpected async error for {url}: {e}")
        return None


async def scrape_all_sources() -> dict[str, str]:
    """
    Scrapes all configured financial news sources concurrently.

    All three sources run in parallel via asyncio.gather, so:
      - A slow-but-working source DOES return its data (up to 25 s).
      - A truly dead/unreachable source gets cut off at 25 s.
      - Total wall-clock time = slowest single source, NOT sum of all three.
    """
    sources = {
        "TradingEconomics": "https://tradingeconomics.com/",
        "InvestingLive":    "https://investinglive.com/live-feed/",
        "LiveSquawk":       "https://www.livesquawk.com/latest-news",
    }

    # The objective currency-strength meter runs in parallel with the scrapers,
    # so it adds no extra wall-clock time. It's enrichment, not a hard source:
    # if it fails it simply isn't included.
    *results, strength_report = await asyncio.gather(
        *[scrape_url(url) for url in sources.values()],
        get_strength_report(lookback_days=1),
        return_exceptions=True,
    )

    scraped: dict[str, str] = {}
    for name, result in zip(sources.keys(), results):
        if isinstance(result, Exception) or result is None:
            print(f"[Scraper] ✗ {name} — no data returned")
            scraped[name] = ""
        else:
            scraped[name] = result
            print(f"[Scraper] ✓ {name} ({len(result):,} chars)")

    if isinstance(strength_report, Exception) or not strength_report:
        print("[Scraper] ✗ CurrencyStrength — no data returned")
        scraped["CurrencyStrength"] = ""
    else:
        scraped["CurrencyStrength"] = strength_report
        print(f"[Scraper] ✓ CurrencyStrength ({len(strength_report):,} chars)")

    loaded = sum(1 for v in scraped.values() if v)
    print(f"[Scraper] Done — {loaded}/{len(scraped)} sources returned data")
    return scraped