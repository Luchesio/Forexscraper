import http.client
import json
import asyncio
from typing import Optional
from config import SERPER_API_KEY


async def scrape_url(url: str) -> Optional[str]:
    """
    Scrapes a single URL using the Serper scrape API.
    Returns the extracted text content or None on failure.
    """
    loop = asyncio.get_event_loop()

    def _do_request():
        conn = http.client.HTTPSConnection("scrape.serper.dev")
        payload = json.dumps({"url": url})
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        raw = res.read()
        conn.close()
        return raw.decode("utf-8")

    try:
        raw_response = await loop.run_in_executor(None, _do_request)
        data = json.loads(raw_response)
        return data.get("text", "").strip()
    except Exception as e:
        print(f"[Scraper] Failed to scrape {url}: {e}")
        return None


async def scrape_all_sources() -> dict[str, str]:
    """
    Scrapes all configured financial news sources concurrently.
    Returns a dict mapping source name -> extracted text.
    """
    sources = {
        "TradingEconomics": "https://tradingeconomics.com/",
        "FXStreet": "https://www.fxstreet.com/news",
        "DeltaOne_X": "https://x.com/DeItaone",
        "LiveSquawk": "https://www.livesquawk.com/latest-news",
    }

    tasks = {
        name: scrape_url(url)
        for name, url in sources.items()
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    scraped: dict[str, str] = {}
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception) or result is None:
            print(f"[Scraper] No data for {name}")
            scraped[name] = ""
        else:
            scraped[name] = result

    return scraped