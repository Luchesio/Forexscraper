import hashlib
import json
import re
import time
from datetime import datetime, timezone
from google import genai
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL         = "gemini-3.5-flash"
MAX_RETRIES   = 4
BASE_DELAY    = 5

_cache: dict = {}
CACHE_TTL_SECONDS = 600


MEGA_PROMPT = """
You are a professional forex and financial market analyst.

TODAY'S DATE: {today}

RECENCY RULE:
Only use news posted within the last 10 minutes.
If a piece of news has no timestamp, include it.
Skip any news explicitly timestamped older than 10 minutes.
ALL output must be derived strictly from the scraped content below. Never guess.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Produce ALL seven sections below in a single response.
Use EXACTLY the section delimiters shown — do not omit or rename any section.

════════════════════════════════════════
SECTION: MACRO_NARRATIVE
════════════════════════════════════════
PART 1 — MACRO NARRATIVE PER CURRENCY
Provide the current macroeconomic narrative for GBP, EUR, USD, AUD, NZD, CAD, CHF and JPY.
For each currency explain: interest rate outlook, inflation trend, central bank stance (hawkish/dovish), overall strength or weakness.

PART 2 — CURRENCY PAIRS FUNDAMENTAL OUTLOOK
For every pair below explain: which currency is stronger and why, key drivers, bullish or bearish pressure.
Pairs: GBPJPY, EURJPY, USDJPY, GBPUSD, EURUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP,
EURAUD, EURNZD, EURCAD, EURCHF, GBPAUD, GBPNZD, GBPCAD, GBPCHF,
AUDJPY, AUDNZD, AUDCAD, AUDCHF, NZDJPY, NZDCAD, NZDCHF, CADJPY, CADCHF, CHFJPY

PART 3 — CRYPTO
For BTCUSD, ETHUSD, ETHBTC: which side is stronger, key drivers, directional pressure.

PART 4 — COMMODITIES
For XAUUSD, XAGUSD, USOIL, UKOIL: key drivers, directional pressure.

PART 5 — INDICES
For NAS100, SPX500, US30, GER40, UK100, JP225: key drivers, directional pressure.

════════════════════════════════════════
SECTION: NEWS_IMPACT
════════════════════════════════════════
For each news item from the last 10 minutes: impact level (HIGH/MEDIUM/LOW), currency or asset affected, policy implication, short-term and medium-term effect, whether it strengthens or weakens the current bias.

════════════════════════════════════════
SECTION: BIAS_SUMMARY
════════════════════════════════════════
For every instrument listed below state: direction (bullish/bearish/neutral), strength (weak/moderate/strong), what is driving it, what must happen to invalidate it.
Include all pairs from Part 2 above plus BTCUSD, ETHUSD, ETHBTC, XAUUSD, XAGUSD, USOIL, UKOIL, NAS100, SPX500, US30, GER40, UK100, JP225.
Label each instrument clearly.

════════════════════════════════════════
SECTION: FUNDAMENTAL_CONFIDENCE
════════════════════════════════════════
Score the FUNDAMENTAL CONFIDENCE for each currency (0.0-10.0) based on: directional clarity from the news, central bank stance clarity, rate differential strength, news-driven momentum.
Respond in this exact format, one currency per line, nothing else:
USD: score=X.X interpretation=<one concise sentence>
EUR: score=X.X interpretation=<one concise sentence>
GBP: score=X.X interpretation=<one concise sentence>
JPY: score=X.X interpretation=<one concise sentence>
AUD: score=X.X interpretation=<one concise sentence>
NZD: score=X.X interpretation=<one concise sentence>
CAD: score=X.X interpretation=<one concise sentence>
CHF: score=X.X interpretation=<one concise sentence>

════════════════════════════════════════
SECTION: SESSION_FLOW
════════════════════════════════════════
Describe market flow for each session based only on the scraped news.
Respond in this exact format, nothing else:
Asia: <one concise sentence>
London: <one concise sentence>
NewYork: <one concise sentence>

════════════════════════════════════════
SECTION: TRADE_ENVIRONMENT
════════════════════════════════════════
Evaluate the current trade environment on three dimensions derived from the scraped news.

MARKET STRUCTURE — CLEAN means clear directional movement and structure respected. CHOPPY means contradictory signals or erratic direction.

REACTION QUALITY — STRONG means strong rejection from HTF POIs (Order Blocks, Breaker Blocks, swing high/low liquidity sweeps), aggressive displacement, clear directional intent, sustained institutional movement. MODERATE means some reaction but not convincing. WEAK means hesitant reactions, shallow displacement, price moving through key zones without meaningful reaction.

CONFIRMATION RELIABILITY — HIGH means LTF MSS confirmations sustain direction cleanly with strong momentum and reduced fakeouts. MODERATE means some reliability with occasional inconsistencies. LOW means frequent reversals after confirmation, fake breakouts common, weak continuation.

Respond in this exact format, nothing else:
MarketStructure: CLEAN or CHOPPY
ReactionQuality: STRONG or MODERATE or WEAK
ConfirmationReliability: HIGH or MODERATE or LOW

════════════════════════════════════════
SECTION: HTF_ALIGNMENT
════════════════════════════════════════
For each instrument determine daily (HTF fundamental direction from the news) and intraday (current session momentum from the news) bias, and whether they are CONFIRMED (same non-neutral direction), CONFLICTED (opposing), or NEUTRAL.
Respond in this exact format, one instrument per line, nothing else:
GBPJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
USDJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
GBPUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
AUDUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
NZDUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
USDCAD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
USDCHF: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURGBP: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURAUD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURNZD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURCAD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
EURCHF: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
GBPAUD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
GBPNZD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
GBPCAD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
GBPCHF: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
AUDJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
AUDNZD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
AUDCAD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
AUDCHF: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
NZDJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
NZDCAD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
NZDCHF: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
CADJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
CADCHF: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
CHFJPY: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
XAUUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
XAGUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
USOIL: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
UKOIL: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
BTCUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
ETHUSD: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
ETHBTC: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
NAS100: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
SPX500: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
US30: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
GER40: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
UK100: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
JP225: daily=BULLISH intraday=BULLISH alignment=CONFIRMED
"""

SECTION_KEYS = [
    "MACRO_NARRATIVE",
    "NEWS_IMPACT",
    "BIAS_SUMMARY",
    "FUNDAMENTAL_CONFIDENCE",
    "SESSION_FLOW",
    "TRADE_ENVIRONMENT",
    "HTF_ALIGNMENT",
]

KEY_MAP = {
    "MACRO_NARRATIVE":        "macro_narrative",
    "NEWS_IMPACT":            "news_impact",
    "BIAS_SUMMARY":           "bias_summary",
    "FUNDAMENTAL_CONFIDENCE": "fundamental_confidence",
    "SESSION_FLOW":           "session_flow",
    "TRADE_ENVIRONMENT":      "trade_environment",
    "HTF_ALIGNMENT":          "htf_alignment",
}


def _scrape_hash(scraped: dict[str, str]) -> str:
    combined = "".join(scraped.get(k) or "" for k in sorted(scraped))
    return hashlib.sha256(combined.encode()).hexdigest()


def _build_context(scraped: dict[str, str]) -> dict:
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    return {
        "today":             today,
        "trading_economics": scraped.get("TradingEconomics") or "No data available.",
        "fxstreet":          scraped.get("FXStreet")         or "No data available.",
        "deltaone":          scraped.get("DeltaOne_X")        or "No data available.",
        "livesquawk":        scraped.get("LiveSquawk")        or "No data available.",
    }


def _parse_sections(raw: str) -> dict[str, str]:
    pattern = re.compile(
        r"═+\s*\nSECTION:\s*(" + "|".join(SECTION_KEYS) + r")\s*\n═+",
        re.IGNORECASE,
    )

    splits = pattern.split(raw)
    result = {v: "" for v in KEY_MAP.values()}

    i = 1
    while i + 1 < len(splits):
        section_name = splits[i].strip().upper()
        content      = splits[i + 1].strip()
        mapped_key   = KEY_MAP.get(section_name)
        if mapped_key:
            result[mapped_key] = content
        i += 2

    return result


def _extract_retry_delay(error_str: str) -> float:
    m = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 2, 90)
    return BASE_DELAY


def _call_gemini_single(prompt: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Gemini] Attempt {attempt}/{MAX_RETRIES}...")
            response = _client.models.generate_content(model=MODEL, contents=prompt)
            print(f"[Gemini] Success on attempt {attempt}")
            return response.text

        except Exception as e:
            last_error = e
            err_str    = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _extract_retry_delay(err_str)
                print(f"[Gemini] 429 rate-limited — waiting {wait:.0f}s before retry {attempt + 1}...")
                time.sleep(wait)

            elif "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                wait = BASE_DELAY * attempt
                print(f"[Gemini] 503 overloaded — waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)

            else:
                raise

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries. Last error: {last_error}")


def _is_cache_valid(entry: dict) -> bool:
    return (time.time() - entry["timestamp"]) < CACHE_TTL_SECONDS


def run_full_analysis(scraped: dict[str, str]) -> dict[str, str]:
    scrape_id = _scrape_hash(scraped)

    if scrape_id in _cache and _is_cache_valid(_cache[scrape_id]):
        age = int(time.time() - _cache[scrape_id]["timestamp"])
        print(f"[Cache] HIT — returning cached result (age {age}s, TTL {CACHE_TTL_SECONDS}s)")
        return _cache[scrape_id]["data"]

    print("[Cache] MISS — calling Gemini (1 request for all 7 sections)...")

    ctx    = _build_context(scraped)
    prompt = MEGA_PROMPT.format(**ctx)
    raw    = _call_gemini_single(prompt)

    parsed = _parse_sections(raw)

    missing = [k for k, v in parsed.items() if not v]
    if missing:
        print(f"[Parser] Warning: sections with no content: {missing}")

    _cache[scrape_id] = {"timestamp": time.time(), "data": parsed}
    print(f"[Cache] Stored result under hash {scrape_id[:12]}...")

    return parsed


def get_cache_status() -> dict:
    now   = time.time()
    valid = [k for k, v in _cache.items() if _is_cache_valid(v)]
    return {
        "entries":       len(_cache),
        "valid_entries": len(valid),
        "ttl_seconds":   CACHE_TTL_SECONDS,
        "ages_seconds":  {k[:12]: int(now - v["timestamp"]) for k, v in _cache.items()},
    }


def invalidate_cache() -> None:
    _cache.clear()
    print("[Cache] Manually cleared.")