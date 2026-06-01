import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import AsyncGenerator
from google import genai
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL             = "gemini-3.5-flash"
MAX_RETRIES       = 4
BASE_DELAY        = 5
CACHE_TTL_SECONDS = 600

_cache: dict = {}

SECTION_DELIMITER = "═" * 40

MEGA_PROMPT = """You are a professional forex and financial market analyst.

TODAY'S DATE: {today}

RECENCY RULE: Only use news posted within the last 10 minutes. If a piece of news has no timestamp, include it. Skip any news explicitly timestamped older than 10 minutes. ALL output must be derived strictly from the scraped content below. Never guess.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Produce ALL seven sections below in a single response. Use EXACTLY the section delimiters shown — do not omit or rename any section. Output them in the order listed.

{DELIM}
SECTION: SESSION_FLOW
{DELIM}
Describe market flow for each session based only on the scraped news.
Respond in this exact format, nothing else:
Asia: <one concise sentence>
London: <one concise sentence>
NewYork: <one concise sentence>

{DELIM}
SECTION: TRADE_ENVIRONMENT
{DELIM}
Evaluate the current trade environment on three dimensions derived from the scraped news.

MARKET STRUCTURE — Purpose: Is price moving clearly or behaving randomly? This affects readability, POI respect, and continuation clarity. CLEAN means clear directional movement with structure respected; price is trending with identifiable highs and lows and the market shows purposeful directional intent. CHOPPY means contradictory signals, erratic price action, or random movement without respect for structure — price is moving without clear direction.

REACTION QUALITY — Measures how strongly price reacts from higher timeframe Points of Interest (HTF POIs), including higher timeframe Order Blocks (OB), higher timeframe Breaker Blocks (BB), HTF swing high liquidity sweeps, and HTF swing low liquidity sweeps. Evaluates whether these higher timeframe zones are currently producing strong, clean, and directional reactions that can be used for profitable lower timeframe confirmation entries. STRONG means strong rejection from HTF POIs with aggressive displacement after reacting from key levels, clear directional intent after liquidity sweeps, sustained movement away from HTF zones, and strong institutional reaction behaviour. MODERATE means some reaction from HTF POIs but not fully convincing — partial displacement with inconsistent follow-through. WEAK means hesitant or weak reactions from HTF POIs, shallow displacement after touching key levels, inconsistent directional movement, poor follow-through after liquidity sweeps, and price easily moving through HTF zones without meaningful reaction.

CONFIRMATION RELIABILITY — Measures how dependable lower timeframe market structure shift (LTF MSS) confirmation entries are after price reacts from higher timeframe points of interest (HTF POIs including Order Blocks, Breaker Blocks, and swing high/low liquidity sweeps). Evaluates whether current market conditions favour confirmation-based execution using HTF POI reactions and LTF MSS entries. HIGH means LTF MSS confirmations sustain direction cleanly with strong momentum, reduced fakeouts after MSS confirmation, cleaner alignment between HTF POI reactions and LTF execution, and high continuation probability after confirmation. MODERATE means some reliability with occasional inconsistencies in follow-through — confirmation entries work but with some reversals. LOW means MSS confirmations frequently fail, fake breakouts are common after confirmation, entries reverse shortly after triggering, weak continuation after MSS confirmation, and inconsistent alignment between HTF reactions and LTF confirmations.

Respond in this exact format, nothing else:
MarketStructure: CLEAN or CHOPPY
ReactionQuality: STRONG or MODERATE or WEAK
ConfirmationReliability: HIGH or MODERATE or LOW

{DELIM}
SECTION: FUNDAMENTAL_CONFIDENCE
{DELIM}
Score the FUNDAMENTAL CONFIDENCE for each currency (0.0-10.0) based on: directional clarity from the news, central bank stance clarity, rate differential strength, news-driven momentum. This score provides a quick conviction level, indicates whether fundamentals are aligned or mixed, and gives context for how aggressively to trade.

Scoring ranges:
- 8-10 = STRONG alignment (strong macro conviction; fundamentals clearly support one direction — trade aggressively in the bias direction)
- 6-7 = MODERATE tradable (fundamentals lean one way but some mixed signals present — trade with normal sizing)
- 5 or below = MIXED conditions (fundamentals unclear or conflicting — avoid or size down significantly)

Respond in this exact format, one currency per line, nothing else:
USD: score=X.X alignment=STRONG interpretation=<one concise sentence>
EUR: score=X.X alignment=MODERATE interpretation=<one concise sentence>
GBP: score=X.X alignment=MIXED interpretation=<one concise sentence>
JPY: score=X.X alignment=<STRONG|MODERATE|MIXED> interpretation=<one concise sentence>
AUD: score=X.X alignment=<STRONG|MODERATE|MIXED> interpretation=<one concise sentence>
NZD: score=X.X alignment=<STRONG|MODERATE|MIXED> interpretation=<one concise sentence>
CAD: score=X.X alignment=<STRONG|MODERATE|MIXED> interpretation=<one concise sentence>
CHF: score=X.X alignment=<STRONG|MODERATE|MIXED> interpretation=<one concise sentence>

{DELIM}
SECTION: HTF_ALIGNMENT
{DELIM}
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

{DELIM}
SECTION: BIAS_SUMMARY
{DELIM}
For every instrument listed below state: direction (bullish/bearish/neutral), strength (weak/moderate/strong), what is driving it, what must happen to invalidate it.
Include: GBPJPY, EURJPY, USDJPY, GBPUSD, EURUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP, EURAUD, EURNZD, EURCAD, EURCHF, GBPAUD, GBPNZD, GBPCAD, GBPCHF, AUDJPY, AUDNZD, AUDCAD, AUDCHF, NZDJPY, NZDCAD, NZDCHF, CADJPY, CADCHF, CHFJPY, BTCUSD, ETHUSD, ETHBTC, XAUUSD, XAGUSD, USOIL, UKOIL, NAS100, SPX500, US30, GER40, UK100, JP225.
Label each instrument clearly.

{DELIM}
SECTION: NEWS_IMPACT
{DELIM}
For each news item from the last 10 minutes: impact level (HIGH/MEDIUM/LOW), currency or asset affected, policy implication, short-term and medium-term effect, whether it strengthens or weakens the current bias.

{DELIM}
SECTION: MACRO_NARRATIVE
{DELIM}
PART 1 — MACRO NARRATIVE PER CURRENCY
For GBP, EUR, USD, AUD, NZD, CAD, CHF and JPY explain: interest rate outlook, inflation trend, central bank stance (hawkish/dovish), overall strength or weakness.

PART 2 — CURRENCY PAIRS FUNDAMENTAL OUTLOOK
For every pair explain: which currency is stronger and why, key drivers, bullish or bearish pressure.
Pairs: GBPJPY, EURJPY, USDJPY, GBPUSD, EURUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP, EURAUD, EURNZD, EURCAD, EURCHF, GBPAUD, GBPNZD, GBPCAD, GBPCHF, AUDJPY, AUDNZD, AUDCAD, AUDCHF, NZDJPY, NZDCAD, NZDCHF, CADJPY, CADCHF, CHFJPY

PART 3 — CRYPTO
For BTCUSD, ETHUSD, ETHBTC: which side is stronger, key drivers, directional pressure.

PART 4 — COMMODITIES
For XAUUSD, XAGUSD, USOIL, UKOIL: key drivers, directional pressure.

PART 5 — INDICES
For NAS100, SPX500, US30, GER40, UK100, JP225: key drivers, directional pressure.
"""

SECTION_KEYS = [
    "SESSION_FLOW",
    "TRADE_ENVIRONMENT",
    "FUNDAMENTAL_CONFIDENCE",
    "HTF_ALIGNMENT",
    "BIAS_SUMMARY",
    "NEWS_IMPACT",
    "MACRO_NARRATIVE",
]

KEY_MAP = {
    "SESSION_FLOW":           "session_flow",
    "TRADE_ENVIRONMENT":      "trade_environment",
    "FUNDAMENTAL_CONFIDENCE": "fundamental_confidence",
    "HTF_ALIGNMENT":          "htf_alignment",
    "BIAS_SUMMARY":           "bias_summary",
    "NEWS_IMPACT":            "news_impact",
    "MACRO_NARRATIVE":        "macro_narrative",
}

_SECTION_RE = re.compile(
    r"═+\s*\nSECTION:\s*(" + "|".join(SECTION_KEYS) + r")\s*\n═+",
    re.IGNORECASE,
)


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
        "DELIM":             SECTION_DELIMITER,
    }


def _parse_sections(raw: str) -> dict[str, str]:
    splits = _SECTION_RE.split(raw)
    result = {v: "" for v in KEY_MAP.values()}
    i = 1
    while i + 1 < len(splits):
        key = KEY_MAP.get(splits[i].strip().upper())
        if key:
            result[key] = splits[i + 1].strip()
        i += 2
    return result


def _extract_retry_delay(error_str: str) -> float:
    m = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    return min(float(m.group(1)) + 2, 90) if m else BASE_DELAY


def _is_cache_valid(entry: dict) -> bool:
    return (time.time() - entry["timestamp"]) < CACHE_TTL_SECONDS


async def stream_analysis(scraped: dict[str, str]) -> AsyncGenerator[dict[str, str], None]:
    scrape_id = _scrape_hash(scraped)

    if scrape_id in _cache and _is_cache_valid(_cache[scrape_id]):
        age = int(time.time() - _cache[scrape_id]["timestamp"])
        print(f"[Cache] HIT (age {age}s) — streaming cached sections instantly")
        for raw_key, mapped_key in KEY_MAP.items():
            content = _cache[scrape_id]["data"].get(mapped_key, "")
            if content:
                yield {mapped_key: content}
                await asyncio.sleep(0)
        yield {"_meta": "cache_hit"}
        return

    print("[Cache] MISS — starting Gemini streaming...")
    ctx    = _build_context(scraped)
    prompt = MEGA_PROMPT.format(**ctx)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        buffer          = ""
        collected       = {v: "" for v in KEY_MAP.values()}
        current_key     = None
        current_content = []

        try:
            print(f"[Gemini] Stream attempt {attempt}/{MAX_RETRIES}...")
            stream = _client.models.generate_content_stream(model=MODEL, contents=prompt)

            for chunk in stream:
                if not chunk.text:
                    continue

                buffer += chunk.text

                while True:
                    m = _SECTION_RE.search(buffer)
                    if not m:
                        break

                    before      = buffer[:m.start()]
                    section_name = m.group(1).strip().upper()
                    buffer      = buffer[m.end():]

                    if current_key and before.strip():
                        content = before.strip()
                        collected[current_key] = content
                        yield {current_key: content}
                        await asyncio.sleep(0)

                    current_key     = KEY_MAP.get(section_name)
                    current_content = []

            if current_key and buffer.strip():
                content = buffer.strip()
                collected[current_key] = content
                yield {current_key: content}
                await asyncio.sleep(0)

            missing = [k for k, v in collected.items() if not v]
            if missing:
                print(f"[Parser] Warning — sections with no content: {missing}")

            _cache[scrape_id] = {"timestamp": time.time(), "data": collected}
            print(f"[Cache] Stored under hash {scrape_id[:12]}...")
            yield {"_meta": "done"}
            return

        except Exception as e:
            last_error = e
            err_str    = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _extract_retry_delay(err_str)
                print(f"[Gemini] 429 — waiting {wait:.0f}s...")
                yield {"_error": f"Rate limited. Retrying in {int(wait)}s... (attempt {attempt}/{MAX_RETRIES})"}
                await asyncio.sleep(wait)

            elif "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                wait = BASE_DELAY * attempt
                print(f"[Gemini] 503 — waiting {wait}s...")
                yield {"_error": f"Gemini overloaded. Retrying... (attempt {attempt}/{MAX_RETRIES})"}
                await asyncio.sleep(wait)

            else:
                yield {"_error": str(e)}
                raise

    yield {"_error": f"Gemini failed after {MAX_RETRIES} retries. Last error: {last_error}"}
    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries. Last error: {last_error}")


def run_full_analysis(scraped: dict[str, str]) -> dict[str, str]:
    scrape_id = _scrape_hash(scraped)

    if scrape_id in _cache and _is_cache_valid(_cache[scrape_id]):
        age = int(time.time() - _cache[scrape_id]["timestamp"])
        print(f"[Cache] HIT — returning cached result (age {age}s)")
        return _cache[scrape_id]["data"]

    print("[Cache] MISS — calling Gemini (blocking single call)...")
    ctx    = _build_context(scraped)
    prompt = MEGA_PROMPT.format(**ctx)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.models.generate_content(model=MODEL, contents=prompt)
            parsed   = _parse_sections(response.text)

            missing = [k for k, v in parsed.items() if not v]
            if missing:
                print(f"[Parser] Warning: {missing}")

            _cache[scrape_id] = {"timestamp": time.time(), "data": parsed}
            return parsed

        except Exception as e:
            last_error = e
            err_str    = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _extract_retry_delay(err_str)
                time.sleep(wait)
            elif "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                time.sleep(BASE_DELAY * attempt)
            else:
                raise

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries. Last error: {last_error}")


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