import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, Any, Callable, Optional

from google import genai

from config import GEMINI_API_KEY
from services.schemas import (
    SessionFlow,
    MarketContext,
    CurrencyScore,
    HtfAlignmentItem,
    InstrumentBias,
    EconomicEvent,
)

_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL             = "gemini-3.5-flash"
MAX_RETRIES       = 4
BASE_DELAY        = 5

SECTION_DELIMITER = "═" * 40

# Currency / instrument universes — kept here so the prompt always asks for the
# exact set the frontend renders. (Single source of truth on the backend.)
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]

INSTRUMENTS = [
    "GBPJPY", "EURJPY", "USDJPY", "GBPUSD", "EURUSD", "AUDUSD", "NZDUSD", "USDCAD",
    "USDCHF", "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPAUD", "GBPNZD",
    "GBPCAD", "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF", "NZDJPY", "NZDCAD",
    "NZDCHF", "CADJPY", "CADCHF", "CHFJPY", "BTCUSD", "ETHUSD", "ETHBTC", "XAUUSD",
    "XAGUSD", "USOIL", "UKOIL", "NAS100", "SPX500", "US30", "GER40", "UK100", "JP225",
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
#
# Five sections now return STRICT JSON (machine-readable). Two stay prose.
# Examples are deliberately VARIED (bullish AND bearish, strong AND weak) so the
# model is not anchored toward a single direction — this is what makes the output
# trustworthy instead of biased.
# ─────────────────────────────────────────────────────────────────────────────
MEGA_PROMPT = """You are a professional forex and financial market analyst.

TODAY'S DATE: {today}

RULES:
- ALL output must be derived strictly from the scraped content below. Never invent data.
- Prefer the most recent news. If an item has no timestamp, you may still use it.
- For every JSON section: output ONLY valid JSON between the delimiters — no markdown code fences, no commentary, no trailing text. Use the EXACT keys shown.

--- TradingEconomics ---
{trading_economics}

--- InvestingLive ---
{investinglive}

--- LiveSquawk ---
{livesquawk}

--- Objective Currency Strength Meter (ECB price-derived) ---
{currency_strength}
This is an objective, price-derived relative-strength ranking computed from ECB daily fixings — not news commentary. Treat it as a quantitative cross-check: weigh it alongside the qualitative news above when ranking currencies in FUNDAMENTAL_CONFIDENCE. Where the news and this meter disagree, prefer the news for forward-looking/policy reasoning but note the meter reflects realised price movement over the window.

---

Produce ALL seven sections below in order. Use EXACTLY the section delimiters shown — do not omit or rename any section.

{DELIM}
SECTION: SESSION_FLOW
{DELIM}
Output a single JSON object describing market flow per session, derived only from the scraped news:
{{"asia": "<one concise sentence>", "london": "<one concise sentence>", "newYork": "<one concise sentence>"}}

{DELIM}
SECTION: TRADE_ENVIRONMENT
{DELIM}
Evaluate the current GLOBAL trade environment and risk sentiment from the scraped news.

marketStructure — CLEAN = clear directional movement, structure respected, purposeful intent. CHOPPY = contradictory/erratic price action with no clear direction.
reactionQuality — STRONG = aggressive displacement and clean rejections from HTF POIs (order blocks, breaker blocks, liquidity sweeps). MODERATE = partial/inconsistent reactions. WEAK = shallow reactions, price slicing through zones.
confirmationReliability — HIGH = LTF MSS confirmations follow through cleanly, few fakeouts. MODERATE = work with occasional reversals. LOW = frequent failed confirmations and fakeouts.
riskMode — ON = risk appetite / flows into risk assets. OFF = risk aversion / safe-haven demand. NEUTRAL = mixed.

Output a single JSON object:
{{"marketStructure": "CLEAN|CHOPPY", "reactionQuality": "STRONG|MODERATE|WEAK", "confirmationReliability": "HIGH|MODERATE|LOW", "riskMode": "ON|OFF|NEUTRAL", "riskDrivers": "<one concise sentence>"}}

{DELIM}
SECTION: FUNDAMENTAL_CONFIDENCE
{DELIM}
For EACH of these currencies output one JSON object: {currencies}.

IMPORTANT — `strength` is RELATIVE CURRENCY STRENGTH. Rank all eight currencies AGAINST EACH OTHER (do not assess any currency in isolation), strongest to weakest, then map each to a label from that single ranking. This is the forex "currency strength meter" concept: the goal is to identify which currencies are strong and which are weak relative to the rest of the board, so the strongest can be bought against the weakest.

Base the relative ranking on:
- interest-rate differentials between the central banks (higher / rising rates = relatively stronger),
- relative hawkish vs dovish positioning (who is tightening vs easing compared to the others),
- relative growth, inflation and data momentum,
- risk sentiment and safe-haven flows (risk-off tends to favour USD, JPY, CHF; risk-on tends to favour AUD, NZD, CAD).

Fields:
- score (0.0-10.0): your CONFIDENCE in this read — how clear and well-supported the signal is — NOT the strength itself.
- alignment: STRONG (score 8-10), MODERATE (6-7), MIXED (5 or below). Must match the score band.
- strength: the relative-ranking label, one of very-strong, strong, neutral, weak, very-weak. SPREAD the labels across the scale to reflect the ordering — broadly the top of the ranking is very-strong and the bottom is very-weak. Do NOT give most currencies the same label; the array as a whole must show a clear strongest-to-weakest ordering. Genuinely similar currencies may share a label, but there must be a distinguishable top and bottom.
- rank: this currency's exact position in the relative ranking — a UNIQUE integer from 1 (strongest) to 8 (weakest). Every currency must receive a DIFFERENT rank; no ties and no gaps. The rank must agree with strength (rank 1-2 ≈ very-strong/strong, rank 7-8 ≈ weak/very-weak).
- stance: hawkish, dovish, or neutral (central bank policy direction).
- driver: the present-tense reason it is moving right now (one short sentence).
- outlook: where it is heading next (one short sentence, forward-looking).

Keep HTF_ALIGNMENT and BIAS_SUMMARY consistent with this ranking: a pair built from a relatively strong base and a relatively weak quote should carry a directional bias in the corresponding direction (e.g. very-strong base vs very-weak quote = strong bias).

Output a JSON array. Example showing the VARIETY expected (use REAL values from the news, not these):
[{{"currency":"USD","score":8.4,"alignment":"STRONG","strength":"very-strong","rank":1,"stance":"hawkish","driver":"Hot CPI print reinforcing higher-for-longer rates","outlook":"Bias stays firm into next Fed meeting"}},
 {{"currency":"JPY","score":4.2,"alignment":"MIXED","strength":"weak","rank":8,"stance":"dovish","driver":"BoJ reaffirming accommodative policy","outlook":"Vulnerable unless intervention risk rises"}}]

{DELIM}
SECTION: HTF_ALIGNMENT
{DELIM}
For EACH instrument output one JSON object with daily (HTF fundamental direction) and intraday (current session momentum) bias, and the alignment between them: CONFIRMED (same non-neutral direction), CONFLICTED (opposing), NEUTRAL (either side neutral).
daily / intraday values: bullish, bearish, or neutral.
Instruments (output ALL of them): {instruments}

Output a JSON array. Example showing the VARIETY expected (do NOT copy — derive from the news):
[{{"symbol":"USDJPY","daily":"bullish","intraday":"bullish","alignment":"CONFIRMED"}},
 {{"symbol":"EURUSD","daily":"bearish","intraday":"bullish","alignment":"CONFLICTED"}},
 {{"symbol":"AUDNZD","daily":"neutral","intraday":"bearish","alignment":"NEUTRAL"}}]

{DELIM}
SECTION: BIAS_SUMMARY
{DELIM}
For EACH instrument output one JSON object: direction (bullish/bearish/neutral), strength (strong/moderate/weak), drivers (what is driving it), invalidation (what must happen to flip it).
Instruments (output ALL of them): {instruments}

Output a JSON array. Example showing the VARIETY expected (derive real values from the news):
[{{"symbol":"GBPUSD","direction":"bearish","strength":"strong","drivers":"UK retail sales miss plus firm dollar","invalidation":"Break and hold above last week's high"}},
 {{"symbol":"XAUUSD","direction":"bullish","strength":"moderate","drivers":"Safe-haven demand on geopolitical risk","invalidation":"Risk-on rotation and rising real yields"}}]

{DELIM}
SECTION: NEWS_IMPACT
{DELIM}
For each notable news item: impact level (HIGH/MEDIUM/LOW), currency or asset affected, policy implication, short-term and medium-term effect, and whether it strengthens or weakens the current bias. Write this as readable prose.

{DELIM}
SECTION: ECONOMIC_EVENTS
{DELIM}
Extract the scheduled or just-released economic events mentioned in the scraped news (rate decisions, CPI/inflation, jobs/NFP, GDP, PMI, central-bank speakers, etc.). Only include events actually referenced in the news — do NOT invent a standard calendar.
Fields per event:
- currency: the 3-letter currency most affected (e.g. USD, EUR, GBP, JPY).
- name: short event name (e.g. "US CPI", "FOMC Decision", "ECB Press Conference").
- timing: a short human-readable label exactly as implied by the news (e.g. "In ~90 min", "Today 13:30 GMT", "Tomorrow", "This week"). Use "" if unclear.
- impact: HIGH, MEDIUM or LOW.
- hoursUntil: your best estimate of hours until release as a number, or null if unknown or already released.

Output a JSON array (empty array [] if the news mentions no datable events). Example showing the VARIETY expected (derive real values from the news):
[{{"currency":"USD","name":"US CPI","timing":"Today 13:30 GMT","impact":"HIGH","hoursUntil":1.5}},
 {{"currency":"GBP","name":"BoE Decision","timing":"Tomorrow","impact":"HIGH","hoursUntil":21}},
 {{"currency":"EUR","name":"ECB Speakers","timing":"This week","impact":"MEDIUM","hoursUntil":null}}]

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

# Mapping from the section name in the prompt to the SSE key the frontend reads.
KEY_MAP = {
    "SESSION_FLOW":           "session_flow",
    "TRADE_ENVIRONMENT":      "trade_environment",
    "FUNDAMENTAL_CONFIDENCE": "fundamental_confidence",
    "HTF_ALIGNMENT":          "htf_alignment",
    "BIAS_SUMMARY":           "bias_summary",
    "NEWS_IMPACT":            "news_impact",
    "ECONOMIC_EVENTS":        "economic_events",
    "MACRO_NARRATIVE":        "macro_narrative",
}
SECTION_KEYS = list(KEY_MAP.keys())

_SECTION_RE = re.compile(
    r"═+\s*\nSECTION:\s*(" + "|".join(SECTION_KEYS) + r")\s*\n═+",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction + per-section validation
# ─────────────────────────────────────────────────────────────────────────────
def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json(text: str) -> Optional[Any]:
    """Tolerantly pull the first complete JSON value out of a section body."""
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass

    start = next((i for i, ch in enumerate(t) if ch in "[{"), None)
    if start is None:
        return None

    opener = t[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(t)):
        c = t[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : j + 1])
                except Exception:
                    return None
    return None


def _validate_session_flow(text: str) -> Optional[dict]:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    try:
        return SessionFlow(**data).model_dump()
    except Exception as e:
        print(f"[Parser] session_flow invalid: {e}")
        return None


def _validate_market_context(text: str) -> Optional[dict]:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    try:
        return MarketContext(**data).model_dump()
    except Exception as e:
        print(f"[Parser] trade_environment invalid: {e}")
        return None


def _validate_list(text: str, model, label: str) -> Optional[list]:
    data = _extract_json(text)
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            out.append(model(**item).model_dump())
        except Exception as e:
            print(f"[Parser] {label} item dropped: {e}")
    return out or None


_STRUCTURED_VALIDATORS: dict[str, Callable[[str], Optional[Any]]] = {
    "session_flow":           _validate_session_flow,
    "trade_environment":      _validate_market_context,
    "fundamental_confidence": lambda t: _validate_list(t, CurrencyScore, "fundamental_confidence"),
    "htf_alignment":          lambda t: _validate_list(t, HtfAlignmentItem, "htf_alignment"),
    "bias_summary":           lambda t: _validate_list(t, InstrumentBias, "bias_summary"),
    "economic_events":        lambda t: _validate_list(t, EconomicEvent, "economic_events"),
}


def _finalize_section(mapped_key: str, raw_text: str) -> Any:
    """Structured key -> validated JSON (or None on failure). Prose -> stripped text."""
    validator = _STRUCTURED_VALIDATORS.get(mapped_key)
    if validator:
        return validator(raw_text)
    return raw_text.strip()


def _empty_collected() -> dict[str, Any]:
    # Prose keys default to "", structured keys default to None.
    return {
        v: ("" if v in ("news_impact", "macro_narrative") else None)
        for v in KEY_MAP.values()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_context(scraped: dict[str, str]) -> dict:
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    return {
        "today":             today,
        "trading_economics": scraped.get("TradingEconomics")  or "No data available.",
        "investinglive":     scraped.get("InvestingLive")     or "No data available.",
        "livesquawk":        scraped.get("LiveSquawk")        or "No data available.",
        "currency_strength": scraped.get("CurrencyStrength")  or "No data available.",
        "DELIM":             SECTION_DELIMITER,
        "currencies":        ", ".join(CURRENCIES),
        "instruments":       ", ".join(INSTRUMENTS),
    }


def _parse_sections(raw: str) -> dict[str, Any]:
    """Blocking-path parser: split the full response, validate each section."""
    splits = _SECTION_RE.split(raw)
    result = _empty_collected()
    i = 1
    while i + 1 < len(splits):
        mapped_key = KEY_MAP.get(splits[i].strip().upper())
        if mapped_key:
            result[mapped_key] = _finalize_section(mapped_key, splits[i + 1])
        i += 2
    return result


def _extract_retry_delay(error_str: str) -> float:
    m = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    return min(float(m.group(1)) + 2, 90) if m else BASE_DELAY


# ─────────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────────
async def stream_analysis(scraped: dict[str, str]) -> AsyncGenerator[dict[str, Any], None]:
    print("[Gemini] Starting streaming analysis...")
    ctx    = _build_context(scraped)
    prompt = MEGA_PROMPT.format(**ctx)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        buffer      = ""
        collected   = _empty_collected()
        current_key = None

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

                    before       = buffer[: m.start()]
                    section_name = m.group(1).strip().upper()
                    buffer       = buffer[m.end():]

                    if current_key and before.strip():
                        content = _finalize_section(current_key, before)
                        if content:
                            collected[current_key] = content
                            yield {current_key: content}
                            await asyncio.sleep(0)

                    current_key = KEY_MAP.get(section_name)

            # Flush the final section (no trailing delimiter follows it)
            if current_key and buffer.strip():
                content = _finalize_section(current_key, buffer)
                if content:
                    collected[current_key] = content
                    yield {current_key: content}
                    await asyncio.sleep(0)

            missing = [k for k, v in collected.items() if not v]
            if missing:
                print(f"[Parser] Warning — sections with no content: {missing}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Blocking (kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────
def run_full_analysis(scraped: dict[str, str]) -> dict[str, Any]:
    print("[Gemini] Calling Gemini (blocking single call)...")
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

            return parsed

        except Exception as e:
            last_error = e
            err_str    = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                time.sleep(_extract_retry_delay(err_str))
            elif "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                time.sleep(BASE_DELAY * attempt)
            else:
                raise

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries. Last error: {last_error}")