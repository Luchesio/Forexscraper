import time
from datetime import datetime, timezone
from google import genai
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.5-flash"
MAX_RETRIES = 5
RETRY_DELAY = 3


PROMPT_1_MACRO = """
You are a professional forex and financial market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
The sources below contain news with timestamps (e.g. "5th May 2026", "4 mins ago", "2 hours ago").
You must ONLY use news and data posted within the last 10 minutes.
Ignore anything older than 10 minutes — it is stale for forex trading purposes.
If a piece of news has no timestamp, include it in the analysis — only skip news that is explicitly timestamped older than 10 minutes.
Base your entire analysis strictly on news from the last 10 minutes only.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Based ONLY on the current news above, answer the following:

PART 1 — MACRO NARRATIVE PER CURRENCY
What is the current macroeconomic narrative for GBP, EUR, USD, AUD, NZD, CAD, CHF and JPY?
For each currency explain:
- Interest rate outlook
- Inflation trend
- Central bank stance (hawkish/dovish)
- Overall strength or weakness as of today

PART 2 — CURRENCY PAIRS FUNDAMENTAL OUTLOOK
Based on the current macro narrative, explain the fundamental outlook for every pair below.
For each pair include:
- Which currency is stronger and why
- The key drivers behind the pair
- Whether the pair has bullish or bearish pressure

Pairs:
GBPJPY, EURJPY, USDJPY, GBPUSD, EURUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP,
EURAUD, EURNZD, EURCAD, EURCHF, GBPAUD, GBPNZD, GBPCAD, GBPCHF,
AUDJPY, AUDNZD, AUDCAD, AUDCHF,
NZDJPY, NZDCAD, NZDCHF,
CADJPY, CADCHF, CHFJPY

PART 3 — CRYPTO
For: BTCUSD, ETHUSD, ETHBTC
Include:
- Which currency is stronger and why
- The key drivers behind the pair
- Whether the pair has bullish or bearish pressure

PART 4 — COMMODITIES
For: XAUUSD (Gold), XAGUSD (Silver), USOIL (WTI Crude Oil), UKOIL (Brent Crude Oil)
Include:
- Which currency is stronger and why
- The key drivers behind the pair
- Whether the pair has bullish or bearish pressure

PART 5 — INDICES
For: NAS100, SPX500 (S&P 500), US30, GER40 (DAX), UK100 (FTSE), JP225 (Nikkei)
Include:
- Which currency is stronger and why
- The key drivers behind the pair
- Whether the pair has bullish or bearish pressure

Keep it concise, clear and actionable. Present each currency/pair as a clearly labelled section.
"""

PROMPT_2_NEWS_IMPACT = """
You are a professional forex and financial market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
The sources below contain news with timestamps (e.g. "5th May 2026", "4 mins ago", "2 hours ago").
You must ONLY analyse news and data posted within the last 10 minutes.
Ignore anything older than 10 minutes — it is stale for forex trading purposes.
If a piece of news has no timestamp, include it in the analysis — only skip news that is explicitly timestamped older than 10 minutes.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

For each news item or data release from the last 10 minutes, explain:
1. Is this high, medium, or low impact on the market?
2. Which currency or asset is affected?
3. How does this change expectations for future policy or price direction?
4. What is the likely short-term and medium-term impact?
5. Does it strengthen or weaken the current bias?

Keep it practical and trading-focused. Label each news item clearly before analysing it.
"""

PROMPT_3_BIAS_SUMMARY = """
You are a professional forex and financial market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
The sources below contain news with timestamps (e.g. "5th May 2026", "4 mins ago", "2 hours ago").
You must ONLY use news and data posted within the last 10 minutes.
Ignore anything older than 10 minutes — it is stale for forex trading purposes.
If a piece of news has no timestamp, include it in the analysis — only skip news that is explicitly timestamped older than 10 minutes.
Build your bias summary strictly from news within the last 10 minutes only.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Summarise the current fundamental bias for every instrument below in a trading context.
For each present:
- Direction (bullish / bearish / neutral)
- Strength of bias (weak / moderate / strong)
- What is driving the strength
- What must happen to invalidate this bias

CURRENCY PAIRS:
GBPJPY, EURJPY, USDJPY, GBPUSD, EURUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP,
EURAUD, EURNZD, EURCAD, EURCHF, GBPAUD, GBPNZD, GBPCAD, GBPCHF,
AUDJPY, AUDNZD, AUDCAD, AUDCHF,
NZDJPY, NZDCAD, NZDCHF,
CADJPY, CADCHF, CHFJPY

CRYPTO — For BTCUSD, ETHUSD, ETHBTC also include:
- Current market sentiment (risk-on / risk-off)
- Relationship to macro conditions (rates, liquidity, USD strength)
- Key drivers (institutional demand, ETF flows, liquidity, etc.)

COMMODITIES — For XAUUSD, XAGUSD, USOIL, UKOIL also include:
- Macro drivers (inflation, USD strength, geopolitical risk, supply/demand)
- Safe-haven vs risk asset behaviour

INDICES — For NAS100, SPX500, US30, GER40, UK100, JP225 also include:
- Risk sentiment (risk-on / risk-off)
- Impact of interest rates and yields
- Economic growth expectations

Answers must be given for EVERY instrument listed above — not a general paragraph.
Label each instrument clearly.
"""

PROMPT_FUNDAMENTAL_CONFIDENCE = """
You are a professional forex market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
Only use news posted within the last 10 minutes. If no timestamp exists, include it.
Skip any news explicitly timestamped older than 10 minutes.
You MUST derive all scores strictly from the actual scraped news content below. Do not guess.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Based ONLY on the scraped news above, score the FUNDAMENTAL CONFIDENCE for each currency (0.0 to 10.0).
Score based on:
- How clear and aligned the directional macro bias is from the scraped news
- Central bank stance clarity (clear hawkish/dovish vs ambiguous)
- Rate differential strength vs other currencies visible in the news
- News and event driven directional momentum from the scraped data

Respond ONLY in this exact format, one currency per line, no other text:
USD: score=X.X interpretation=<one concise sentence about macro alignment from the news>
EUR: score=X.X interpretation=<one concise sentence>
GBP: score=X.X interpretation=<one concise sentence>
JPY: score=X.X interpretation=<one concise sentence>
AUD: score=X.X interpretation=<one concise sentence>
NZD: score=X.X interpretation=<one concise sentence>
CAD: score=X.X interpretation=<one concise sentence>
CHF: score=X.X interpretation=<one concise sentence>
"""

PROMPT_SESSION_FLOW = """
You are a professional forex market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
Only use news posted within the last 10 minutes. If no timestamp exists, include it.
Skip any news explicitly timestamped older than 10 minutes.
Derive session descriptions strictly from the scraped news content. Do not guess.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Based ONLY on the scraped news above, describe the market flow and dominant drivers for each trading session.

Respond ONLY in this exact format, no other text:
Asia: <one concise sentence about what drove markets during Asia session based on the news>
London: <one concise sentence about what drove markets during London session based on the news>
NewYork: <one concise sentence about current NY session focus or expected drivers based on the news>
"""

PROMPT_TRADE_ENVIRONMENT = """
You are a professional forex market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
Only use news posted within the last 10 minutes. If no timestamp exists, include it.
Skip any news explicitly timestamped older than 10 minutes.
All ratings must be derived strictly from the scraped news content. Do not guess.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Based ONLY on the scraped news above, evaluate the current trade environment across three dimensions.

MARKET STRUCTURE — Is price moving clearly or behaving randomly?
CLEAN = the news narrative shows clear directional movement, structure is respected, consistent directional signals.
CHOPPY = contradictory signals in the news, mixed conditions, unclear or erratic direction.

REACTION QUALITY — How strongly is price reacting from higher timeframe points of interest (HTF POIs)?
HTF POIs include: higher timeframe Order Blocks (OB), Breaker Blocks (BB), HTF swing high liquidity sweeps, HTF swing low liquidity sweeps.
STRONG = the news shows strong rejection from key levels, aggressive displacement, clear directional intent after liquidity sweeps, sustained institutional movement.
MODERATE = some reaction visible in the news but not convincing or consistent.
WEAK = the news shows weak or hesitant reactions, shallow displacement, price moving through key zones without meaningful reaction.

CONFIRMATION RELIABILITY — How dependable are LTF market structure shift (MSS) confirmation entries after HTF POI reactions?
HIGH = the news indicates LTF MSS confirmations sustaining direction cleanly, strong momentum continuation, reduced fakeouts, clean alignment between HTF reactions and LTF execution.
MODERATE = some reliability but with occasional inconsistencies visible in the news.
LOW = the news shows frequent reversals after confirmation signals, fake breakouts common, weak continuation after MSS confirmation.

Respond ONLY in this exact format, no other text:
MarketStructure: CLEAN or CHOPPY
ReactionQuality: STRONG or MODERATE or WEAK
ConfirmationReliability: HIGH or MODERATE or LOW
"""

PROMPT_HTF_ALIGNMENT = """
You are a professional forex market analyst.

TODAY'S DATE: {today}

IMPORTANT — RECENCY RULE:
Only use news posted within the last 10 minutes. If no timestamp exists, include it.
Skip any news explicitly timestamped older than 10 minutes.
Derive all alignment values strictly from the scraped news content. Do not guess.

--- TradingEconomics ---
{trading_economics}

--- FXStreet ---
{fxstreet}

--- DeltaOne (X) ---
{deltaone}

--- LiveSquawk ---
{livesquawk}

---

Based ONLY on the scraped fundamental news data above, for each instrument determine:
- daily: BULLISH, BEARISH, or NEUTRAL (higher timeframe fundamental direction derived from macro narrative in the news)
- intraday: BULLISH, BEARISH, or NEUTRAL (current session momentum derived from the news)
- alignment: CONFIRMED (both pointing same non-neutral direction), CONFLICTED (opposing directions), NEUTRAL (both neutral or unclear)

Respond ONLY in this exact format, one instrument per line, no other text:
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


def _build_context(scraped: dict[str, str]) -> dict:
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    return {
        "today": today,
        "trading_economics": scraped.get("TradingEconomics") or "No data available.",
        "fxstreet": scraped.get("FXStreet") or "No data available.",
        "deltaone": scraped.get("DeltaOne_X") or "No data available.",
        "livesquawk": scraped.get("LiveSquawk") or "No data available.",
    }


def _call_gemini(prompt: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Gemini] Attempt {attempt}/{MAX_RETRIES}...")
            response = _client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            print(f"[Gemini] Success on attempt {attempt}")
            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e)

            if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                wait = RETRY_DELAY * attempt
                print(f"[Gemini] 503 — waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries. Last error: {last_error}")


def run_full_analysis(scraped: dict[str, str]) -> dict[str, str]:
    ctx = _build_context(scraped)

    macro     = _call_gemini(PROMPT_1_MACRO.format(**ctx))
    news      = _call_gemini(PROMPT_2_NEWS_IMPACT.format(**ctx))
    bias      = _call_gemini(PROMPT_3_BIAS_SUMMARY.format(**ctx))
    fund_conf = _call_gemini(PROMPT_FUNDAMENTAL_CONFIDENCE.format(**ctx))
    session   = _call_gemini(PROMPT_SESSION_FLOW.format(**ctx))
    trade_env = _call_gemini(PROMPT_TRADE_ENVIRONMENT.format(**ctx))
    htf       = _call_gemini(PROMPT_HTF_ALIGNMENT.format(**ctx))

    return {
        "macro_narrative":        macro,
        "news_impact":            news,
        "bias_summary":           bias,
        "fundamental_confidence": fund_conf,
        "session_flow":           session,
        "trade_environment":      trade_env,
        "htf_alignment":          htf,
    }