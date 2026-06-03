import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from services.scraper import scrape_all_sources, SCRAPER_TIMEOUT_SECONDS
from services.analyzer import stream_analysis, run_full_analysis
from services.schemas import (
    SessionFlow,
    MarketContext,
    CurrencyScore,
    HtfAlignmentItem,
    InstrumentBias,
    EconomicEvent,
)

router = APIRouter()

# Route-level safety net: SCRAPER_TIMEOUT_SECONDS (30 s) covers each source,
# plus a 5 s buffer for gather overhead. This should never fire in practice
# because per-source timeouts in scraper.py handle everything first.
ROUTE_SCRAPER_TIMEOUT = SCRAPER_TIMEOUT_SECONDS + 5


class AnalysisResponse(BaseModel):
    macro_narrative: str = ""
    news_impact: str = ""
    bias_summary: Optional[list[InstrumentBias]] = None
    fundamental_confidence: Optional[list[CurrencyScore]] = None
    session_flow: Optional[SessionFlow] = None
    trade_environment: Optional[MarketContext] = None
    htf_alignment: Optional[list[HtfAlignmentItem]] = None
    economic_events: Optional[list[EconomicEvent]] = None
    sources_scraped: list[str]
    from_cache: bool = False


async def _scrape_with_timeout() -> dict[str, str]:
    try:
        return await asyncio.wait_for(
            scrape_all_sources(),
            timeout=ROUTE_SCRAPER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(
            f"[Route] Scraper exceeded route safety-net ({ROUTE_SCRAPER_TIMEOUT}s). "
            "Per-source timeouts should have prevented this — check scraper logs."
        )
        return {}


@router.post(
    "/analyse/stream",
    summary="Stream analysis sections as they are generated (fast progressive UI)",
)
async def stream_analysis_endpoint():
    async def event_generator():
        scraped = await _scrape_with_timeout()

        sources = [k for k, v in scraped.items() if v]
        yield f"data: {json.dumps({'_sources': sources})}\n\n"

        if not sources:
            yield f"data: {json.dumps({'_error': 'All scrapers returned empty. Check your SERPER_API_KEY.'})}\n\n"
            return

        try:
            async for chunk in stream_analysis(scraped):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'_error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@router.post(
    "/analyse",
    response_model=AnalysisResponse,
    summary="Blocking full analysis (kept for compatibility)",
)
async def run_analysis():
    scraped = await _scrape_with_timeout()

    if not any(scraped.values()):
        raise HTTPException(status_code=503, detail="All scrapers returned empty.")

    try:
        results = run_full_analysis(scraped)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")

    return AnalysisResponse(
        macro_narrative=        results["macro_narrative"],
        news_impact=            results["news_impact"],
        bias_summary=           results["bias_summary"],
        fundamental_confidence= results["fundamental_confidence"],
        session_flow=           results["session_flow"],
        trade_environment=      results["trade_environment"],
        htf_alignment=          results["htf_alignment"],
        economic_events=        results["economic_events"],
        sources_scraped=        [k for k, v in scraped.items() if v],
    )