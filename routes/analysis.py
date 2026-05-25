import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.scraper import scrape_all_sources
from services.analyzer import stream_analysis, run_full_analysis, get_cache_status, invalidate_cache

router = APIRouter()

_analysis_lock = asyncio.Lock()

SCRAPER_TIMEOUT = 45.0


class AnalysisResponse(BaseModel):
    macro_narrative: str
    news_impact: str
    bias_summary: str
    fundamental_confidence: str
    session_flow: str
    trade_environment: str
    htf_alignment: str
    sources_scraped: list[str]
    from_cache: bool = False


class CacheStatusResponse(BaseModel):
    entries: int
    valid_entries: int
    ttl_seconds: int
    ages_seconds: dict[str, int]


async def _scrape_with_timeout() -> dict[str, str]:
    try:
        return await asyncio.wait_for(scrape_all_sources(), timeout=SCRAPER_TIMEOUT)
    except asyncio.TimeoutError:
        print(f"[Scraper] Timed out after {SCRAPER_TIMEOUT}s — proceeding with partial data")
        return {}


@router.post("/analyse/stream", summary="Stream analysis sections as they are generated (fast progressive UI)")
async def stream_analysis_endpoint():
    if _analysis_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Analysis already in progress. Please wait for the current run to finish."
        )

    async def event_generator():
        async with _analysis_lock:
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
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@router.post("/analyse", response_model=AnalysisResponse, summary="Blocking full analysis (kept for compatibility)")
async def run_analysis():
    if _analysis_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Analysis already in progress. Please wait for the current run to finish."
        )

    async with _analysis_lock:
        scraped = await _scrape_with_timeout()

        if not any(scraped.values()):
            raise HTTPException(status_code=503, detail="All scrapers returned empty.")

        cached_before = get_cache_status()["valid_entries"]

        try:
            results = run_full_analysis(scraped)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")

        from_cache = get_cache_status()["valid_entries"] == cached_before

        return AnalysisResponse(
            macro_narrative=        results["macro_narrative"],
            news_impact=            results["news_impact"],
            bias_summary=           results["bias_summary"],
            fundamental_confidence= results["fundamental_confidence"],
            session_flow=           results["session_flow"],
            trade_environment=      results["trade_environment"],
            htf_alignment=          results["htf_alignment"],
            sources_scraped=        [k for k, v in scraped.items() if v],
            from_cache=             from_cache,
        )


@router.get("/analyse/cache", response_model=CacheStatusResponse, summary="Check cache status")
async def cache_status():
    return get_cache_status()


@router.delete("/analyse/cache", summary="Invalidate analysis cache")
async def clear_cache():
    invalidate_cache()
    return {"detail": "Cache cleared. Next analysis will call Gemini."}