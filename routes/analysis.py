import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.scraper import scrape_all_sources
from services.analyzer import run_full_analysis, get_cache_status, invalidate_cache

router = APIRouter()

_analysis_lock = asyncio.Lock()


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


@router.post("/analyse", response_model=AnalysisResponse, summary="Run full market analysis (1 Gemini call)")
async def run_analysis():
    if _analysis_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Analysis already in progress. Please wait for the current run to finish."
        )

    async with _analysis_lock:
        scraped = await scrape_all_sources()

        if not any(scraped.values()):
            raise HTTPException(
                status_code=503,
                detail="All scrapers returned empty. Check your SERPER_API_KEY and source URLs."
            )

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


@router.delete("/analyse/cache", summary="Invalidate analysis cache (force fresh Gemini call on next run)")
async def clear_cache():
    invalidate_cache()
    return {"detail": "Cache cleared. Next analysis will call Gemini."}