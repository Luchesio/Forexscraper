from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from services.scraper import scrape_all_sources
from services.analyzer import run_full_analysis

router = APIRouter()


class AnalysisResponse(BaseModel):
    macro_narrative: str
    news_impact: str
    bias_summary: str
    fundamental_confidence: str
    session_flow: str
    trade_environment: str
    htf_alignment: str
    sources_scraped: list[str]


class PartialAnalysisResponse(BaseModel):
    result: str
    section: str


@router.post("/analyse", response_model=AnalysisResponse, summary="Run full market analysis")
async def run_analysis():
    scraped = await scrape_all_sources()

    if not any(scraped.values()):
        raise HTTPException(
            status_code=503,
            detail="All scrapers returned empty. Check your SERPER_API_KEY and source URLs."
        )

    try:
        results = run_full_analysis(scraped)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")

    return AnalysisResponse(
        macro_narrative=results["macro_narrative"],
        news_impact=results["news_impact"],
        bias_summary=results["bias_summary"],
        fundamental_confidence=results["fundamental_confidence"],
        session_flow=results["session_flow"],
        trade_environment=results["trade_environment"],
        htf_alignment=results["htf_alignment"],
        sources_scraped=[k for k, v in scraped.items() if v],
    )


@router.post("/analyse/macro", response_model=PartialAnalysisResponse, summary="Macro narrative only")
async def run_macro_only():
    from services.analyzer import _build_context, _call_gemini, PROMPT_1_MACRO

    scraped = await scrape_all_sources()
    ctx = _build_context(scraped)

    try:
        result = _call_gemini(PROMPT_1_MACRO.format(**ctx))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PartialAnalysisResponse(result=result, section="macro_narrative")


@router.post("/analyse/news", response_model=PartialAnalysisResponse, summary="News impact only")
async def run_news_only():
    from services.analyzer import _build_context, _call_gemini, PROMPT_2_NEWS_IMPACT

    scraped = await scrape_all_sources()
    ctx = _build_context(scraped)

    try:
        result = _call_gemini(PROMPT_2_NEWS_IMPACT.format(**ctx))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PartialAnalysisResponse(result=result, section="news_impact")


@router.post("/analyse/bias", response_model=PartialAnalysisResponse, summary="Bias summary only")
async def run_bias_only():
    from services.analyzer import _build_context, _call_gemini, PROMPT_3_BIAS_SUMMARY

    scraped = await scrape_all_sources()
    ctx = _build_context(scraped)

    try:
        result = _call_gemini(PROMPT_3_BIAS_SUMMARY.format(**ctx))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PartialAnalysisResponse(result=result, section="bias_summary")