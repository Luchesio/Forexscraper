from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analysis import router as analysis_router

app = FastAPI(
    title="Market Fundamental Analyst API",
    description="Scrapes financial news and returns AI-powered macro analysis via Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "Market Analyst API is running"}