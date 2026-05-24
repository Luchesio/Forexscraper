import os
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is missing from .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")