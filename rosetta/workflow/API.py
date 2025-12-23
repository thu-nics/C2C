"""API key placeholders loaded from .env for backward compatibility."""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
TONGXIAO_API_KEY = os.getenv("TONGXIAO_API_KEY", "")
METASO_API_KEY = os.getenv("METASO_API_KEY", "")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
