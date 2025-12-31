import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not GOOGLE_API_KEY or not FIRECRAWL_API_KEY:
    raise RuntimeError("Missing API keys in .env file")
