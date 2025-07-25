import os
from dotenv import load_dotenv
from pathlib import Path

# Resolve project root: 3 levels up from this file
ROOT_DIR = Path(__file__).resolve().parents[3]

# Load .env variables
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Runtime config
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"  # or your local path
SESSION_COOKIE = os.getenv("SESSION_COOKIE")
if not SESSION_COOKIE:
    raise ValueError("SESSION_COOKIE is not set in .env file.")

BASE_URL = "https://www.bauteileditor.de"
HEADERS = {"User-Agent": "Mozilla/5.0"}
