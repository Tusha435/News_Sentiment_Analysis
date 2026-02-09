"""Application configuration with environment variable support."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'stockprediction.db'}")

# Alpha Vantage settings
AV_BASE_URL = "https://www.alphavantage.co/query"
AV_RATE_LIMIT_DELAY = 12  # seconds between calls (free tier: 5/min)
AV_MAX_RETRIES = 3

# News API settings
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Model settings
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

# Cache settings
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 3600  # 1 hour

# ML settings
PREDICTION_HORIZON_DAYS = [1, 5, 21]  # next-day, weekly, monthly
TRAIN_TEST_SPLIT = 0.8
SEQUENCE_LENGTH = 60  # for LSTM lookback window
