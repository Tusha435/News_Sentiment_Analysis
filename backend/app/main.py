"""FastAPI application for Stock Prediction Platform.

Main entry point providing REST API for:
- Stock data retrieval with technical indicators
- News sentiment analysis
- Reputation risk scoring
- ML-based stock price predictions
- Full dashboard data aggregation
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models.database import init_db
from backend.app.routers import stocks, predictions, sentiment_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Stock Prediction Platform started")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Stock Prediction Intelligence Platform",
    description=(
        "Hybrid ML stock prediction system combining sentiment analysis, "
        "technical indicators, TF-IDF text features, and reputation risk scoring. "
        "Uses ensemble of XGBoost, Random Forest, and LSTM models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(stocks.router)
app.include_router(predictions.router)
app.include_router(sentiment_router.router)


@app.get("/")
async def root():
    """API health check and info."""
    return {
        "name": "Stock Prediction Intelligence Platform",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "stock_search": "/api/stocks/search?q=AAPL",
            "stock_prices": "/api/stocks/{symbol}/prices",
            "company_overview": "/api/stocks/{symbol}/overview",
            "sentiment_analysis": "/api/sentiment/{symbol}/analyze",
            "predictions": "/api/predictions/{symbol}",
            "dashboard": "/api/predictions/{symbol}/dashboard",
            "train_model": "POST /api/predictions/train",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
