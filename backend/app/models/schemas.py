"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class StockPriceResponse(BaseModel):
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None


class SentimentResponse(BaseModel):
    label: str
    polarity: float
    subjectivity: float


class NewsArticleResponse(BaseModel):
    title: str
    source: str
    url: str
    published_at: Optional[datetime] = None
    sentiment: SentimentResponse
    reputation_score: float
    event_type: Optional[str] = None
    tfidf_top_terms: Optional[dict] = None


class ReputationScoreResponse(BaseModel):
    symbol: str
    overall_score: float
    sentiment_impact_score: float
    event_breakdown: list[dict]
    risk_level: str  # low, medium, high, critical


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predictions: list[dict]  # [{horizon, predicted_price, direction, confidence}]
    model_used: str
    generated_at: datetime


class TechnicalIndicatorsResponse(BaseModel):
    symbol: str
    date: datetime
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_mid: Optional[float] = None


class DashboardResponse(BaseModel):
    symbol: str
    company_name: str
    current_price: float
    price_history: list[dict]
    predictions: list[dict]
    sentiment_timeline: list[dict]
    reputation: ReputationScoreResponse
    technical_indicators: list[dict]
    news_articles: list[NewsArticleResponse]
    tfidf_important_terms: list[dict]


class StockSearchRequest(BaseModel):
    symbol: str
    days: int = 365


class TrainModelRequest(BaseModel):
    symbol: str
    model_type: str = "hybrid"  # random_forest, xgboost, lstm, hybrid
    days: int = 730
