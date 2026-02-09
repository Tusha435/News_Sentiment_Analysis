"""SQLAlchemy database models for stock prediction platform."""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, Boolean, ForeignKey, JSON, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from backend.app.config import DATABASE_URL

Base = declarative_base()


class StockPrice(Base):
    """Historical OHLCV stock price data."""
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_stock_prices_symbol_date", "symbol", "date", unique=True),
    )


class CompanyFundamentals(Base):
    """Company financial statement data."""
    __tablename__ = "company_fundamentals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    report_type = Column(String(50))  # income_statement, balance_sheet, cash_flow
    fiscal_date = Column(DateTime)
    data = Column(JSON)  # Full financial data as JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsArticle(Base):
    """Stored news articles with sentiment data."""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    title = Column(String(500))
    source = Column(String(200))
    url = Column(Text)
    content = Column(Text)
    published_at = Column(DateTime)
    sentiment_label = Column(String(20))
    sentiment_polarity = Column(Float)
    sentiment_subjectivity = Column(Float)
    reputation_score = Column(Float)
    event_type = Column(String(100))
    tfidf_features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_news_symbol_date", "symbol", "published_at"),
    )


class Prediction(Base):
    """Model predictions stored for comparison with actuals."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    horizon = Column(String(20))  # 1d, 5d, 21d
    predicted_price = Column(Float)
    predicted_direction = Column(String(10))  # up, down, neutral
    confidence = Column(Float)
    actual_price = Column(Float, nullable=True)
    model_name = Column(String(50))
    features_used = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetadata(Base):
    """Metadata about trained ML models."""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    symbol = Column(String(10))
    model_type = Column(String(50))  # random_forest, xgboost, lstm, hybrid
    accuracy = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    training_samples = Column(Integer)
    feature_importance = Column(JSON)
    hyperparameters = Column(JSON)
    model_path = Column(Text)
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class SentimentScore(Base):
    """Aggregated sentiment scores per symbol per day."""
    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    avg_polarity = Column(Float)
    avg_subjectivity = Column(Float)
    positive_count = Column(Integer)
    negative_count = Column(Integer)
    neutral_count = Column(Integer)
    reputation_score = Column(Float)
    news_frequency = Column(Integer)
    event_volatility_index = Column(Float)

    __table_args__ = (
        Index("ix_sentiment_symbol_date", "symbol", "date", unique=True),
    )


# Database engine and session factory
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(engine)


def get_db():
    """Get database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
