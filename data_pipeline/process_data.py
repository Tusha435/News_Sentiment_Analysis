"""Data processing pipeline.

Processes collected raw data into features:
1. Compute technical indicators on price data
2. Run sentiment analysis on stored news articles
3. Build daily aggregated feature vectors
4. Store processed features for model training

Usage:
    python -m data_pipeline.process_data --symbol AAPL
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

import pandas as pd

from backend.app.models.database import (
    init_db, SessionLocal, StockPrice, NewsArticle, SentimentScore
)
from backend.app.services.sentiment import sentiment_analyzer
from backend.app.services.tfidf_engine import tfidf_engine
from backend.app.services.reputation import reputation_scorer
from backend.app.services.technical import technical_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_sentiment(symbol: str):
    """Process sentiment for all unscored articles."""
    db = SessionLocal()

    try:
        articles = db.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.sentiment_label.is_(None)
        ).all()

        if not articles:
            logger.info("No unprocessed articles found")
            return

        logger.info(f"Processing sentiment for {len(articles)} articles...")

        contents = [a.content or "" for a in articles]
        tfidf_engine.fit(contents)

        for article in articles:
            content = article.content or ""

            # Sentiment analysis
            sentiment = sentiment_analyzer.analyze_ensemble(content)
            article.sentiment_label = sentiment["label"]
            article.sentiment_polarity = sentiment["polarity"]
            article.sentiment_subjectivity = sentiment["subjectivity"]

            # TF-IDF features
            tfidf_features = tfidf_engine.get_feature_vector(content)
            article.tfidf_features = tfidf_features

            # Reputation scoring
            article_dict = {
                "content": content,
                "source": article.source or "",
                "published_at": article.published_at,
                "sentiment": sentiment,
            }
            rep = reputation_scorer.score_article(article_dict)
            article.reputation_score = rep["sentiment_impact_score"]
            article.event_type = rep["event_type"]

        db.commit()
        logger.info(f"Processed sentiment for {len(articles)} articles")

    finally:
        db.close()


def aggregate_daily_sentiment(symbol: str):
    """Aggregate article-level sentiment into daily scores."""
    db = SessionLocal()

    try:
        articles = db.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.sentiment_label.isnot(None)
        ).all()

        if not articles:
            logger.info("No scored articles found")
            return

        # Group by date
        daily_data = {}
        for article in articles:
            if not article.published_at:
                continue
            date_key = article.published_at.date()
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append(article)

        for date_key, day_articles in daily_data.items():
            existing = db.query(SentimentScore).filter(
                SentimentScore.symbol == symbol,
                SentimentScore.date == date_key
            ).first()

            polarities = [a.sentiment_polarity or 0 for a in day_articles]
            subjectivities = [a.sentiment_subjectivity or 0 for a in day_articles]
            labels = [a.sentiment_label for a in day_articles]
            rep_scores = [a.reputation_score or 0 for a in day_articles]

            n = len(day_articles)
            avg_pol = sum(polarities) / n
            avg_sub = sum(subjectivities) / n
            avg_rep = sum(rep_scores) / n

            if existing:
                existing.avg_polarity = avg_pol
                existing.avg_subjectivity = avg_sub
                existing.positive_count = labels.count("Positive")
                existing.negative_count = labels.count("Negative")
                existing.neutral_count = labels.count("Neutral")
                existing.reputation_score = avg_rep
                existing.news_frequency = n
            else:
                score = SentimentScore(
                    symbol=symbol,
                    date=date_key,
                    avg_polarity=avg_pol,
                    avg_subjectivity=avg_sub,
                    positive_count=labels.count("Positive"),
                    negative_count=labels.count("Negative"),
                    neutral_count=labels.count("Neutral"),
                    reputation_score=avg_rep,
                    news_frequency=n,
                    event_volatility_index=0,
                )
                db.add(score)

        db.commit()
        logger.info(f"Aggregated sentiment for {len(daily_data)} days")

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Process collected data")
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    args = parser.parse_args()

    init_db()
    symbol = args.symbol.upper()

    process_sentiment(symbol)
    aggregate_daily_sentiment(symbol)

    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
