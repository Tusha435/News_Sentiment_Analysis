"""Data collection pipeline for batch data ingestion.

Usage:
    python -m data_pipeline.collect_data --symbol AAPL --days 730

Collects:
- Historical price data from Alpha Vantage
- News articles from NewsAPI and RSS
- Company fundamentals
"""

import argparse
import logging
import sys
from datetime import datetime

sys.path.insert(0, ".")

from backend.app.services.alpha_vantage import av_client
from backend.app.services.news_service import news_collector
from backend.app.models.database import init_db, SessionLocal, StockPrice, NewsArticle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_prices(symbol: str, days: int = 730):
    """Fetch and store historical price data."""
    logger.info(f"Collecting price data for {symbol}...")
    df = av_client.get_daily_prices(symbol, outputsize="full")

    if df.empty:
        logger.error("No price data returned")
        return 0

    db = SessionLocal()
    count = 0
    try:
        for _, row in df.iterrows():
            existing = db.query(StockPrice).filter(
                StockPrice.symbol == symbol,
                StockPrice.date == row["date"]
            ).first()

            if not existing:
                record = StockPrice(
                    symbol=symbol,
                    date=row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    adjusted_close=row.get("adjusted_close"),
                )
                db.add(record)
                count += 1

        db.commit()
        logger.info(f"Stored {count} new price records for {symbol}")
    finally:
        db.close()

    return count


def collect_news(symbol: str, days_back: int = 30):
    """Fetch and store news articles."""
    logger.info(f"Collecting news for {symbol}...")
    articles = news_collector.collect_all(f"{symbol} stock", days_back=days_back)

    db = SessionLocal()
    count = 0
    try:
        for article in articles:
            existing = db.query(NewsArticle).filter(
                NewsArticle.symbol == symbol,
                NewsArticle.title == article["title"]
            ).first()

            if not existing:
                record = NewsArticle(
                    symbol=symbol,
                    title=article["title"],
                    source=article["source"],
                    url=article["url"],
                    content=article.get("content", ""),
                    published_at=article.get("published_at"),
                )
                db.add(record)
                count += 1

        db.commit()
        logger.info(f"Stored {count} new articles for {symbol}")
    finally:
        db.close()

    return count


def collect_fundamentals(symbol: str):
    """Fetch and log company fundamentals."""
    logger.info(f"Collecting fundamentals for {symbol}...")
    overview = av_client.get_company_overview(symbol)
    if overview and "Symbol" in overview:
        logger.info(f"Company: {overview.get('Name')}")
        logger.info(f"Sector: {overview.get('Sector')}")
        logger.info(f"Market Cap: {overview.get('MarketCapitalization')}")
        logger.info(f"P/E Ratio: {overview.get('PERatio')}")
    else:
        logger.warning("No company overview data available")
    return overview


def main():
    parser = argparse.ArgumentParser(description="Collect stock data")
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    parser.add_argument("--days", type=int, default=730, help="Days of history")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    args = parser.parse_args()

    init_db()

    if not args.skip_prices:
        collect_prices(args.symbol.upper(), args.days)

    if not args.skip_news:
        collect_news(args.symbol.upper(), min(args.days, 30))

    if not args.skip_fundamentals:
        collect_fundamentals(args.symbol.upper())

    logger.info("Data collection complete")


if __name__ == "__main__":
    main()
