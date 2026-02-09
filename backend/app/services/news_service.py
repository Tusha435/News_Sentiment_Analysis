"""News collection service for financial news ingestion.

Sources:
- NewsAPI
- RSS feeds (financial outlets)
- Web scraping fallback

Includes text cleaning and preprocessing.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from backend.app.config import NEWS_API_KEY, NEWS_API_URL

logger = logging.getLogger(__name__)

# Financial RSS feed sources
RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

# User agent for web requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class NewsCollector:
    """Collects and preprocesses financial news from multiple sources."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NEWS_API_KEY
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_from_newsapi(
        self, query: str, days_back: int = 30, max_articles: int = 50
    ) -> list[dict]:
        """Fetch news articles from NewsAPI.

        Args:
            query: Search query (company name or ticker)
            days_back: How many days back to search
            max_articles: Maximum number of articles

        Returns:
            List of article dicts with title, source, url, content, published_at
        """
        if not self.api_key:
            logger.warning("No NewsAPI key configured, returning empty results")
            return []

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(max_articles, 100),
            "from": from_date,
        }

        try:
            response = self.session.get(NEWS_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data}")
                return []

            articles = []
            for item in data.get("articles", []):
                content = item.get("content") or item.get("description") or ""
                full_content = self._scrape_full_content(item.get("url", ""))
                if full_content:
                    content = full_content

                articles.append({
                    "title": item.get("title", ""),
                    "source": item.get("source", {}).get("name", "Unknown"),
                    "url": item.get("url", ""),
                    "content": self.clean_text(content),
                    "published_at": self._parse_date(item.get("publishedAt")),
                })

            return articles

        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []

    def fetch_from_rss(self, query: str) -> list[dict]:
        """Fetch news from RSS feeds filtered by query terms."""
        articles = []
        query_terms = query.lower().split()

        for source_name, feed_url in RSS_FEEDS.items():
            try:
                response = self.session.get(feed_url, timeout=10)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.content, "xml")
                items = soup.find_all("item")

                for item in items[:20]:
                    title = item.find("title")
                    title_text = title.get_text() if title else ""
                    desc = item.find("description")
                    desc_text = desc.get_text() if desc else ""
                    link = item.find("link")
                    link_text = link.get_text() if link else ""
                    pub_date = item.find("pubDate")
                    pub_text = pub_date.get_text() if pub_date else ""

                    combined = f"{title_text} {desc_text}".lower()
                    if any(term in combined for term in query_terms):
                        articles.append({
                            "title": title_text,
                            "source": source_name,
                            "url": link_text,
                            "content": self.clean_text(desc_text),
                            "published_at": self._parse_rss_date(pub_text),
                        })

            except Exception as e:
                logger.debug(f"RSS feed {source_name} failed: {e}")

        return articles

    def _scrape_full_content(self, url: str) -> Optional[str]:
        """Scrape full article content from URL."""
        if not url:
            return None
        try:
            response = self.session.get(url, timeout=8)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text(strip=True) for p in paragraphs)
            return self.clean_text(content) if content else None

        except Exception:
            return None

    def collect_all(self, query: str, days_back: int = 30, max_articles: int = 50) -> list[dict]:
        """Collect news from all available sources, deduplicated."""
        all_articles = []

        # NewsAPI (primary)
        newsapi_articles = self.fetch_from_newsapi(query, days_back, max_articles)
        all_articles.extend(newsapi_articles)

        # RSS feeds (supplementary)
        rss_articles = self.fetch_from_rss(query)
        all_articles.extend(rss_articles)

        # Deduplicate by title similarity
        seen_titles = set()
        unique = []
        for article in all_articles:
            normalized = article["title"].lower().strip()
            if normalized not in seen_titles and article["title"]:
                seen_titles.add(normalized)
                unique.append(article)

        return unique[:max_articles]

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Remove HTML entities and tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
        return text

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse ISO date string."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_rss_date(date_str: str) -> Optional[datetime]:
        """Parse RSS date format."""
        if not date_str:
            return None
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None


news_collector = NewsCollector()
