"""Reputation Risk Scoring System.

Converts sentiment analysis results into numeric market impact signals
using the Sentiment Impact Score (SIS) formula:

    SIS = Sentiment_Polarity x Event_Weight x News_Reach x Recency_Decay

Where:
- Sentiment_Polarity: [-1, 1] from ensemble sentiment model
- Event_Weight: Predefined weight based on event category
- News_Reach: Normalized score based on source credibility/reach
- Recency_Decay: Exponential decay factor based on article age

Risk levels: LOW (0-25), MEDIUM (25-50), HIGH (50-75), CRITICAL (75-100)
"""

import math
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# Event categories with associated weights and volatility impact
EVENT_CATEGORIES = {
    # Strong Negative Events
    "bankruptcy": {"weight": -3.0, "volatility": 0.9, "keywords": [
        "bankruptcy", "bankrupt", "chapter 11", "chapter 7", "insolvency", "insolvent"
    ]},
    "fraud": {"weight": -3.0, "volatility": 0.85, "keywords": [
        "fraud", "scandal", "embezzlement", "misconduct", "corruption", "scam"
    ]},
    "mass_layoffs": {"weight": -2.0, "volatility": 0.7, "keywords": [
        "layoffs", "laid off", "job cuts", "workforce reduction", "downsizing",
        "restructuring", "fired", "terminated employees"
    ]},
    "legal_action": {"weight": -1.8, "volatility": 0.6, "keywords": [
        "lawsuit", "litigation", "sued", "legal action", "class action",
        "regulatory fine", "penalty", "settlement", "sec investigation"
    ]},
    "ceo_departure": {"weight": -1.5, "volatility": 0.65, "keywords": [
        "ceo resignation", "ceo fired", "ceo departure", "ceo steps down",
        "executive departure", "leadership change"
    ]},
    "downgrade": {"weight": -1.5, "volatility": 0.5, "keywords": [
        "downgrade", "underperform", "sell rating", "underweight",
        "price target cut", "negative outlook"
    ]},

    # Moderate Negative Events
    "earnings_miss": {"weight": -1.2, "volatility": 0.55, "keywords": [
        "earnings miss", "missed expectations", "revenue miss", "profit warning",
        "guidance cut", "lower guidance"
    ]},
    "debt_concern": {"weight": -1.0, "volatility": 0.4, "keywords": [
        "debt concern", "credit downgrade", "junk status", "overleveraged",
        "debt burden", "default risk"
    ]},

    # Strong Positive Events
    "profit_surge": {"weight": 2.5, "volatility": 0.6, "keywords": [
        "profit surge", "record profit", "earnings beat", "revenue surge",
        "exceeded expectations", "blowout earnings"
    ]},
    "product_launch": {"weight": 1.5, "volatility": 0.4, "keywords": [
        "product launch", "new product", "unveiled", "announced",
        "breakthrough", "innovative", "game changer"
    ]},
    "upgrade": {"weight": 1.5, "volatility": 0.45, "keywords": [
        "upgrade", "outperform", "buy rating", "overweight",
        "price target raised", "positive outlook"
    ]},
    "acquisition": {"weight": 1.2, "volatility": 0.5, "keywords": [
        "acquisition", "acquire", "takeover", "buyout", "merger",
        "merge", "deal announced"
    ]},
    "expansion": {"weight": 1.0, "volatility": 0.3, "keywords": [
        "expansion", "new market", "international growth", "new facility",
        "hiring spree", "job creation", "market entry"
    ]},
    "dividend": {"weight": 0.8, "volatility": 0.2, "keywords": [
        "dividend increase", "special dividend", "buyback", "share repurchase",
        "return to shareholders"
    ]},
    "partnership": {"weight": 0.8, "volatility": 0.3, "keywords": [
        "partnership", "collaboration", "strategic alliance", "joint venture",
        "contract win", "major deal"
    ]},

    # Neutral / Mixed
    "regulatory": {"weight": -0.3, "volatility": 0.35, "keywords": [
        "regulation", "regulatory", "compliance", "government policy",
        "new rules", "legislation"
    ]},
    "market_general": {"weight": 0.0, "volatility": 0.2, "keywords": [
        "market update", "trading volume", "market cap", "index",
        "sector performance"
    ]},
}

# Source credibility scores (proxy for "News Reach")
SOURCE_CREDIBILITY = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "wall street journal": 0.95,
    "wsj": 0.95,
    "financial times": 0.95,
    "cnbc": 0.9,
    "yahoo finance": 0.85,
    "marketwatch": 0.85,
    "seeking alpha": 0.7,
    "motley fool": 0.65,
    "benzinga": 0.7,
    "investopedia": 0.6,
    "business insider": 0.75,
    "forbes": 0.8,
    "barrons": 0.85,
}
DEFAULT_SOURCE_CREDIBILITY = 0.5


class ReputationScorer:
    """Computes reputation risk scores from sentiment and event analysis."""

    def __init__(self, decay_half_life_days: float = 7.0):
        """Initialize scorer.

        Args:
            decay_half_life_days: Half-life for recency decay (default 7 days)
        """
        self.decay_lambda = math.log(2) / decay_half_life_days

    def classify_event(self, text: str) -> tuple[str, dict]:
        """Classify the type of financial event from text content.

        Returns:
            Tuple of (event_type, event_config)
        """
        text_lower = text.lower()

        best_match = None
        best_count = 0

        for event_type, config in EVENT_CATEGORIES.items():
            count = sum(1 for kw in config["keywords"] if kw in text_lower)
            if count > best_count:
                best_count = count
                best_match = event_type

        if best_match and best_count > 0:
            return best_match, EVENT_CATEGORIES[best_match]

        return "market_general", EVENT_CATEGORIES["market_general"]

    def get_source_credibility(self, source: str) -> float:
        """Get credibility score for a news source (proxy for reach)."""
        source_lower = source.lower().strip()
        for name, score in SOURCE_CREDIBILITY.items():
            if name in source_lower:
                return score
        return DEFAULT_SOURCE_CREDIBILITY

    def recency_decay(self, published_at: datetime) -> float:
        """Calculate recency decay factor.

        Uses exponential decay: decay = e^(-lambda * days_old)
        More recent articles have higher impact.
        """
        if not published_at:
            return 0.5  # Default for unknown dates

        now = datetime.now(timezone.utc)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        days_old = max((now - published_at).total_seconds() / 86400, 0)
        return math.exp(-self.decay_lambda * days_old)

    def compute_sis(
        self,
        sentiment_polarity: float,
        event_weight: float,
        news_reach: float,
        recency: float,
    ) -> float:
        """Compute Sentiment Impact Score.

        SIS = Sentiment_Polarity x Event_Weight x News_Reach x Recency_Decay

        The score ranges from approximately -3 to +3 for extreme cases.
        """
        return sentiment_polarity * abs(event_weight) * news_reach * recency

    def score_article(self, article: dict) -> dict:
        """Score a single news article for reputation impact.

        Args:
            article: Dict with keys: content, source, published_at, sentiment

        Returns:
            Dict with SIS score, event classification, and components
        """
        content = article.get("content", "")
        source = article.get("source", "")
        published_at = article.get("published_at")
        sentiment = article.get("sentiment", {})

        polarity = sentiment.get("polarity", 0.0)
        event_type, event_config = self.classify_event(content)
        event_weight = event_config["weight"]
        news_reach = self.get_source_credibility(source)
        recency = self.recency_decay(published_at)
        volatility = event_config["volatility"]

        sis = self.compute_sis(polarity, event_weight, news_reach, recency)

        return {
            "sentiment_impact_score": round(sis, 4),
            "event_type": event_type,
            "event_weight": event_weight,
            "news_reach": round(news_reach, 2),
            "recency_decay": round(recency, 4),
            "volatility_contribution": round(volatility * abs(polarity), 4),
            "sentiment_polarity": round(polarity, 4),
        }

    def score_batch(self, articles: list[dict]) -> dict:
        """Score a batch of articles and compute aggregate reputation metrics.

        Returns:
            Dict with overall_score, risk_level, event_breakdown, individual scores
        """
        if not articles:
            return {
                "overall_score": 50.0,
                "risk_level": "medium",
                "event_breakdown": [],
                "individual_scores": [],
                "sentiment_impact_score": 0.0,
                "event_volatility_index": 0.0,
            }

        individual_scores = [self.score_article(a) for a in articles]

        # Aggregate SIS: weighted by recency
        total_sis = sum(s["sentiment_impact_score"] for s in individual_scores)
        avg_sis = total_sis / len(individual_scores)

        # Event volatility index: average volatility contribution
        avg_volatility = sum(
            s["volatility_contribution"] for s in individual_scores
        ) / len(individual_scores)

        # Convert to 0-100 reputation scale
        # SIS ranges roughly from -3 to +3
        # Map to 0-100 where 50 is neutral, >50 is positive, <50 is negative
        overall_score = max(0, min(100, 50 + (avg_sis * 16.67)))

        # Determine risk level
        if overall_score >= 75:
            risk_level = "low"
        elif overall_score >= 50:
            risk_level = "medium"
        elif overall_score >= 25:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Event breakdown
        from collections import Counter
        event_counts = Counter(s["event_type"] for s in individual_scores)
        event_breakdown = [
            {"event": event, "count": count,
             "avg_sis": round(
                 sum(s["sentiment_impact_score"] for s in individual_scores
                     if s["event_type"] == event) / count, 4
             )}
            for event, count in event_counts.most_common()
        ]

        return {
            "overall_score": round(overall_score, 2),
            "risk_level": risk_level,
            "sentiment_impact_score": round(avg_sis, 4),
            "event_volatility_index": round(avg_volatility, 4),
            "event_breakdown": event_breakdown,
            "individual_scores": individual_scores,
            "total_articles": len(articles),
        }


# Module-level singleton
reputation_scorer = ReputationScorer()
