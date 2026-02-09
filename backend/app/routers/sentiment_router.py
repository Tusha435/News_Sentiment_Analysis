"""Sentiment analysis and reputation scoring API endpoints."""

from fastapi import APIRouter, Query, HTTPException

from backend.app.services.news_service import news_collector
from backend.app.services.sentiment import sentiment_analyzer
from backend.app.services.tfidf_engine import tfidf_engine
from backend.app.services.reputation import reputation_scorer

router = APIRouter(prefix="/api/sentiment", tags=["Sentiment"])


@router.get("/{symbol}/analyze")
async def analyze_sentiment(
    symbol: str,
    days: int = Query(30, ge=1, le=90, description="Days of news to analyze"),
    max_articles: int = Query(20, ge=5, le=100),
):
    """Analyze sentiment for a stock symbol from recent news."""
    try:
        # Collect news
        articles = news_collector.collect_all(
            f"{symbol} stock", days_back=days, max_articles=max_articles
        )

        if not articles:
            return {
                "symbol": symbol.upper(),
                "articles_analyzed": 0,
                "sentiment_summary": {
                    "avg_polarity": 0,
                    "avg_subjectivity": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                },
                "articles": [],
            }

        # Analyze each article
        analyzed = []
        contents = []
        for article in articles:
            content = article.get("content", "")
            contents.append(content)
            sentiment = sentiment_analyzer.analyze_ensemble(content)
            article["sentiment"] = sentiment

            # Reputation score
            rep = reputation_scorer.score_article(article)
            article["reputation"] = rep

            analyzed.append({
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "published_at": str(article.get("published_at", "")),
                "sentiment_label": sentiment["label"],
                "polarity": sentiment["polarity"],
                "subjectivity": sentiment["subjectivity"],
                "vader_compound": sentiment["vader_compound"],
                "event_type": rep["event_type"],
                "sentiment_impact_score": rep["sentiment_impact_score"],
            })

        # Aggregate statistics
        polarities = [a["polarity"] for a in analyzed]
        labels = [a["sentiment_label"] for a in analyzed]

        # TF-IDF analysis
        important_terms = tfidf_engine.get_corpus_important_terms(contents)

        # Reputation aggregate
        rep_aggregate = reputation_scorer.score_batch(articles)

        return {
            "symbol": symbol.upper(),
            "articles_analyzed": len(analyzed),
            "sentiment_summary": {
                "avg_polarity": round(sum(polarities) / len(polarities), 4),
                "avg_subjectivity": round(
                    sum(a["subjectivity"] for a in analyzed) / len(analyzed), 4
                ),
                "positive_count": labels.count("Positive"),
                "negative_count": labels.count("Negative"),
                "neutral_count": labels.count("Neutral"),
            },
            "reputation": {
                "overall_score": rep_aggregate["overall_score"],
                "risk_level": rep_aggregate["risk_level"],
                "event_breakdown": rep_aggregate["event_breakdown"],
            },
            "tfidf_important_terms": important_terms[:15],
            "articles": analyzed,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-text")
async def analyze_text_sentiment(text: str):
    """Analyze sentiment of arbitrary text."""
    try:
        result = sentiment_analyzer.analyze_ensemble(text)
        tfidf_features = tfidf_engine.get_feature_vector(text)

        return {
            "sentiment": result,
            "tfidf_features": tfidf_features,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
