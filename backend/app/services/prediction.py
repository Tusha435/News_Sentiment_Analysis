"""Prediction engine - orchestrates the full prediction pipeline.

Coordinates:
1. Data fetching (Alpha Vantage)
2. News collection and sentiment analysis
3. Reputation scoring
4. TF-IDF feature extraction
5. Feature engineering
6. Model training / loading
7. Prediction generation (1d, 5d, 21d)
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.app.services.alpha_vantage import av_client
from backend.app.services.news_service import news_collector
from backend.app.services.sentiment import sentiment_analyzer
from backend.app.services.tfidf_engine import tfidf_engine
from backend.app.services.reputation import reputation_scorer
from backend.app.services.features import feature_engineer
from backend.app.services.ml_models import (
    HybridPredictor, XGBoostPredictor, RandomForestPredictor, LSTMPredictor
)
from backend.app.config import PREDICTION_HORIZON_DAYS, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

# Direction labels
DIRECTION_LABELS = {-1: "down", 0: "neutral", 1: "up"}


class PredictionEngine:
    """End-to-end stock prediction pipeline."""

    def __init__(self):
        self.models = {}

    def _get_or_create_model(self, symbol: str, model_type: str = "hybrid"):
        """Get cached model or create new instance."""
        key = f"{symbol}_{model_type}"
        if key not in self.models:
            if model_type == "hybrid":
                model = HybridPredictor(sequence_length=SEQUENCE_LENGTH)
            elif model_type == "xgboost":
                model = XGBoostPredictor()
            elif model_type == "random_forest":
                model = RandomForestPredictor()
            elif model_type == "lstm":
                model = LSTMPredictor(sequence_length=SEQUENCE_LENGTH)
            else:
                model = HybridPredictor(sequence_length=SEQUENCE_LENGTH)

            # Try loading saved model
            model.load(symbol)
            self.models[key] = model

        return self.models[key]

    def fetch_data(self, symbol: str, days: int = 730) -> dict:
        """Fetch all required data for a symbol.

        Returns dict with price_df, news_articles, company_overview
        """
        logger.info(f"Fetching data for {symbol}...")

        # Fetch stock prices
        price_df = av_client.get_daily_prices(symbol, outputsize="full")
        if not price_df.empty and days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            price_df = price_df[price_df["date"] >= cutoff].reset_index(drop=True)

        # Fetch company overview
        overview = {}
        try:
            overview = av_client.get_company_overview(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch company overview: {e}")

        # Fetch news
        company_name = overview.get("Name", symbol)
        news_articles = news_collector.collect_all(
            f"{company_name} {symbol} stock", days_back=min(days, 30)
        )

        return {
            "price_df": price_df,
            "news_articles": news_articles,
            "company_overview": overview,
        }

    def analyze_sentiment_pipeline(self, news_articles: list[dict]) -> list[dict]:
        """Run full sentiment + reputation + TF-IDF pipeline on news.

        Returns enriched article list with all scores.
        """
        if not news_articles:
            return []

        # Extract contents for batch processing
        contents = [a.get("content", "") for a in news_articles]

        # TF-IDF analysis
        tfidf_engine.fit(contents)

        # Analyze each article
        enriched = []
        for article in news_articles:
            content = article.get("content", "")

            # Sentiment
            sentiment = sentiment_analyzer.analyze_ensemble(content)
            article["sentiment"] = sentiment

            # TF-IDF features
            tfidf_features = tfidf_engine.get_feature_vector(content)
            article["tfidf_features"] = tfidf_features

            # Reputation score
            rep_score = reputation_scorer.score_article(article)
            article["reputation"] = rep_score

            enriched.append(article)

        return enriched

    def build_daily_sentiment_features(
        self, enriched_articles: list[dict], price_dates: pd.Series
    ) -> list[dict]:
        """Aggregate article-level sentiment into daily features.

        Groups articles by date and computes daily aggregates.
        """
        if not enriched_articles:
            return []

        # Group by date
        date_groups = {}
        for article in enriched_articles:
            pub = article.get("published_at")
            if pub:
                date_key = pub.date() if hasattr(pub, "date") else pub
            else:
                continue

            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(article)

        daily_features = []
        for date_key, articles in sorted(date_groups.items()):
            polarities = [a["sentiment"]["polarity"] for a in articles]
            subjectivities = [a["sentiment"]["subjectivity"] for a in articles]
            labels = [a["sentiment"]["label"] for a in articles]
            rep_scores = [a["reputation"]["sentiment_impact_score"] for a in articles]
            vol_scores = [a["reputation"]["volatility_contribution"] for a in articles]
            tfidf_mags = [a["tfidf_features"]["tfidf_magnitude"] for a in articles]
            term_counts = [a["tfidf_features"]["financial_term_count"] for a in articles]

            n = len(articles)
            daily_features.append({
                "date": pd.Timestamp(date_key),
                "sentiment_polarity": np.mean(polarities),
                "sentiment_subjectivity": np.mean(subjectivities),
                "sentiment_positive_ratio": labels.count("Positive") / n,
                "sentiment_negative_ratio": labels.count("Negative") / n,
                "reputation_score": 50 + np.mean(rep_scores) * 16.67,
                "event_volatility_index": np.mean(vol_scores),
                "news_frequency": n,
                "tfidf_magnitude": np.mean(tfidf_mags),
                "financial_term_density": np.mean(term_counts),
            })

        return daily_features

    def train_model(
        self, symbol: str, model_type: str = "hybrid", days: int = 730
    ) -> dict:
        """Full training pipeline for a symbol.

        Args:
            symbol: Stock ticker
            model_type: Model to train
            days: Days of historical data

        Returns:
            Training metrics dict
        """
        # Fetch data
        data = self.fetch_data(symbol, days)
        price_df = data["price_df"]

        if price_df.empty:
            return {"error": "No price data available"}

        # Sentiment pipeline
        enriched_articles = self.analyze_sentiment_pipeline(data["news_articles"])
        daily_sentiment = self.build_daily_sentiment_features(
            enriched_articles, price_df["date"]
        )

        # Get model
        model = self._get_or_create_model(symbol, model_type)

        if model_type == "hybrid":
            # Prepare flat features for tree models
            X_flat, y_price, y_direction, feature_names = \
                feature_engineer.prepare_training_data(
                    price_df, daily_sentiment, target_horizon=1
                )

            # Prepare sequential features for LSTM
            X_seq, y_seq_price, _, _ = feature_engineer.prepare_sequences(
                price_df, daily_sentiment,
                sequence_length=SEQUENCE_LENGTH, target_horizon=1
            )

            if len(X_flat) < 30:
                return {"error": "Insufficient training data"}

            metrics = model.train(X_flat, X_seq, y_price, y_direction, feature_names)

        elif model_type == "lstm":
            X_seq, y_price, y_direction, feature_names = \
                feature_engineer.prepare_sequences(
                    price_df, daily_sentiment,
                    sequence_length=SEQUENCE_LENGTH, target_horizon=1
                )
            if len(X_seq) == 0:
                return {"error": "Insufficient sequential data"}
            metrics = model.train(X_seq, y_price, feature_names=feature_names)

        else:
            X, y_price, y_direction, feature_names = \
                feature_engineer.prepare_training_data(
                    price_df, daily_sentiment, target_horizon=1
                )
            if len(X) < 30:
                return {"error": "Insufficient training data"}
            metrics = model.train(X, y_price, y_direction, feature_names)

        # Save model
        model.save(symbol)

        return metrics

    def predict(self, symbol: str, model_type: str = "hybrid") -> dict:
        """Generate predictions for a symbol.

        Returns predictions for multiple horizons with confidence scores.
        """
        # Fetch recent data
        data = self.fetch_data(symbol, days=365)
        price_df = data["price_df"]

        if price_df.empty:
            return {"error": "No price data available"}

        # Sentiment pipeline
        enriched_articles = self.analyze_sentiment_pipeline(data["news_articles"])
        daily_sentiment = self.build_daily_sentiment_features(
            enriched_articles, price_df["date"]
        )

        # Get model
        model = self._get_or_create_model(symbol, model_type)

        if not model.is_trained:
            # Auto-train if no saved model
            logger.info(f"No trained model for {symbol}, training now...")
            train_result = self.train_model(symbol, model_type)
            if "error" in train_result:
                return train_result

        current_price = float(price_df["close"].iloc[-1])

        predictions = []
        for horizon in PREDICTION_HORIZON_DAYS:
            # Prepare features for this horizon
            X, y_price, y_dir, feature_names = \
                feature_engineer.prepare_training_data(
                    price_df, daily_sentiment, target_horizon=horizon
                )

            if len(X) == 0:
                continue

            # Use last data point for prediction
            X_latest = X[-1:].copy()

            if model_type == "hybrid":
                # Also get sequence for LSTM
                X_seq, _, _, _ = feature_engineer.prepare_sequences(
                    price_df, daily_sentiment,
                    sequence_length=SEQUENCE_LENGTH, target_horizon=horizon
                )
                X_seq_latest = X_seq[-1:] if len(X_seq) > 0 else None
                result = model.predict(X_latest, X_seq_latest)
            else:
                if model_type == "lstm":
                    X_seq, _, _, _ = feature_engineer.prepare_sequences(
                        price_df, daily_sentiment,
                        sequence_length=SEQUENCE_LENGTH, target_horizon=horizon
                    )
                    if len(X_seq) == 0:
                        continue
                    result = {"predicted_price": model.predict(X_seq[-1:])}
                else:
                    result = model.predict(X_latest)

            predicted_price = float(result["predicted_price"][0])
            direction_val = int(result.get("predicted_direction", [0])[0])
            direction = DIRECTION_LABELS.get(direction_val, "neutral")
            confidence = float(result.get("confidence", [0.5])[0])

            horizon_label = {1: "1d", 5: "5d", 21: "21d"}.get(horizon, f"{horizon}d")
            predictions.append({
                "horizon": horizon_label,
                "target_days": horizon,
                "predicted_price": round(predicted_price, 2),
                "predicted_change_pct": round(
                    (predicted_price - current_price) / current_price * 100, 2
                ),
                "direction": direction,
                "confidence": round(confidence, 4),
            })

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predictions": predictions,
            "model_used": model_type,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_dashboard_data(self, symbol: str, days: int = 365) -> dict:
        """Get all data needed for the frontend dashboard."""
        data = self.fetch_data(symbol, days)
        price_df = data["price_df"]
        overview = data["company_overview"]

        if price_df.empty:
            return {"error": "No price data available for this symbol"}

        # Sentiment pipeline
        enriched_articles = self.analyze_sentiment_pipeline(data["news_articles"])
        daily_sentiment = self.build_daily_sentiment_features(
            enriched_articles, price_df["date"]
        )

        # Technical indicators
        tech_df = feature_engineer.build_price_features(price_df)

        # Reputation aggregate
        rep_data = reputation_scorer.score_batch(enriched_articles)

        # TF-IDF important terms
        contents = [a.get("content", "") for a in enriched_articles]
        important_terms = tfidf_engine.get_corpus_important_terms(contents) if contents else []

        # Generate predictions
        predictions = self.predict(symbol)

        # Price history for chart
        price_history = []
        for _, row in tech_df.tail(min(days, len(tech_df))).iterrows():
            price_history.append({
                "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
                "sma_20": round(float(row.get("sma_20", 0)), 2),
                "sma_50": round(float(row.get("sma_50", 0)), 2),
                "rsi_14": round(float(row.get("rsi_14", 50)), 2),
                "macd": round(float(row.get("macd", 0)), 4),
                "bb_upper": round(float(row.get("bb_upper", 0)), 2),
                "bb_lower": round(float(row.get("bb_lower", 0)), 2),
            })

        # Sentiment timeline
        sentiment_timeline = []
        for feat in daily_sentiment:
            sentiment_timeline.append({
                "date": feat["date"].isoformat() if hasattr(feat["date"], "isoformat") else str(feat["date"]),
                "polarity": round(feat["sentiment_polarity"], 4),
                "positive_ratio": round(feat["sentiment_positive_ratio"], 2),
                "negative_ratio": round(feat["sentiment_negative_ratio"], 2),
                "news_count": feat["news_frequency"],
                "reputation_score": round(feat["reputation_score"], 2),
            })

        # News articles for display
        news_display = []
        for article in enriched_articles[:20]:
            news_display.append({
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "published_at": article.get("published_at", ""),
                "sentiment": article.get("sentiment", {}),
                "reputation_score": article.get("reputation", {}).get("sentiment_impact_score", 0),
                "event_type": article.get("reputation", {}).get("event_type", ""),
                "tfidf_top_terms": article.get("tfidf_features", {}).get("financial_term_scores", {}),
            })

        return {
            "symbol": symbol,
            "company_name": overview.get("Name", symbol),
            "description": overview.get("Description", ""),
            "sector": overview.get("Sector", ""),
            "industry": overview.get("Industry", ""),
            "market_cap": overview.get("MarketCapitalization", ""),
            "pe_ratio": overview.get("PERatio", ""),
            "current_price": round(float(price_df["close"].iloc[-1]), 2),
            "price_history": price_history,
            "predictions": predictions.get("predictions", []),
            "model_used": predictions.get("model_used", ""),
            "sentiment_timeline": sentiment_timeline,
            "reputation": {
                "overall_score": rep_data.get("overall_score", 50),
                "risk_level": rep_data.get("risk_level", "medium"),
                "sentiment_impact_score": rep_data.get("sentiment_impact_score", 0),
                "event_breakdown": rep_data.get("event_breakdown", []),
            },
            "news_articles": news_display,
            "tfidf_important_terms": important_terms[:15],
            "technical_summary": {
                "rsi": round(float(tech_df["rsi_14"].iloc[-1]), 2) if "rsi_14" in tech_df else 50,
                "macd": round(float(tech_df["macd"].iloc[-1]), 4) if "macd" in tech_df else 0,
                "sma_20": round(float(tech_df["sma_20"].iloc[-1]), 2) if "sma_20" in tech_df else 0,
                "bb_position": "middle",
            },
        }


# Module-level singleton
prediction_engine = PredictionEngine()
