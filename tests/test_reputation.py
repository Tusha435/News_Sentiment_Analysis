"""Tests for reputation risk scoring system."""

import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.services.reputation import ReputationScorer


class TestReputationScorer:
    """Test suite for ReputationScorer."""

    def setup_method(self):
        self.scorer = ReputationScorer(decay_half_life_days=7.0)

    def test_classify_event_bankruptcy(self):
        text = "The company has filed for chapter 11 bankruptcy protection."
        event_type, config = self.scorer.classify_event(text)
        assert event_type == "bankruptcy"
        assert config["weight"] < 0

    def test_classify_event_profit(self):
        text = "Company reports record profit surge exceeding all expectations."
        event_type, config = self.scorer.classify_event(text)
        assert event_type == "profit_surge"
        assert config["weight"] > 0

    def test_classify_event_layoffs(self):
        text = "Major tech company announces massive layoffs cutting 10000 jobs."
        event_type, config = self.scorer.classify_event(text)
        assert event_type == "mass_layoffs"

    def test_recency_decay_recent(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        decay = self.scorer.recency_decay(recent)
        assert decay > 0.9  # Recent articles should have high decay

    def test_recency_decay_old(self):
        old = datetime.now(timezone.utc) - timedelta(days=30)
        decay = self.scorer.recency_decay(old)
        assert decay < 0.2  # Old articles should have low decay

    def test_sis_formula(self):
        sis = self.scorer.compute_sis(
            sentiment_polarity=0.8,
            event_weight=-2.0,
            news_reach=0.9,
            recency=0.95,
        )
        # SIS = 0.8 * |-2.0| * 0.9 * 0.95 = 1.368
        assert abs(sis - 1.368) < 0.001

    def test_score_article(self):
        article = {
            "content": "Company reports record profit surge beating all estimates",
            "source": "Reuters",
            "published_at": datetime.now(timezone.utc) - timedelta(hours=2),
            "sentiment": {"polarity": 0.7, "label": "Positive"},
        }
        result = self.scorer.score_article(article)
        assert "sentiment_impact_score" in result
        assert "event_type" in result
        assert "news_reach" in result
        assert "recency_decay" in result
        assert result["news_reach"] == 1.0  # Reuters is top credibility

    def test_score_batch(self):
        articles = [
            {
                "content": "Revenue surged 50% this quarter with record earnings",
                "source": "Bloomberg",
                "published_at": datetime.now(timezone.utc),
                "sentiment": {"polarity": 0.8, "label": "Positive"},
            },
            {
                "content": "CEO resigns amid fraud investigation",
                "source": "CNBC",
                "published_at": datetime.now(timezone.utc) - timedelta(days=1),
                "sentiment": {"polarity": -0.6, "label": "Negative"},
            },
        ]
        result = self.scorer.score_batch(articles)
        assert "overall_score" in result
        assert "risk_level" in result
        assert "event_breakdown" in result
        assert 0 <= result["overall_score"] <= 100
        assert result["risk_level"] in ["low", "medium", "high", "critical"]

    def test_empty_batch(self):
        result = self.scorer.score_batch([])
        assert result["overall_score"] == 50.0
        assert result["risk_level"] == "medium"

    def test_source_credibility(self):
        assert self.scorer.get_source_credibility("Reuters") == 1.0
        assert self.scorer.get_source_credibility("Bloomberg") == 1.0
        assert self.scorer.get_source_credibility("unknown blog") == 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
