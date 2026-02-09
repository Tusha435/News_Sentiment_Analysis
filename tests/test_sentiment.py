"""Tests for sentiment analysis engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.services.sentiment import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""

    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        text = "The company reported record profits and strong revenue growth this quarter."
        result = self.analyzer.analyze_ensemble(text)
        assert result["label"] == "Positive"
        assert result["polarity"] > 0

    def test_negative_sentiment(self):
        text = "The company faces bankruptcy and massive layoffs following the fraud scandal."
        result = self.analyzer.analyze_ensemble(text)
        assert result["label"] == "Negative"
        assert result["polarity"] < 0

    def test_neutral_sentiment(self):
        text = "The company held its annual meeting today."
        result = self.analyzer.analyze_ensemble(text)
        assert result["label"] in ["Neutral", "Positive", "Negative"]
        assert -1 <= result["polarity"] <= 1

    def test_ensemble_has_all_fields(self):
        text = "Apple announced a new product launch."
        result = self.analyzer.analyze_ensemble(text)
        assert "label" in result
        assert "polarity" in result
        assert "subjectivity" in result
        assert "vader_compound" in result
        assert "textblob_polarity" in result
        assert "entities" in result
        assert "vader_detail" in result

    def test_vader_financial_lexicon(self):
        text = "The stock is bullish with an upgrade from analysts."
        result = self.analyzer.analyze_vader(text)
        assert result["compound"] > 0

        text = "Bearish outlook with downgrade concerns."
        result = self.analyzer.analyze_vader(text)
        assert result["compound"] < 0

    def test_preprocess(self):
        text = "Check http://example.com for   details!!!   🚀"
        processed = self.analyzer.preprocess(text)
        assert "http" not in processed
        assert "  " not in processed

    def test_lemmatize(self):
        text = "the companies are running quickly towards growth"
        result = self.analyzer.lemmatize(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_analysis(self):
        texts = [
            "Revenue surged 50% this quarter",
            "The CEO resigned amid controversy",
            "Regular trading day with normal volume",
        ]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert all("label" in r for r in results)

    def test_empty_text(self):
        result = self.analyzer.analyze_ensemble("")
        assert result["label"] in ["Neutral", "Positive", "Negative"]
        assert "polarity" in result

    def test_polarity_range(self):
        texts = [
            "Amazing growth and record profits!",
            "Terrible losses and bankruptcy risk.",
            "Normal day.",
        ]
        for text in texts:
            result = self.analyzer.analyze_ensemble(text)
            assert -1 <= result["polarity"] <= 1
            assert 0 <= result["subjectivity"] <= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
