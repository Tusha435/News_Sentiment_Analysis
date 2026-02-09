"""Tests for TF-IDF engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.services.tfidf_engine import TFIDFEngine


class TestTFIDFEngine:
    """Test suite for TF-IDF analysis engine."""

    def setup_method(self):
        self.engine = TFIDFEngine(max_features=100)
        self.documents = [
            "Apple reported strong revenue growth and record earnings this quarter.",
            "The merger between the two companies raises antitrust concerns.",
            "Layoffs announced as the company restructures its workforce.",
            "New product launch drives innovation and market expansion.",
            "Bankruptcy filing after years of mounting debt and declining revenue.",
        ]

    def test_fit_transform(self):
        matrix = self.engine.fit_transform(self.documents)
        assert matrix.shape[0] == 5
        assert matrix.shape[1] > 0
        assert self.engine.is_fitted

    def test_get_top_terms(self):
        self.engine.fit(self.documents)
        terms = self.engine.get_top_terms(self.documents[0], top_n=5)
        assert len(terms) <= 5
        assert all("term" in t and "score" in t for t in terms)
        assert all(t["score"] > 0 for t in terms)

    def test_financial_term_scores(self):
        text = "The company faces bankruptcy after massive layoffs and debt concerns."
        scores = self.engine.get_financial_term_scores(text)
        assert "bankruptcy" in scores
        assert "layoffs" in scores
        assert "debt" in scores
        assert all(v > 0 for v in scores.values())

    def test_corpus_important_terms(self):
        terms = self.engine.get_corpus_important_terms(self.documents, top_n=10)
        assert len(terms) <= 10
        assert all("term" in t and "avg_score" in t and "doc_frequency" in t for t in terms)

    def test_feature_vector(self):
        self.engine.fit(self.documents)
        fv = self.engine.get_feature_vector("Revenue growth and earnings beat expectations.")
        assert "tfidf_features" in fv
        assert "financial_term_scores" in fv
        assert "tfidf_magnitude" in fv
        assert "financial_term_count" in fv
        assert isinstance(fv["tfidf_magnitude"], float)

    def test_preprocess(self):
        text = "Check https://example.com - Revenue: $500M in 2024!"
        processed = self.engine.preprocess_for_tfidf(text)
        assert "https" not in processed
        assert "$" not in processed
        assert processed == processed.lower()

    def test_empty_corpus(self):
        terms = self.engine.get_corpus_important_terms([], top_n=10)
        assert terms == []

    def test_single_document(self):
        # Single document: use a dedicated engine with min_df=1, max_df=1.0
        engine = TFIDFEngine(max_features=50)
        engine.vectorizer.max_df = 1.0
        matrix = engine.fit_transform(["Revenue growth was strong."])
        assert matrix.shape[0] == 1
        assert matrix.shape[1] > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
