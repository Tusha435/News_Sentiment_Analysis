"""Tests for FastAPI endpoints."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from backend.app.main import app


class TestAPI:
    """Test suite for API endpoints."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Stock Prediction Intelligence Platform"
        assert data["status"] == "running"
        assert "endpoints" in data

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_stock_search_requires_query(self):
        response = self.client.get("/api/stocks/search")
        assert response.status_code == 422  # Validation error

    def test_analyze_text_endpoint(self):
        response = self.client.post(
            "/api/sentiment/analyze-text",
            params={"text": "The company reported strong earnings growth."}
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "tfidf_features" in data
        assert data["sentiment"]["label"] in ["Positive", "Negative", "Neutral"]


class TestAPIValidation:
    """Test API input validation."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_prices_invalid_days(self):
        response = self.client.get("/api/stocks/AAPL/prices?days=0")
        assert response.status_code == 422

    def test_sentiment_invalid_days(self):
        response = self.client.get("/api/sentiment/AAPL/analyze?days=0")
        assert response.status_code == 422


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
