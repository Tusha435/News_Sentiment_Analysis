"""Tests for ML models."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from backend.app.services.ml_models import (
    LinearRegressionPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)


def make_synthetic_data(n_samples=200, n_features=10):
    """Generate synthetic training data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Target: linear combination + noise
    weights = np.random.randn(n_features)
    y_price = X @ weights + np.random.randn(n_samples) * 0.5 + 100
    y_direction = np.sign(np.diff(y_price, prepend=y_price[0])).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y_price, y_direction, feature_names


class TestLinearRegression:
    def test_train_and_predict(self):
        X, y_price, _, feature_names = make_synthetic_data()
        model = LinearRegressionPredictor()
        metrics = model.train(X, y_price, feature_names)
        assert model.is_trained
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0  # Should fit reasonably

        preds = model.predict(X[:5])
        assert len(preds) == 5


class TestRandomForest:
    def test_train_and_predict(self):
        X, y_price, y_direction, feature_names = make_synthetic_data()
        model = RandomForestPredictor()
        metrics = model.train(X, y_price, y_direction, feature_names)
        assert model.is_trained
        assert "rmse" in metrics
        assert "feature_importance" in metrics

        result = model.predict(X[:5])
        assert "predicted_price" in result
        assert len(result["predicted_price"]) == 5

    def test_direction_classification(self):
        X, y_price, y_direction, feature_names = make_synthetic_data()
        model = RandomForestPredictor()
        model.train(X, y_price, y_direction, feature_names)
        result = model.predict(X[:5])
        assert "predicted_direction" in result
        assert model.classifier_trained


class TestXGBoost:
    def test_train_and_predict(self):
        X, y_price, y_direction, feature_names = make_synthetic_data()
        model = XGBoostPredictor()
        metrics = model.train(X, y_price, y_direction, feature_names)
        assert model.is_trained
        assert "rmse" in metrics

        result = model.predict(X[:5])
        assert "predicted_price" in result
        assert len(result["predicted_price"]) == 5

    def test_feature_importance(self):
        X, y_price, y_direction, feature_names = make_synthetic_data()
        model = XGBoostPredictor()
        metrics = model.train(X, y_price, y_direction, feature_names)
        assert "feature_importance" in metrics
        assert len(metrics["feature_importance"]) > 0


class TestFeatureEngineering:
    def test_prepare_training_data(self):
        from backend.app.services.features import feature_engineer
        import pandas as pd

        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 100 + np.cumsum(np.random.randn(n) * 2)

        df = pd.DataFrame({
            "date": dates,
            "open": close + np.random.randn(n) * 0.5,
            "high": close + np.abs(np.random.randn(n)),
            "low": close - np.abs(np.random.randn(n)),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        })

        X, y_price, y_dir, names = feature_engineer.prepare_training_data(df)
        assert len(X) > 0
        assert len(y_price) == len(X)
        assert len(y_dir) == len(X)
        assert len(names) > 0
        assert not np.any(np.isnan(X))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
