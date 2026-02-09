"""Model evaluation script.

Usage:
    python -m models.evaluate --symbol AAPL --model hybrid

Evaluates trained models against recent holdout data and
generates performance reports.
"""

import argparse
import logging
import json
import sys

sys.path.insert(0, ".")

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from backend.app.services.prediction import prediction_engine
from backend.app.services.features import feature_engineer
from backend.app.config import SEQUENCE_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(symbol: str, model_type: str, test_days: int = 60):
    """Evaluate a trained model on holdout data."""
    # Fetch data
    data = prediction_engine.fetch_data(symbol, days=730)
    price_df = data["price_df"]

    if price_df.empty:
        logger.error("No price data available")
        return

    # Sentiment pipeline
    enriched = prediction_engine.analyze_sentiment_pipeline(data["news_articles"])
    daily_sentiment = prediction_engine.build_daily_sentiment_features(
        enriched, price_df["date"]
    )

    # Prepare features
    X, y_price, y_dir, feature_names = feature_engineer.prepare_training_data(
        price_df, daily_sentiment, target_horizon=1
    )

    if len(X) < test_days + 30:
        logger.error("Insufficient data for evaluation")
        return

    # Split into train/test
    split_idx = len(X) - test_days
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_price[:split_idx], y_price[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]

    # Get model
    model = prediction_engine._get_or_create_model(symbol, model_type)

    if not model.is_trained:
        logger.info("Model not trained, training now...")
        if model_type == "hybrid":
            X_seq, y_seq, _, _ = feature_engineer.prepare_sequences(
                price_df.iloc[:split_idx], daily_sentiment,
                sequence_length=SEQUENCE_LENGTH, target_horizon=1
            )
            model.train(X_train, X_seq, y_train, y_dir_train, feature_names)
        else:
            model.train(X_train, y_train, y_dir_train, feature_names)

    # Predict on test set
    if model_type == "hybrid":
        X_seq_test, _, _, _ = feature_engineer.prepare_sequences(
            price_df.iloc[split_idx - SEQUENCE_LENGTH:], daily_sentiment,
            sequence_length=SEQUENCE_LENGTH, target_horizon=1
        )
        result = model.predict(X_test, X_seq_test if len(X_seq_test) > 0 else None)
    elif model_type == "lstm":
        X_seq_test, _, _, _ = feature_engineer.prepare_sequences(
            price_df.iloc[split_idx - SEQUENCE_LENGTH:], daily_sentiment,
            sequence_length=SEQUENCE_LENGTH, target_horizon=1
        )
        if len(X_seq_test) == 0:
            logger.error("Cannot create sequences for LSTM evaluation")
            return
        preds = model.predict(X_seq_test)
        result = {"predicted_price": preds}
    else:
        result = model.predict(X_test)

    y_pred = result["predicted_price"]

    # Align lengths
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[-min_len:]
    y_pred = y_pred[-min_len:]

    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    report = {
        "symbol": symbol,
        "model_type": model_type,
        "test_samples": int(min_len),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2_score": round(float(r2), 4),
        "mape_pct": round(float(mape), 2),
    }

    # Direction accuracy
    if "predicted_direction" in result:
        dir_pred = result["predicted_direction"][-min_len:]
        dir_acc = accuracy_score(y_dir_test[-min_len:], dir_pred)
        report["direction_accuracy"] = round(float(dir_acc), 4)

    logger.info("Evaluation Results:")
    logger.info(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    parser.add_argument("--model", default="hybrid",
                        choices=["hybrid", "xgboost", "random_forest", "lstm"])
    parser.add_argument("--test-days", type=int, default=60)
    args = parser.parse_args()

    evaluate_model(args.symbol.upper(), args.model, args.test_days)


if __name__ == "__main__":
    main()
