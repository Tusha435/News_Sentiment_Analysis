"""Model training script.

Usage:
    python -m models.train --symbol AAPL --model hybrid --days 730

Trains ML models (Random Forest, XGBoost, LSTM, Hybrid) on historical
price data combined with sentiment features.
"""

import argparse
import logging
import json
import sys

sys.path.insert(0, ".")

from backend.app.services.prediction import prediction_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    parser.add_argument("--model", default="hybrid",
                        choices=["hybrid", "xgboost", "random_forest", "lstm"],
                        help="Model type to train")
    parser.add_argument("--days", type=int, default=730, help="Days of training data")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    logger.info(f"Training {args.model} model for {symbol} with {args.days} days of data")

    metrics = prediction_engine.train_model(symbol, args.model, args.days)

    if "error" in metrics:
        logger.error(f"Training failed: {metrics['error']}")
        sys.exit(1)

    logger.info("Training complete!")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    main()
