"""Feature engineering layer.

Combines structured (price/technical) and unstructured (sentiment/NLP) features
into a unified feature matrix for ML model training and inference.

Structured features:
- OHLCV data
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Derived price features (returns, momentum, volatility)

Unstructured features:
- Sentiment polarity (ensemble)
- Reputation/SIS score
- TF-IDF feature magnitudes
- News frequency
- Event volatility index
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.app.services.technical import technical_indicators

logger = logging.getLogger(__name__)

# All feature columns used by the models
STRUCTURED_FEATURES = [
    "open", "high", "low", "close", "volume",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr_14",
    "obv", "volume_ratio",
    "daily_return", "log_return", "price_range",
    "momentum_5", "momentum_10", "momentum_20",
    "volatility_10", "volatility_20",
    "price_to_sma20", "price_to_sma50",
]

UNSTRUCTURED_FEATURES = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
    "reputation_score",
    "event_volatility_index",
    "news_frequency",
    "tfidf_magnitude",
    "financial_term_density",
]

ALL_FEATURES = STRUCTURED_FEATURES + UNSTRUCTURED_FEATURES


class FeatureEngineer:
    """Builds combined feature matrices from price and sentiment data."""

    def build_price_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicator features to price data.

        Args:
            price_df: DataFrame with columns: date, open, high, low, close, volume

        Returns:
            DataFrame with technical indicator columns added
        """
        if price_df.empty:
            return price_df

        df = technical_indicators.compute_all(price_df)
        return df

    def build_sentiment_features(
        self, sentiment_data: list[dict], dates: pd.Series
    ) -> pd.DataFrame:
        """Build daily sentiment feature vectors.

        Args:
            sentiment_data: List of dicts with date, polarity, subjectivity,
                           reputation_score, event_volatility, tfidf_magnitude, etc.
            dates: Series of dates to align with

        Returns:
            DataFrame indexed by date with sentiment features
        """
        if not sentiment_data:
            # Return zeros for all sentiment features
            n = len(dates)
            return pd.DataFrame({
                "sentiment_polarity": np.zeros(n),
                "sentiment_subjectivity": np.zeros(n),
                "sentiment_positive_ratio": np.zeros(n),
                "sentiment_negative_ratio": np.zeros(n),
                "reputation_score": np.full(n, 50.0),
                "event_volatility_index": np.zeros(n),
                "news_frequency": np.zeros(n),
                "tfidf_magnitude": np.zeros(n),
                "financial_term_density": np.zeros(n),
            }, index=dates.values)

        sent_df = pd.DataFrame(sentiment_data)
        if "date" in sent_df.columns:
            sent_df["date"] = pd.to_datetime(sent_df["date"])
            sent_df = sent_df.set_index("date")

        # Reindex to match price dates, forward-fill sentiment
        date_index = pd.DatetimeIndex(dates.values)
        feature_cols = [c for c in UNSTRUCTURED_FEATURES if c in sent_df.columns]

        if feature_cols:
            sent_aligned = sent_df[feature_cols].reindex(date_index, method="ffill")
        else:
            sent_aligned = pd.DataFrame(index=date_index)

        # Fill missing columns
        for col in UNSTRUCTURED_FEATURES:
            if col not in sent_aligned.columns:
                default = 50.0 if col == "reputation_score" else 0.0
                sent_aligned[col] = default

        sent_aligned = sent_aligned.fillna(0)
        return sent_aligned

    def combine_features(
        self,
        price_df: pd.DataFrame,
        sentiment_data: list[dict] = None,
    ) -> pd.DataFrame:
        """Combine price and sentiment features into unified matrix.

        Args:
            price_df: Raw OHLCV DataFrame
            sentiment_data: Optional sentiment scores per date

        Returns:
            Feature matrix DataFrame with all features
        """
        if price_df.empty:
            return pd.DataFrame()

        # Build structured features
        df = self.build_price_features(price_df)

        # Build unstructured features
        sent_features = self.build_sentiment_features(
            sentiment_data or [], df["date"]
        )

        # Align and join
        for col in UNSTRUCTURED_FEATURES:
            if col in sent_features.columns:
                df[col] = sent_features[col].values

        return df

    def create_targets(
        self,
        df: pd.DataFrame,
        horizons: list[int] = None,
    ) -> pd.DataFrame:
        """Create prediction targets for different horizons.

        Args:
            df: Feature DataFrame with 'close' column
            horizons: List of forward-looking days [1, 5, 21]

        Returns:
            DataFrame with target columns added
        """
        if horizons is None:
            horizons = [1, 5, 21]

        result = df.copy()

        for h in horizons:
            # Future price
            result[f"target_price_{h}d"] = result["close"].shift(-h)
            # Future return
            result[f"target_return_{h}d"] = result["close"].pct_change(h).shift(-h)
            # Direction: 1 = up, 0 = neutral, -1 = down
            returns = result[f"target_return_{h}d"]
            result[f"target_direction_{h}d"] = np.where(
                returns > 0.005, 1,
                np.where(returns < -0.005, -1, 0)
            )

        return result

    def normalize_features(
        self, df: pd.DataFrame, feature_cols: list[str] = None
    ) -> tuple[pd.DataFrame, dict]:
        """Normalize features using z-score normalization.

        Returns:
            Tuple of (normalized DataFrame, stats dict for inverse transform)
        """
        if feature_cols is None:
            feature_cols = [c for c in ALL_FEATURES if c in df.columns]

        stats = {}
        result = df.copy()

        for col in feature_cols:
            mean = result[col].mean()
            std = result[col].std()
            if std == 0:
                std = 1.0
            result[col] = (result[col] - mean) / std
            stats[col] = {"mean": mean, "std": std}

        return result, stats

    def prepare_training_data(
        self,
        price_df: pd.DataFrame,
        sentiment_data: list[dict] = None,
        target_horizon: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Full pipeline: features + targets ready for training.

        Returns:
            Tuple of (X_features, y_price, y_direction, feature_names)
        """
        # Combine features
        df = self.combine_features(price_df, sentiment_data)
        if df.empty:
            return np.array([]), np.array([]), np.array([]), []

        # Create targets
        df = self.create_targets(df, [target_horizon])

        # Drop rows with NaN targets
        target_col = f"target_price_{target_horizon}d"
        direction_col = f"target_direction_{target_horizon}d"
        df = df.dropna(subset=[target_col, direction_col])

        # Select features
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feature_cols].values
        y_price = df[target_col].values
        y_direction = df[direction_col].values

        # Handle any remaining NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y_price, y_direction, feature_cols

    def prepare_sequences(
        self,
        price_df: pd.DataFrame,
        sentiment_data: list[dict] = None,
        sequence_length: int = 60,
        target_horizon: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Prepare sequential data for LSTM/GRU models.

        Returns:
            Tuple of (X_sequences, y_price, y_direction, feature_names)
            X_sequences shape: (n_samples, sequence_length, n_features)
        """
        X, y_price, y_direction, feature_names = self.prepare_training_data(
            price_df, sentiment_data, target_horizon
        )

        if len(X) < sequence_length + 1:
            return np.array([]), np.array([]), np.array([]), feature_names

        # Create sliding window sequences
        X_seq = []
        y_p = []
        y_d = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length:i])
            y_p.append(y_price[i])
            y_d.append(y_direction[i])

        return np.array(X_seq), np.array(y_p), np.array(y_d), feature_names


# Module-level singleton
feature_engineer = FeatureEngineer()
