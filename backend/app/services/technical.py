"""Technical indicators calculator for stock price data.

Implements:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume-weighted metrics
- Additional derived features
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV price data."""

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over `period` days
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD: Moving Average Convergence Divergence.

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range - volatility indicator."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        return (volume * direction).cumsum()

    @staticmethod
    def vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators and add them to the dataframe.

        Expects columns: open, high, low, close, volume

        Returns:
            DataFrame with all indicator columns added
        """
        result = df.copy()

        close = result["close"]
        high = result["high"]
        low = result["low"]
        volume = result["volume"]

        # Moving Averages
        result["sma_5"] = self.sma(close, 5)
        result["sma_10"] = self.sma(close, 10)
        result["sma_20"] = self.sma(close, 20)
        result["sma_50"] = self.sma(close, 50)
        result["sma_200"] = self.sma(close, 200)
        result["ema_12"] = self.ema(close, 12)
        result["ema_26"] = self.ema(close, 26)

        # RSI
        result["rsi_14"] = self.rsi(close, 14)

        # MACD
        macd_line, signal_line, histogram = self.macd(close)
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self.bollinger_bands(close)
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_mid
        result["bb_lower"] = bb_lower
        result["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

        # ATR
        result["atr_14"] = self.atr(high, low, close, 14)

        # Volume indicators
        result["obv"] = self.obv(close, volume)
        result["vwap"] = self.vwap(high, low, close, volume)
        result["volume_sma_20"] = self.sma(volume, 20)
        result["volume_ratio"] = volume / result["volume_sma_20"].replace(0, np.nan)

        # Price-derived features
        result["daily_return"] = close.pct_change()
        result["log_return"] = np.log(close / close.shift(1))
        result["price_range"] = (high - low) / close.replace(0, np.nan)
        result["gap"] = result["open"] - close.shift(1)

        # Momentum features
        result["momentum_5"] = close / close.shift(5) - 1
        result["momentum_10"] = close / close.shift(10) - 1
        result["momentum_20"] = close / close.shift(20) - 1

        # Volatility
        result["volatility_10"] = result["daily_return"].rolling(10, min_periods=1).std()
        result["volatility_20"] = result["daily_return"].rolling(20, min_periods=1).std()

        # Price relative to moving averages
        result["price_to_sma20"] = close / result["sma_20"].replace(0, np.nan) - 1
        result["price_to_sma50"] = close / result["sma_50"].replace(0, np.nan) - 1

        # Fill NaN values
        result = result.ffill().bfill().fillna(0)

        return result


# Module-level singleton
technical_indicators = TechnicalIndicators()
