"""Tests for technical indicators."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backend.app.services.technical import TechnicalIndicators


class TestTechnicalIndicators:
    """Test suite for technical indicator calculations."""

    def setup_method(self):
        self.ti = TechnicalIndicators()
        # Generate synthetic price data
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

        self.df = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_sma(self):
        sma = self.ti.sma(self.df["close"], 20)
        assert len(sma) == len(self.df)
        assert not sma.isna().all()

    def test_ema(self):
        ema = self.ti.ema(self.df["close"], 12)
        assert len(ema) == len(self.df)
        assert not ema.isna().all()

    def test_rsi_range(self):
        rsi = self.ti.rsi(self.df["close"])
        valid = rsi.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_macd(self):
        macd_line, signal_line, histogram = self.ti.macd(self.df["close"])
        assert len(macd_line) == len(self.df)
        assert len(signal_line) == len(self.df)
        assert len(histogram) == len(self.df)

    def test_bollinger_bands(self):
        upper, middle, lower = self.ti.bollinger_bands(self.df["close"])
        # Drop NaN rows before comparison (first few points may have NaN std)
        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_atr(self):
        atr = self.ti.atr(self.df["high"], self.df["low"], self.df["close"])
        assert len(atr) == len(self.df)
        assert (atr.dropna() >= 0).all()

    def test_compute_all(self):
        result = self.ti.compute_all(self.df)
        assert "sma_20" in result.columns
        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "daily_return" in result.columns
        assert "momentum_5" in result.columns
        assert len(result) == len(self.df)

    def test_compute_all_no_nans(self):
        result = self.ti.compute_all(self.df)
        # After ffill/bfill/fillna, there should be no NaNs
        assert result.isna().sum().sum() == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
