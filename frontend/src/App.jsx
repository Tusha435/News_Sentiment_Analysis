import React, { useState, useCallback } from 'react';
import StockSearch from './components/StockSearch';
import Dashboard from './components/Dashboard';
import { getDashboard } from './services/api';

export default function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentSymbol, setCurrentSymbol] = useState('');

  const handleSearch = useCallback(async (symbol) => {
    setLoading(true);
    setError(null);
    setCurrentSymbol(symbol.toUpperCase());
    try {
      const response = await getDashboard(symbol.toUpperCase());
      setDashboardData(response.data);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to load data';
      setError(msg);
      setDashboardData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <header className="border-b border-dark-700 bg-dark-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center text-xl font-bold">
              S
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">StockPredict AI</h1>
              <p className="text-xs text-gray-400">Hybrid Intelligence Platform</p>
            </div>
          </div>
          <StockSearch onSearch={handleSearch} loading={loading} />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-32">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-gray-400 text-lg">Analyzing {currentSymbol}...</p>
              <p className="text-gray-500 text-sm mt-2">
                Fetching prices, analyzing sentiment, scoring reputation, generating predictions
              </p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="card border-red-800 bg-red-900/20 text-center py-12 animate-fade-in">
            <div className="text-4xl mb-4">!</div>
            <h3 className="text-xl font-semibold text-red-400 mb-2">Analysis Failed</h3>
            <p className="text-gray-400">{error}</p>
            <p className="text-gray-500 text-sm mt-4">
              Make sure the backend is running and API keys are configured.
            </p>
          </div>
        )}

        {/* Dashboard */}
        {dashboardData && !loading && (
          <Dashboard data={dashboardData} />
        )}

        {/* Empty State */}
        {!dashboardData && !loading && !error && (
          <div className="text-center py-32 animate-fade-in">
            <div className="text-6xl mb-6">&#x1F4C8;</div>
            <h2 className="text-3xl font-bold text-white mb-4">
              Stock Prediction Intelligence Platform
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto text-lg mb-8">
              Enter a stock symbol above to get AI-powered predictions combining
              sentiment analysis, technical indicators, TF-IDF text analysis,
              and reputation risk scoring.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
              {['AAPL', 'GOOGL', 'MSFT', 'TSLA'].map((s) => (
                <button
                  key={s}
                  onClick={() => handleSearch(s)}
                  className="btn-secondary text-lg py-3"
                >
                  {s}
                </button>
              ))}
            </div>
            <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto text-left">
              <div className="card">
                <h3 className="font-semibold text-primary-400 mb-2">Sentiment Engine</h3>
                <p className="text-sm text-gray-400">
                  VADER + TextBlob ensemble with financial lexicon augmentation and NER entity extraction.
                </p>
              </div>
              <div className="card">
                <h3 className="font-semibold text-primary-400 mb-2">ML Ensemble</h3>
                <p className="text-sm text-gray-400">
                  XGBoost + Random Forest + LSTM hybrid model with inverse-RMSE weighted predictions.
                </p>
              </div>
              <div className="card">
                <h3 className="font-semibold text-primary-400 mb-2">Risk Scoring</h3>
                <p className="text-sm text-gray-400">
                  SIS formula: Polarity x Event_Weight x News_Reach x Recency_Decay for market impact.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-dark-700 py-4 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-500 text-sm">
          Stock Prediction Intelligence Platform | Hybrid ML + NLP Analysis
        </div>
      </footer>
    </div>
  );
}
