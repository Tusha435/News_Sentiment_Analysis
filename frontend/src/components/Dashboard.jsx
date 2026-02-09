import React from 'react';
import PredictionChart from './PredictionChart';
import SentimentTimeline from './SentimentTimeline';
import ReputationGauge from './ReputationGauge';
import NewsImpact from './NewsImpact';
import VolumeSentimentOverlay from './VolumeSentimentOverlay';

/**
 * Main dashboard layout assembling all visualization components.
 */
export default function Dashboard({ data }) {
  if (!data) return null;

  const {
    symbol, company_name, description, sector, industry,
    market_cap, pe_ratio, current_price,
    price_history, predictions, sentiment_timeline,
    reputation, news_articles, tfidf_important_terms,
    technical_summary, model_used,
  } = data;

  const formatMarketCap = (cap) => {
    const n = Number(cap);
    if (!n) return 'N/A';
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
    return `$${n.toLocaleString()}`;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Company Header */}
      <div className="card">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold text-white">{symbol}</h2>
              <span className="text-gray-400 text-lg">{company_name}</span>
            </div>
            {(sector || industry) && (
              <p className="text-sm text-gray-500 mt-1">
                {sector}{sector && industry && ' / '}{industry}
              </p>
            )}
            {description && (
              <p className="text-xs text-gray-600 mt-2 max-w-2xl line-clamp-2">{description}</p>
            )}
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="metric-value text-white">${current_price?.toFixed(2)}</p>
              <p className="metric-label">Current Price</p>
            </div>
            <div className="text-right">
              <p className="text-xl font-semibold text-gray-300">{formatMarketCap(market_cap)}</p>
              <p className="metric-label">Market Cap</p>
            </div>
            {pe_ratio && (
              <div className="text-right">
                <p className="text-xl font-semibold text-gray-300">{Number(pe_ratio).toFixed(1)}</p>
                <p className="metric-label">P/E Ratio</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prediction Cards */}
      {predictions && predictions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {predictions.map((pred, i) => {
            const isPositive = pred.predicted_change_pct >= 0;
            return (
              <div key={i} className={`card border-l-4 ${isPositive ? 'border-l-green-500' : 'border-l-red-500'}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide">
                      {pred.horizon === '1d' ? 'Next Day' : pred.horizon === '5d' ? 'Weekly' : 'Monthly'} Prediction
                    </p>
                    <p className="text-2xl font-bold text-white mt-1">
                      ${pred.predicted_price?.toFixed(2)}
                    </p>
                    <p className={`text-sm font-semibold mt-0.5 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                      {isPositive ? '+' : ''}{pred.predicted_change_pct}%
                    </p>
                  </div>
                  <div className="text-right">
                    <div className={`text-3xl ${pred.direction === 'up' ? 'text-green-400' : pred.direction === 'down' ? 'text-red-400' : 'text-gray-400'}`}>
                      {pred.direction === 'up' ? '\u2191' : pred.direction === 'down' ? '\u2193' : '\u2192'}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {(pred.confidence * 100).toFixed(1)}% conf
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Technical Summary Bar */}
      {technical_summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="card py-3 px-4">
            <p className="text-xs text-gray-500">RSI (14)</p>
            <p className={`text-lg font-semibold ${
              technical_summary.rsi > 70 ? 'text-red-400' :
              technical_summary.rsi < 30 ? 'text-green-400' : 'text-white'
            }`}>
              {technical_summary.rsi}
              <span className="text-xs text-gray-500 ml-1">
                {technical_summary.rsi > 70 ? 'Overbought' : technical_summary.rsi < 30 ? 'Oversold' : 'Neutral'}
              </span>
            </p>
          </div>
          <div className="card py-3 px-4">
            <p className="text-xs text-gray-500">MACD</p>
            <p className={`text-lg font-semibold ${technical_summary.macd >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {technical_summary.macd?.toFixed(4)}
            </p>
          </div>
          <div className="card py-3 px-4">
            <p className="text-xs text-gray-500">SMA 20</p>
            <p className="text-lg font-semibold text-white">${technical_summary.sma_20?.toFixed(2)}</p>
          </div>
          <div className="card py-3 px-4">
            <p className="text-xs text-gray-500">Model</p>
            <p className="text-lg font-semibold text-primary-400 capitalize">{model_used || 'hybrid'}</p>
          </div>
        </div>
      )}

      {/* Main Chart */}
      <PredictionChart
        priceHistory={price_history}
        predictions={predictions}
        currentPrice={current_price}
      />

      {/* Two-column layout: Sentiment + Reputation */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <SentimentTimeline data={sentiment_timeline} />
          <VolumeSentimentOverlay
            priceHistory={price_history}
            sentimentTimeline={sentiment_timeline}
          />
        </div>
        <div>
          <ReputationGauge reputation={reputation} />
        </div>
      </div>

      {/* News + TF-IDF */}
      <NewsImpact
        articles={news_articles}
        tfidfTerms={tfidf_important_terms}
      />
    </div>
  );
}
