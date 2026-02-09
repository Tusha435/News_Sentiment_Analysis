import React from 'react';

/**
 * News impact display with sentiment-colored cards and TF-IDF terms.
 */
export default function NewsImpact({ articles, tfidfTerms }) {
  const getSentimentStyle = (label) => {
    switch (label) {
      case 'Positive': return 'border-l-green-500 bg-green-900/10';
      case 'Negative': return 'border-l-red-500 bg-red-900/10';
      default: return 'border-l-gray-500 bg-gray-900/10';
    }
  };

  const getSentimentBadge = (label) => {
    switch (label) {
      case 'Positive': return 'badge-positive';
      case 'Negative': return 'badge-negative';
      default: return 'badge-neutral';
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* TF-IDF Important Terms - Heatmap style */}
      {tfidfTerms && tfidfTerms.length > 0 && (
        <div className="card">
          <h3 className="card-header">News Impact Heatmap (TF-IDF Terms)</h3>
          <div className="flex flex-wrap gap-2">
            {tfidfTerms.map((term, i) => {
              const intensity = Math.min(term.avg_score * 10, 1);
              const r = Math.round(59 + intensity * 100);
              const g = Math.round(130 + intensity * 70);
              const b = Math.round(246 - intensity * 100);
              return (
                <div
                  key={i}
                  className="px-3 py-1.5 rounded-lg text-sm font-medium border border-dark-700"
                  style={{
                    backgroundColor: `rgba(${r}, ${g}, ${b}, ${0.15 + intensity * 0.25})`,
                    color: `rgb(${r}, ${g}, ${b})`,
                    fontSize: `${Math.max(11, 11 + intensity * 5)}px`,
                  }}
                  title={`Score: ${term.avg_score.toFixed(4)} | Docs: ${term.doc_frequency}`}
                >
                  {term.term}
                  <span className="ml-1 opacity-50 text-xs">({term.doc_frequency})</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* News articles */}
      <div className="card">
        <h3 className="card-header">Recent News ({articles?.length || 0} articles)</h3>
        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
          {(!articles || articles.length === 0) ? (
            <p className="text-gray-500 text-center py-4">No news articles found</p>
          ) : (
            articles.map((article, i) => (
              <div
                key={i}
                className={`border-l-4 rounded-r-lg p-3 ${getSentimentStyle(article.sentiment?.label)}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <a
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-medium text-gray-200 hover:text-primary-400 transition-colors line-clamp-2"
                    >
                      {article.title || 'Untitled'}
                    </a>
                    <div className="flex items-center gap-2 mt-1.5 text-xs text-gray-500">
                      <span>{article.source}</span>
                      {article.published_at && (
                        <>
                          <span>&middot;</span>
                          <span>{new Date(article.published_at).toLocaleDateString()}</span>
                        </>
                      )}
                      {article.event_type && article.event_type !== 'market_general' && (
                        <>
                          <span>&middot;</span>
                          <span className="text-primary-400 capitalize">{article.event_type.replace(/_/g, ' ')}</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className={getSentimentBadge(article.sentiment?.label)}>
                      {article.sentiment?.label || 'N/A'}
                    </span>
                    <span className="text-xs text-gray-500">
                      SIS: {(article.reputation_score || 0).toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
