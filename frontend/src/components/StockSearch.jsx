import React, { useState } from 'react';

export default function StockSearch({ onSearch, loading }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const symbol = query.trim().toUpperCase();
    if (symbol) {
      onSearch(symbol);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter ticker (e.g., AAPL)"
          className="input-field w-56 pr-10"
          disabled={loading}
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
          </svg>
        </div>
      </div>
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="btn-primary whitespace-nowrap"
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Analyzing
          </span>
        ) : (
          'Analyze'
        )}
      </button>
    </form>
  );
}
