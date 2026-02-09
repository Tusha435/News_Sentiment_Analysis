import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Bar, ComposedChart
} from 'recharts';

/**
 * Sentiment timeline graph showing polarity over time.
 */
export default function SentimentTimeline({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="card text-center py-8 text-gray-500">
        No sentiment data available
      </div>
    );
  }

  const chartData = data.map((d) => ({
    date: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    polarity: d.polarity,
    positiveRatio: d.positive_ratio,
    negativeRatio: -d.negative_ratio,
    newsCount: d.news_count,
    reputation: d.reputation_score,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload) return null;
    return (
      <div className="bg-dark-800 border border-dark-700 rounded-lg p-3 shadow-xl text-sm">
        <p className="text-gray-400 mb-2">{label}</p>
        {payload.map((entry, i) => (
          <p key={i} style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="card animate-fade-in">
      <h3 className="card-header">Sentiment Timeline</h3>
      <ResponsiveContainer width="100%" height={250}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="posGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="negGrad" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 10 }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[-1, 1]} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#334155" />

          <Area
            type="monotone" dataKey="polarity" name="Sentiment Polarity"
            stroke="#60a5fa" fill="url(#posGrad)" strokeWidth={2}
          />
          <Bar dataKey="newsCount" name="News Count" fill="#3b82f6" opacity={0.3} barSize={6} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
