import React, { useMemo } from 'react';
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';

/**
 * Volume vs Sentiment overlay chart.
 * Shows trading volume as bars with sentiment polarity as a line overlay.
 */
export default function VolumeSentimentOverlay({ priceHistory, sentimentTimeline }) {
  const chartData = useMemo(() => {
    if (!priceHistory || priceHistory.length === 0) return [];

    const sentimentMap = {};
    if (sentimentTimeline) {
      sentimentTimeline.forEach((s) => {
        const key = new Date(s.date).toISOString().split('T')[0];
        sentimentMap[key] = s;
      });
    }

    return priceHistory.slice(-60).map((p) => {
      const dateKey = new Date(p.date).toISOString().split('T')[0];
      const sentiment = sentimentMap[dateKey];
      return {
        date: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        volume: p.volume,
        polarity: sentiment?.polarity || 0,
        reputation: sentiment?.reputation_score || 50,
        volumeColor: p.close >= p.open ? '#10b981' : '#ef4444',
      };
    });
  }, [priceHistory, sentimentTimeline]);

  if (chartData.length === 0) {
    return <div className="card text-center py-8 text-gray-500">No data available</div>;
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload) return null;
    return (
      <div className="bg-dark-800 border border-dark-700 rounded-lg p-3 shadow-xl text-sm">
        <p className="text-gray-400 mb-2">{label}</p>
        {payload.map((entry, i) => (
          <p key={i} style={{ color: entry.color }}>
            {entry.name}: {entry.name === 'Volume'
              ? (entry.value / 1e6).toFixed(1) + 'M'
              : Number(entry.value).toFixed(3)}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="card animate-fade-in">
      <h3 className="card-header">Volume vs Sentiment Overlay</h3>
      <ResponsiveContainer width="100%" height={250}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis
            yAxisId="volume"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickFormatter={(v) => v > 1e6 ? `${(v / 1e6).toFixed(0)}M` : v}
          />
          <YAxis
            yAxisId="sentiment" orientation="right"
            tick={{ fill: '#64748b', fontSize: 10 }}
            domain={[-1, 1]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={(value) => <span className="text-gray-400 text-xs">{value}</span>}
          />
          <Bar
            yAxisId="volume" dataKey="volume" name="Volume"
            fill="#3b82f6" opacity={0.4} barSize={4}
          />
          <Line
            yAxisId="sentiment" type="monotone" dataKey="polarity" name="Sentiment"
            stroke="#f59e0b" strokeWidth={2} dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
