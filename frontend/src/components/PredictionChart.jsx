import React, { useMemo } from 'react';
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Area, ReferenceLine
} from 'recharts';

/**
 * Dual-line chart showing actual market price vs AI predicted price.
 * Also displays volume as a bar chart overlay.
 */
export default function PredictionChart({ priceHistory, predictions, currentPrice }) {
  const chartData = useMemo(() => {
    if (!priceHistory || priceHistory.length === 0) return [];

    // Take last 120 data points for clarity
    const recent = priceHistory.slice(-120);

    const data = recent.map((p) => ({
      date: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      fullDate: p.date,
      actual: p.close,
      sma20: p.sma_20,
      sma50: p.sma_50,
      bbUpper: p.bb_upper,
      bbLower: p.bb_lower,
      volume: p.volume,
    }));

    // Add prediction points (future dates)
    if (predictions && predictions.length > 0) {
      const lastDate = new Date(recent[recent.length - 1].date);

      predictions.forEach((pred) => {
        const futureDate = new Date(lastDate);
        futureDate.setDate(futureDate.getDate() + pred.target_days);

        data.push({
          date: futureDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          fullDate: futureDate.toISOString(),
          predicted: pred.predicted_price,
          volume: 0,
        });
      });

      // Connect the last actual point to predictions
      if (data.length > 0 && recent.length > 0) {
        const lastActualIdx = recent.length - 1;
        if (data[lastActualIdx]) {
          data[lastActualIdx].predicted = data[lastActualIdx].actual;
        }
      }
    }

    return data;
  }, [priceHistory, predictions]);

  if (chartData.length === 0) {
    return <div className="card text-center py-12 text-gray-400">No price data available</div>;
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload) return null;
    return (
      <div className="bg-dark-800 border border-dark-700 rounded-lg p-3 shadow-xl">
        <p className="text-xs text-gray-400 mb-2">{label}</p>
        {payload.map((entry, i) => (
          <p key={i} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: ${Number(entry.value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h3 className="card-header mb-0">Price Chart &amp; AI Predictions</h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-primary-400" /> Actual
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-green-400" style={{ borderBottom: '2px dashed' }} /> Predicted
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-yellow-500 opacity-50" /> SMA 20
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="volumeGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="bbGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.1} />
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="date"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={{ stroke: '#334155' }}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="price"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={{ stroke: '#334155' }}
            tickFormatter={(v) => `$${v}`}
            domain={['auto', 'auto']}
          />
          <YAxis
            yAxisId="volume"
            orientation="right"
            tick={{ fill: '#334155', fontSize: 9 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v > 1e6 ? `${(v / 1e6).toFixed(0)}M` : ''}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Bollinger Bands area */}
          <Area yAxisId="price" type="monotone" dataKey="bbUpper" stroke="none" fill="url(#bbGrad)" />
          <Area yAxisId="price" type="monotone" dataKey="bbLower" stroke="none" fill="transparent" />

          {/* Volume bars */}
          <Bar yAxisId="volume" dataKey="volume" fill="url(#volumeGrad)" barSize={3} />

          {/* SMA lines */}
          <Line yAxisId="price" type="monotone" dataKey="sma20" stroke="#eab308" strokeWidth={1} strokeOpacity={0.5} dot={false} />
          <Line yAxisId="price" type="monotone" dataKey="sma50" stroke="#f97316" strokeWidth={1} strokeOpacity={0.4} dot={false} />

          {/* Actual price line */}
          <Line
            yAxisId="price" type="monotone" dataKey="actual"
            stroke="#60a5fa" strokeWidth={2} dot={false}
            name="Actual Price"
            activeDot={{ r: 4, fill: '#60a5fa' }}
          />

          {/* Predicted price line */}
          <Line
            yAxisId="price" type="monotone" dataKey="predicted"
            stroke="#10b981" strokeWidth={2.5}
            strokeDasharray="8 4"
            dot={{ r: 5, fill: '#10b981', stroke: '#064e3b', strokeWidth: 2 }}
            name="AI Predicted"
            connectNulls
          />

          {currentPrice && (
            <ReferenceLine
              yAxisId="price" y={currentPrice}
              stroke="#94a3b8" strokeDasharray="3 3"
              label={{ value: `$${currentPrice}`, fill: '#94a3b8', fontSize: 11, position: 'right' }}
            />
          )}

          <Legend
            wrapperStyle={{ paddingTop: '10px' }}
            iconType="line"
            formatter={(value) => <span className="text-gray-400 text-xs">{value}</span>}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
