import React from 'react';

/**
 * Reputation Risk Gauge - semicircular gauge visualization.
 * Score 0-100, color coded by risk level.
 */
export default function ReputationGauge({ reputation }) {
  const { overall_score = 50, risk_level = 'medium', sentiment_impact_score = 0, event_breakdown = [] } = reputation || {};

  const score = Math.max(0, Math.min(100, overall_score));
  // Convert 0-100 to angle (0-180 degrees)
  const angle = (score / 100) * 180;
  // SVG arc path
  const radius = 80;
  const cx = 100;
  const cy = 95;
  const startAngle = Math.PI;
  const endAngle = Math.PI - (angle * Math.PI) / 180;

  const x1 = cx + radius * Math.cos(startAngle);
  const y1 = cy + radius * Math.sin(startAngle);
  const x2 = cx + radius * Math.cos(endAngle);
  const y2 = cy + radius * Math.sin(endAngle);
  const largeArc = angle > 180 ? 1 : 0;

  const getColor = (level) => {
    switch (level) {
      case 'low': return { main: '#10b981', bg: 'bg-green-900/30', text: 'text-green-400', label: 'LOW RISK' };
      case 'medium': return { main: '#f59e0b', bg: 'bg-yellow-900/30', text: 'text-yellow-400', label: 'MEDIUM RISK' };
      case 'high': return { main: '#ef4444', bg: 'bg-red-900/30', text: 'text-red-400', label: 'HIGH RISK' };
      case 'critical': return { main: '#dc2626', bg: 'bg-red-900/50', text: 'text-red-300', label: 'CRITICAL' };
      default: return { main: '#6b7280', bg: 'bg-gray-900/30', text: 'text-gray-400', label: 'UNKNOWN' };
    }
  };

  const colors = getColor(risk_level);

  return (
    <div className="card animate-fade-in">
      <h3 className="card-header">Reputation Risk Score</h3>

      <div className="flex flex-col items-center">
        {/* Gauge SVG */}
        <svg viewBox="0 0 200 110" className="w-48 h-28">
          {/* Background arc */}
          <path
            d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
            fill="none" stroke="#1e293b" strokeWidth="12" strokeLinecap="round"
          />
          {/* Value arc */}
          {score > 0 && (
            <path
              d={`M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`}
              fill="none" stroke={colors.main} strokeWidth="12" strokeLinecap="round"
            />
          )}
          {/* Score text */}
          <text x={cx} y={cy - 10} textAnchor="middle" className="text-3xl font-bold" fill="white" fontSize="28">
            {Math.round(score)}
          </text>
          <text x={cx} y={cy + 10} textAnchor="middle" fill="#64748b" fontSize="10">
            / 100
          </text>
        </svg>

        {/* Risk level badge */}
        <div className={`mt-2 px-4 py-1.5 rounded-full text-sm font-semibold ${colors.bg} ${colors.text}`}>
          {colors.label}
        </div>

        {/* SIS Score */}
        <div className="mt-4 text-center">
          <span className="text-xs text-gray-500">Sentiment Impact Score</span>
          <p className={`text-lg font-mono font-semibold ${sentiment_impact_score >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {sentiment_impact_score >= 0 ? '+' : ''}{sentiment_impact_score.toFixed(4)}
          </p>
        </div>

        {/* Event breakdown */}
        {event_breakdown.length > 0 && (
          <div className="mt-4 w-full">
            <p className="text-xs text-gray-500 mb-2">Event Breakdown</p>
            <div className="space-y-1.5">
              {event_breakdown.slice(0, 5).map((ev, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-gray-300 capitalize">{ev.event.replace(/_/g, ' ')}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">{ev.count}x</span>
                    <span className={ev.avg_sis >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {ev.avg_sis >= 0 ? '+' : ''}{ev.avg_sis.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
