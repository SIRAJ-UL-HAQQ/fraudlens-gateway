import React from 'react';

export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;

  const cards = [
    { label: 'Total Volume', value: metrics.total_transactions, color: '#38bdf8' },
    { label: 'Approved', value: metrics.approved_count, color: '#34d399' },
    { label: 'Under Review', value: metrics.review_count, color: '#fbbf24' },
    { label: 'Declined (Fraud)', value: metrics.declined_count, color: '#f87171' },
    { label: 'Fraud Rate', value: `${metrics.fraud_rate_pct}%`, color: '#f43f5e' },
    { label: 'Avg Risk Score', value: metrics.average_risk_score, color: '#a78bfa' },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
      {cards.map((c, i) => (
        <div key={i} style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', borderLeft: `4px solid ${c.color}` }}>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{c.label}</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginTop: '0.3rem', color: '#f8fafc' }}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}
