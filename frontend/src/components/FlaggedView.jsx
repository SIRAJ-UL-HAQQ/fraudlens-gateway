import React from 'react';

export default function FlaggedView({ transactions }) {
  const flagged = transactions.filter(t => t.decision === 'REVIEW' || t.decision === 'DECLINE');

  if (flagged.length === 0) {
    return <div style={{ color: '#94a3b8', textAlign: 'center', padding: '1.5rem' }}>No flagged transactions currently in queue.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {flagged.map((tx) => (
        <div key={tx.id} style={{ background: '#1e293b', borderLeft: tx.decision === 'DECLINE' ? '4px solid #ef4444' : '4px solid #f59e0b', padding: '1rem', borderRadius: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <strong style={{ fontSize: '1rem' }}>{tx.id}</strong> — User: <code>{tx.user_id}</code>
            </div>
            <span style={{ fontWeight: 'bold', color: tx.decision === 'DECLINE' ? '#f87171' : '#fbbf24' }}>
              {tx.decision} (Score: {tx.risk_score})
            </span>
          </div>
          <div style={{ marginTop: '0.5rem', color: '#cbd5e1', fontSize: '0.9rem' }}>
            Amount: <strong>${tx.amount.toFixed(2)}</strong> | Time: {new Date(tx.created_at).toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
}
