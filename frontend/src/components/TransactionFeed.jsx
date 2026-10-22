import React from 'react';

export default function TransactionFeed({ transactions }) {
  if (!transactions || transactions.length === 0) {
    return <div style={{ color: '#94a3b8', textAlign: 'center', padding: '2rem' }}>No transactions recorded yet.</div>;
  }

  const getBadgeStyle = (decision) => {
    switch (decision) {
      case 'APPROVE': return { background: '#064e3b', color: '#34d399' };
      case 'REVIEW': return { background: '#78350f', color: '#fcd34d' };
      case 'DECLINE': return { background: '#7f1d1d', color: '#fca5a5' };
      default: return { background: '#334155', color: '#cbd5e1' };
    }
  };

  return (
    <div style={{ background: '#1e293b', borderRadius: '8px', overflow: 'hidden' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
        <thead>
          <tr style={{ background: '#0f172a', color: '#94a3b8', borderBottom: '1px solid #334155' }}>
            <th style={{ padding: '0.75rem 1rem' }}>TX ID</th>
            <th style={{ padding: '0.75rem 1rem' }}>User</th>
            <th style={{ padding: '0.75rem 1rem' }}>Amount</th>
            <th style={{ padding: '0.75rem 1rem' }}>Decision</th>
            <th style={{ padding: '0.75rem 1rem' }}>Risk Score</th>
            <th style={{ padding: '0.75rem 1rem' }}>Time</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((tx) => (
            <tr key={tx.id} style={{ borderBottom: '1px solid #334155' }}>
              <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace' }}>{tx.id.substring(0, 10)}...</td>
              <td style={{ padding: '0.75rem 1rem' }}>{tx.user_id}</td>
              <td style={{ padding: '0.75rem 1rem', fontWeight: 'bold' }}>${tx.amount.toFixed(2)} {tx.currency}</td>
              <td style={{ padding: '0.75rem 1rem' }}>
                <span style={{ padding: '3px 8px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: 'bold', ...getBadgeStyle(tx.decision) }}>
                  {tx.decision}
                </span>
              </td>
              <td style={{ padding: '0.75rem 1rem' }}>{tx.risk_score}/100</td>
              <td style={{ padding: '0.75rem 1rem', color: '#94a3b8', fontSize: '0.8rem' }}>
                {new Date(tx.created_at).toLocaleTimeString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
