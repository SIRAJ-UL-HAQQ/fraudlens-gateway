import React, { useState, useEffect } from 'react';
import MetricsPanel from '../components/MetricsPanel';
import TestTransactionForm from '../components/TestTransactionForm';
import TransactionFeed from '../components/TransactionFeed';
import FlaggedView from '../components/FlaggedView';
import { getMetrics, getTransactions, getExportUrl } from '../api/client';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [activeTab, setActiveTab] = useState('feed'); // 'feed' | 'flagged'

  const refreshData = async () => {
    try {
      const [m, txs] = await Promise.all([getMetrics(), getTransactions()]);
      setMetrics(m);
      setTransactions(txs);
    } catch (err) {
      console.error("Dashboard refresh error:", err);
    }
  };

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.75rem', color: '#f8fafc' }}>FraudLens Gateway</h1>
          <p style={{ margin: '0.25rem 0 0', color: '#94a3b8' }}>Real-time Risk Scoring, SHAP Explainability & Automated Decisioning</p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <a href={getExportUrl('csv')} target="_blank" rel="noreferrer" style={{ textDecoration: 'none', background: '#334155', color: '#f8fafc', padding: '0.5rem 1rem', borderRadius: '4px', fontSize: '0.85rem' }}>Export CSV</a>
          <button onClick={refreshData} style={{ background: '#2563eb', color: '#fff', border: 'none', padding: '0.5rem 1rem', borderRadius: '4px', cursor: 'pointer', fontSize: '0.85rem' }}>Refresh</button>
        </div>
      </div>

      <MetricsPanel metrics={metrics} />
      <TestTransactionForm onTransactionScored={refreshData} />

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
        <button
          onClick={() => setActiveTab('feed')}
          style={{ background: activeTab === 'feed' ? '#3b82f6' : '#1e293b', color: '#fff', border: 'none', padding: '0.5rem 1rem', borderRadius: '4px', cursor: 'pointer' }}
        >
          Live Transaction Feed
        </button>
        <button
          onClick={() => setActiveTab('flagged')}
          style={{ background: activeTab === 'flagged' ? '#3b82f6' : '#1e293b', color: '#fff', border: 'none', padding: '0.5rem 1rem', borderRadius: '4px', cursor: 'pointer' }}
        >
          Flagged / Review Queue
        </button>
      </div>

      {activeTab === 'feed' ? (
        <TransactionFeed transactions={transactions} />
      ) : (
        <FlaggedView transactions={transactions} />
      )}
    </div>
  );
}
