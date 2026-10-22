import React, { useState } from 'react';
import { scoreTransaction } from '../api/client';

export default function TestTransactionForm({ onTransactionScored }) {
  const [userId, setUserId] = useState('usr_alice_44');
  const [amount, setAmount] = useState('149.99');
  const [merchantId, setMerchantId] = useState('merch_bestbuy');
  const [location, setLocation] = useState('US');
  const [velocity, setVelocity] = useState('1');
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const payload = {
        user_id: userId,
        amount: parseFloat(amount),
        currency: 'USD',
        merchant_id: merchantId,
        location: location,
        custom_features: {
          velocity_1h: parseFloat(velocity)
        }
      };
      const result = await scoreTransaction(payload);
      setLastResult(result);
      if (onTransactionScored) onTransactionScored(result);
    } catch (err) {
      alert(`Scoring failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
      <h3 style={{ margin: '0 0 1rem 0', color: '#38bdf8' }}>Simulate Real-time Transaction</h3>
      <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8' }}>User ID</label>
          <input value={userId} onChange={e => setUserId(e.target.value)} required style={{ width: '100%', padding: '0.5rem', background: '#0f172a', border: '1px solid #334155', color: '#fff', borderRadius: '4px' }} />
        </div>
        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Amount ($)</label>
          <input type="number" step="0.01" value={amount} onChange={e => setAmount(e.target.value)} required style={{ width: '100%', padding: '0.5rem', background: '#0f172a', border: '1px solid #334155', color: '#fff', borderRadius: '4px' }} />
        </div>
        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Merchant ID</label>
          <input value={merchantId} onChange={e => setMerchantId(e.target.value)} required style={{ width: '100%', padding: '0.5rem', background: '#0f172a', border: '1px solid #334155', color: '#fff', borderRadius: '4px' }} />
        </div>
        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Country Code</label>
          <input value={location} onChange={e => setLocation(e.target.value)} required style={{ width: '100%', padding: '0.5rem', background: '#0f172a', border: '1px solid #334155', color: '#fff', borderRadius: '4px' }} />
        </div>
        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8' }}>1h Velocity</label>
          <input type="number" value={velocity} onChange={e => setVelocity(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: '#0f172a', border: '1px solid #334155', color: '#fff', borderRadius: '4px' }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button type="submit" disabled={loading} style={{ width: '100%', padding: '0.6rem', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
            {loading ? 'Evaluating...' : 'Score Transaction'}
          </button>
        </div>
      </form>

      {lastResult && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#0f172a', borderRadius: '6px', border: '1px solid #334155' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong>Outcome: <span style={{ color: lastResult.decision === 'APPROVE' ? '#34d399' : lastResult.decision === 'REVIEW' ? '#fbbf24' : '#f87171' }}>{lastResult.decision}</span></strong>
            <span>Risk Score: <strong>{lastResult.risk_score}/100</strong> (Latency: {lastResult.latency_ms}ms)</span>
          </div>
          {lastResult.shap_explanation && lastResult.shap_explanation.length > 0 && (
            <div style={{ marginTop: '0.8rem' }}>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>SHAP Explainability Impact:</div>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {lastResult.shap_explanation.map((item, idx) => (
                  <span key={idx} style={{ fontSize: '0.75rem', background: item.impact === 'INCREASES_RISK' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)', color: item.impact === 'INCREASES_RISK' ? '#fca5a5' : '#6ee7b7', padding: '3px 8px', borderRadius: '4px' }}>
                    {item.feature}: {item.shap_value > 0 ? `+${item.shap_value}` : item.shap_value}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
