export const mockOverrides = {
  activeState: 'normal', // 'normal' | 'stub' | 'unloaded' | 'disconnected' | '503' | 'network_error'
  simulatedDelayMs: 150,

  fakeCredentials: {
    username: 'analyst_1',
    password: 'password123',
    token: 'mock-analyst-token-dev-only',
    role: 'analyst',
  },

  healthStates: {
    normal: {
      status: 'healthy',
      service: 'FraudLens Gateway',
      version: '1.0.0',
      database: 'connected',
      model: {
        mode: 'trained',
        version: 'v1.0.0-xgb',
      },
    },
    stub: {
      status: 'healthy',
      service: 'FraudLens Gateway',
      version: '1.0.0',
      database: 'connected',
      model: {
        mode: 'stub',
        version: 'stub',
      },
    },
    unloaded: {
      status: 'unhealthy',
      service: 'FraudLens Gateway',
      version: '1.0.0',
      database: 'connected',
      model: {
        mode: 'unloaded',
        version: 'none',
      },
    },
    disconnected: {
      status: 'unhealthy',
      service: 'FraudLens Gateway',
      version: '1.0.0',
      database: 'disconnected',
      model: {
        mode: 'stub',
        version: 'stub',
      },
    },
  },

  variantBLowRiskTransaction: {
    transaction_id: 'e5b20112-789a-4cde-8901-234567abcdef',
    transaction_data: {
      step: 15,
      type: 'TRANSFER',
      amount: 205000.0,
      name_orig: 'C999888777',
      old_balance_orig: 500000.0,
      new_balance_orig: 295000.0,
      name_dest: 'M111222333',
      old_balance_dest: 10000.0,
      new_balance_dest: 215000.0,
    },
    prediction: {
      transaction_id: 'e5b20112-789a-4cde-8901-234567abcdef',
      risk_score: 0.15,
      top_features: [
        { feature: 'amount', shap_value: 0.08, direction: 'increases_risk' },
        { feature: 'oldbalanceOrg', shap_value: -0.12, direction: 'decreases_risk' },
        { feature: 'newbalanceOrig', shap_value: -0.05, direction: 'decreases_risk' },
      ],
      explanation_status: 'Stable',
      gateway_decision: 'Low Risk',
      rule_flags: ['LARGE_TRANSFER_AMOUNT'],
      imbalance_strategy: 'smote_enn',
      model_version: 'v1.0.0-xgb',
      latency_ms: 22.4,
    },
    created_at: '2026-10-24T10:00:00Z',
    reviews: [],
    latest_review: null,
  },

  mockConfusionMatrix: {
    true_positives: 1551,
    false_positives: 94,
    true_negatives: 116410,
    false_negatives: 91,
  },

  exportError: false,
};
