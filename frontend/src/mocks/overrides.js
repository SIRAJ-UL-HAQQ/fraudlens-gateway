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
};
