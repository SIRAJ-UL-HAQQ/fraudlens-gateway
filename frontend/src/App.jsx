import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './auth/AuthContext';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { Shell } from './components/layout/Shell';
import { Login } from './pages/Login';
import { DashboardFeed } from './pages/DashboardFeed';
import { FlaggedQueue } from './pages/FlaggedQueue';
import { MetricsView } from './pages/MetricsView';
import { TestTransactionView } from './pages/TestTransactionView';
import { ExportView } from './pages/ExportView';
import { ErrorBoundary } from './components/shared/ErrorBoundary';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ErrorBoundary>
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <Shell />
                  </ProtectedRoute>
                }
              >
                <Route index element={<DashboardFeed />} />
                <Route path="flagged" element={<FlaggedQueue />} />
                <Route path="metrics" element={<MetricsView />} />
                <Route path="test" element={<TestTransactionView />} />
                <Route path="export" element={<ExportView />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ErrorBoundary>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
