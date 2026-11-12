import React from 'react';
import { Navigate, useLocation, Outlet } from 'react-router-dom';
import { useAuth } from './useAuth';

export function ProtectedRoute({ children }) {
  const { isAuthenticated, sessionExpired } = useAuth();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location, sessionExpired }} replace />;
  }

  return children ? children : <Outlet />;
}

export default ProtectedRoute;
