import React, { useState, useEffect, useCallback } from 'react';
import { AuthContext } from './context';
import { tokenStore } from './tokenStore';
import { login as apiLogin } from '../api/client';

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => tokenStore.getToken());
  const [user, setUser] = useState(() => tokenStore.getUser());
  const [sessionExpired, setSessionExpired] = useState(false);

  useEffect(() => {
    const unsubscribe = tokenStore.onAuthExpired(() => {
      setToken(null);
      setUser(null);
      setSessionExpired(true);
    });
    return unsubscribe;
  }, []);

  const login = useCallback(async (credentials) => {
    const data = await apiLogin(credentials);
    const username = data.username || credentials.username;
    const role = data.role || 'analyst';
    const expiresIn = data.expires_in || 3600;

    tokenStore.setToken(data.access_token, expiresIn, username, role);
    setToken(data.access_token);
    setUser({ username, role });
    setSessionExpired(false);
    return data;
  }, []);

  const logout = useCallback(() => {
    tokenStore.clearToken();
    setToken(null);
    setUser(null);
    setSessionExpired(false);
  }, []);

  const resetSessionExpired = useCallback(() => {
    setSessionExpired(false);
  }, []);

  const value = {
    token,
    user,
    isAuthenticated: Boolean(token && !tokenStore.isExpired()),
    sessionExpired,
    resetSessionExpired,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
