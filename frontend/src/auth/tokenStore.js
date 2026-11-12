const TOKEN_KEY = 'fraudlens_access_token';
const EXPIRES_AT_KEY = 'fraudlens_token_expires_at';
const USER_KEY = 'fraudlens_user';

let authExpiredListeners = [];

export const tokenStore = {
  getToken() {
    try {
      if (this.isExpired()) {
        this.clearToken();
        return null;
      }
      return sessionStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },

  setToken(token, expiresIn = 3600, username = '', role = 'analyst') {
    try {
      const expiresAt = Date.now() + expiresIn * 1000;
      sessionStorage.setItem(TOKEN_KEY, token);
      sessionStorage.setItem(EXPIRES_AT_KEY, String(expiresAt));
      sessionStorage.setItem(USER_KEY, JSON.stringify({ username, role }));
    } catch (e) {
      console.error('Failed to save auth state to sessionStorage', e);
    }
  },

  clearToken() {
    try {
      sessionStorage.removeItem(TOKEN_KEY);
      sessionStorage.removeItem(EXPIRES_AT_KEY);
      sessionStorage.removeItem(USER_KEY);
    } catch (e) {
      console.error('Failed to clear auth state from sessionStorage', e);
    }
  },

  isExpired() {
    try {
      const expiresAt = sessionStorage.getItem(EXPIRES_AT_KEY);
      if (!expiresAt) return true;
      return Date.now() >= Number(expiresAt);
    } catch {
      return true;
    }
  },

  getUser() {
    try {
      const user = sessionStorage.getItem(USER_KEY);
      return user ? JSON.parse(user) : null;
    } catch {
      return null;
    }
  },

  onAuthExpired(callback) {
    authExpiredListeners.push(callback);
    return () => {
      authExpiredListeners = authExpiredListeners.filter((cb) => cb !== callback);
    };
  },

  notifyAuthExpired() {
    authExpiredListeners.forEach((callback) => {
      try {
        callback();
      } catch (err) {
        console.error('Error in authExpired listener', err);
      }
    });
  },
};
