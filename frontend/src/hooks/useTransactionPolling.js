import { useState, useEffect, useCallback } from 'react';

/**
 * Hook for managing 2-second real-time polling with visibility detection and manual toggle.
 * Pauses automatically when document.hidden is true.
 */
export function useTransactionPolling(initialEnabled = true, intervalMs = 2000) {
  const [isPollingEnabled, setIsPollingEnabled] = useState(initialEnabled);
  const [isTabVisible, setIsTabVisible] = useState(
    typeof document !== 'undefined' ? !document.hidden : true
  );

  useEffect(() => {
    if (typeof document === 'undefined') return;

    const handleVisibilityChange = () => {
      setIsTabVisible(!document.hidden);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const togglePolling = useCallback(() => {
    setIsPollingEnabled((prev) => !prev);
  }, []);

  const pollingInterval = isPollingEnabled && isTabVisible ? intervalMs : false;

  return {
    isPollingEnabled,
    setIsPollingEnabled,
    togglePolling,
    isTabVisible,
    pollingInterval,
  };
}

export default useTransactionPolling;
