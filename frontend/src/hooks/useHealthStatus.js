import { useQuery } from '@tanstack/react-query';
import { getHealth } from '../api/client';

export function calculateHealthPollingInterval(queryInstance) {
  if (typeof document !== 'undefined' && document.hidden) {
    return false;
  }
  const data = queryInstance?.state?.data;
  const error = queryInstance?.state?.error;

  // Consecutive errors (including 503 and network failures) use exponential backoff
  if (error) {
    const failureCount = Math.max(0, (queryInstance?.state?.errorUpdateCount || 1) - 1);
    return Math.min(3000 * 2 ** failureCount, 30000);
  }

  // Unloaded model or disconnected database polls every 3s
  if (
    data?.status === 'degraded' &&
    (data?.model?.mode === 'unloaded' || data?.database !== 'connected')
  ) {
    return 3000;
  }

  // Healthy normal state and stub mode poll every 10s
  return 10000;
}

export function useHealthStatus() {
  const query = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      return await getHealth();
    },
    refetchInterval: calculateHealthPollingInterval,
    refetchOnWindowFocus: true,
    retry: false, // let refetchInterval handle polling on failure
  });

  const healthData = query.data || query.error?.data || null;
  const isStub = healthData?.model?.mode === 'stub';
  const isUnloaded =
    healthData?.model?.mode === 'unloaded' ||
    (query.error?.status === 503 && healthData?.model?.mode === 'unloaded');
  const isDbDisconnected = healthData?.database === 'disconnected';
  const isUnreachable =
    Boolean(query.error && query.error.status === 0) ||
    Boolean(query.error && !healthData);

  return {
    ...query,
    healthData,
    isStub,
    isUnloaded,
    isDbDisconnected,
    isUnreachable,
  };
}
