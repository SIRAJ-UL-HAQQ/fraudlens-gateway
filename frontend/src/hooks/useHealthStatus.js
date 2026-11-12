import { useQuery } from '@tanstack/react-query';
import { getHealth } from '../api/client';

export function useHealthStatus() {
  const query = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      return await getHealth();
    },
    refetchInterval: (queryInstance) => {
      if (typeof document !== 'undefined' && document.hidden) {
        return false;
      }
      const data = queryInstance.state.data;
      const error = queryInstance.state.error;

      // Degraded or error states poll faster (3s) to detect recovery
      if (error || data?.status === 'degraded' || data?.model?.mode === 'stub') {
        return 3000;
      }
      // Healthy normal state polls every 10s
      return 10000;
    },
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
