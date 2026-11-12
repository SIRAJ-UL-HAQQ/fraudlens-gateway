export function parseStrategyData(strategyKey, rawData) {
  if (!rawData) return null;

  const labelMap = {
    none: 'No Resampling (Baseline)',
    class_weight: 'Class Weighting',
    smote_enn: 'SMOTE-ENN Hybrid',
  };

  const label = rawData.label || labelMap[strategyKey] || strategyKey;
  const isDeployed = strategyKey === 'class_weight' || String(label).toLowerCase().includes('deployed');

  const precision = rawData.precision ?? rawData.metrics_at_high?.precision ?? null;
  const recall = rawData.recall ?? rawData.metrics_at_high?.recall ?? null;
  const f1 = rawData.f1 ?? rawData.metrics_at_high?.f1 ?? null;
  const prAuc = rawData.pr_auc ?? rawData.test_pr_auc ?? rawData.metrics_at_high?.pr_auc ?? null;
  const stability = rawData.stability ?? rawData.shap_stability ?? rawData.stability_score ?? null;

  return {
    key: strategyKey,
    label,
    isDeployed,
    precision: precision !== null ? Number(precision) : null,
    recall: recall !== null ? Number(recall) : null,
    f1: f1 !== null ? Number(f1) : null,
    prAuc: prAuc !== null ? Number(prAuc) : null,
    stability: stability !== null ? Number(stability) : null,
  };
}
