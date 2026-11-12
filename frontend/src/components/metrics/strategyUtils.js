/**
 * Utility functions for parsing, validating, and normalizing ML strategy benchmark metrics.
 */

/**
 * Known non-strategy keys in research metrics report that represent summary metadata,
 * cross-strategy analyses, or statistical hypothesis tests rather than model candidates.
 */
const NON_STRATEGY_KEYS = new Set([
  'cross_strategy_shap_agreement',
  'informative_nature_of_unstable',
  'evaluation_metrics',
  'confusion_matrix',
]);

/**
 * Extracts a numeric SHAP stability rate/score from various representations:
 * 1. A primitive number (e.g. 0.8276 from fixtures or synthetic mocks)
 * 2. An object with `across_10_sets.fraud_cohort_stability.mean` (live ml/reports/metrics.json)
 * 3. An object with `fraud_cohort.primary_k5.stability_rate`
 * 4. An object with `primary_k5.stability_rate` (e.g. variant_A_reference)
 * 5. Other common nesting variants (`stability_rate`, `mean`, `score`)
 *
 * Always returns null if no valid finite number can be extracted.
 */
export function extractStabilityScore(rawVal) {
  if (rawVal === null || rawVal === undefined) return null;

  if (typeof rawVal === 'number' && !Number.isNaN(rawVal)) {
    return rawVal;
  }

  if (typeof rawVal === 'string' && rawVal.trim() !== '') {
    const num = Number(rawVal);
    return !Number.isNaN(num) ? num : null;
  }

  if (typeof rawVal === 'object') {
    // 1. Live across_10_sets fraud cohort stability mean
    const across10Mean = rawVal.across_10_sets?.fraud_cohort_stability?.mean;
    if (typeof across10Mean === 'number' && !Number.isNaN(across10Mean)) {
      return across10Mean;
    }

    // 2. Fraud cohort primary k5 stability rate
    const fraudCohortRate = rawVal.fraud_cohort?.primary_k5?.stability_rate;
    if (typeof fraudCohortRate === 'number' && !Number.isNaN(fraudCohortRate)) {
      return fraudCohortRate;
    }

    // 3. Variant A / direct primary k5 stability rate
    const primaryK5Rate = rawVal.primary_k5?.stability_rate;
    if (typeof primaryK5Rate === 'number' && !Number.isNaN(primaryK5Rate)) {
      return primaryK5Rate;
    }

    // 4. Direct property variants
    const directVal = rawVal.stability_rate ?? rawVal.mean ?? rawVal.score;
    if (typeof directVal === 'number' && !Number.isNaN(directVal)) {
      return directVal;
    }
  }

  return null;
}

/**
 * Safe numeric extractor ensuring finite numbers only.
 */
function toValidNumber(val) {
  if (val === null || val === undefined) return null;
  const num = Number(val);
  return typeof num === 'number' && !Number.isNaN(num) ? num : null;
}

export function parseStrategyData(strategyKey, rawData) {
  if (!rawData || typeof rawData !== 'object') return null;

  // Filter out non-strategy summary objects
  if (NON_STRATEGY_KEYS.has(strategyKey)) return null;

  const labelMap = {
    none: 'No Resampling (Baseline)',
    class_weight: 'Class Weighting (Deployed)',
    smote_enn: 'SMOTE-ENN Hybrid',
    logistic_regression: 'Logistic Regression Baseline',
    random_forest: 'Random Forest Baseline',
    rules_baseline: 'Rules Baseline',
    full_train_none: 'Full-Train Reference (None)',
    full_train_class_weight: 'Full-Train Reference (Class Weight)',
    variant_A_reference: 'Variant A Reference',
  };

  const label = rawData.label || labelMap[strategyKey] || strategyKey;
  const isDeployed =
    strategyKey === 'class_weight' ||
    String(label).toLowerCase().includes('deployed');

  // Extract core classification metrics supporting:
  // 1. Direct fields: rawData.precision, rawData.recall, etc.
  // 2. Nested under metrics_at_high (e.g. class_weight, none, smote_enn)
  // 3. Heuristic rules union under union_all_rules (e.g. rules_baseline)
  const precision =
    rawData.precision ??
    rawData.metrics_at_high?.precision ??
    rawData.union_all_rules?.precision ??
    null;

  const recall =
    rawData.recall ??
    rawData.metrics_at_high?.recall ??
    rawData.union_all_rules?.recall ??
    null;

  const f1 =
    rawData.f1 ??
    rawData.metrics_at_high?.f1 ??
    rawData.union_all_rules?.f1 ??
    null;

  const prAuc =
    rawData.pr_auc ??
    rawData.test_pr_auc ??
    rawData.metrics_at_high?.pr_auc ??
    rawData.union_all_rules?.pr_auc ??
    null;

  const rawStability =
    rawData.stability ??
    rawData.shap_stability ??
    rawData.stability_score ??
    null;

  const stability = extractStabilityScore(rawStability);

  const numPrecision = toValidNumber(precision);
  const numRecall = toValidNumber(recall);
  const numF1 = toValidNumber(f1);
  const numPrAuc = toValidNumber(prAuc);
  const numStability = toValidNumber(stability);

  // If entry lacks all performance and stability metrics, discard it
  const hasAnyMetric =
    numPrecision !== null ||
    numRecall !== null ||
    numF1 !== null ||
    numPrAuc !== null ||
    numStability !== null;

  if (!hasAnyMetric) {
    return null;
  }

  return {
    key: strategyKey,
    label,
    isDeployed,
    precision: numPrecision,
    recall: numRecall,
    f1: numF1,
    prAuc: numPrAuc,
    stability: numStability,
  };
}
