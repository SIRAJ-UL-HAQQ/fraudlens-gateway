import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import * as client from '../api/client';
import OperationalMetricsCard from '../components/metrics/OperationalMetricsCard';
import DecisionDistributionChart from '../components/metrics/DecisionDistributionChart';
import ConfusionMatrixGrid from '../components/metrics/ConfusionMatrixGrid';
import StrategyComparisonTable from '../components/metrics/StrategyComparisonTable';
import StrategyComparisonChart from '../components/metrics/StrategyComparisonChart';
import MetricsView from '../pages/MetricsView';

describe('OperationalMetricsCard', () => {
  const mockMetrics = {
    total_transactions: 1450,
    explanation_stability_ratio: 0.88,
    average_risk_score: 0.14,
    latency_p50_ms: 13.4,
    latency_p95_ms: 24.8,
  };

  it('renders KPI values accurately', () => {
    render(<OperationalMetricsCard metrics={mockMetrics} />);
    expect(screen.getByText('1,450')).toBeInTheDocument();
    expect(screen.getByText('88.0%')).toBeInTheDocument();
    expect(screen.getByText('0.140')).toBeInTheDocument();
    expect(screen.getByText('13.4 ms')).toBeInTheDocument();
    expect(screen.getByText('24.8 ms')).toBeInTheDocument();
  });

  it('shows tooltip on latency trigger button interaction', () => {
    render(<OperationalMetricsCard metrics={mockMetrics} />);
    const trigger = screen.getByLabelText('Latency calculation details');
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();

    fireEvent.mouseEnter(trigger);
    expect(screen.getByRole('tooltip')).toBeInTheDocument();
    expect(screen.getByRole('tooltip')).toHaveTextContent('Server-side scoring + explanation time');

    fireEvent.mouseLeave(trigger);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });
});

describe('DecisionDistributionChart', () => {
  const distribution = {
    'Low Risk': 1280,
    'Analyst Review': 75,
    'High Risk': 95,
  };

  it('renders accessible SVG with correct titles and counts', () => {
    render(<DecisionDistributionChart distribution={distribution} />);
    const svg = screen.getByRole('img', { name: /decision distribution chart/i });
    expect(svg).toBeInTheDocument();
    expect(screen.getByText('1,280')).toBeInTheDocument();
    expect(screen.getByText('75')).toBeInTheDocument();
    expect(screen.getByText('95')).toBeInTheDocument();
    expect(screen.getByText('Total: 1,450')).toBeInTheDocument();
  });

  it('handles empty distribution without crashing', () => {
    render(<DecisionDistributionChart distribution={{}} />);
    expect(screen.getByText('Total: 0')).toBeInTheDocument();
  });
});

describe('ConfusionMatrixGrid', () => {
  const matrix = {
    true_positives: 1551,
    false_positives: 94,
    true_negatives: 116410,
    false_negatives: 91,
  };

  it('renders 2x2 grid with formatted counts and rates', () => {
    render(<ConfusionMatrixGrid matrix={matrix} />);
    expect(screen.getByTestId('matrix-tp')).toHaveTextContent('1,551');
    expect(screen.getByTestId('matrix-fp')).toHaveTextContent('94');
    expect(screen.getByTestId('matrix-tn')).toHaveTextContent('116,410');
    expect(screen.getByTestId('matrix-fn')).toHaveTextContent('91');
    expect(screen.getByText(/Total Evaluated: 118,146/i)).toBeInTheDocument();
  });

  it('renders fallback notice when matrix is null', () => {
    render(<ConfusionMatrixGrid matrix={null} />);
    expect(screen.getByTestId('confusion-matrix-fallback')).toBeInTheDocument();
    expect(screen.getByText(/Confusion matrix is not included in the current ML benchmark payload/i)).toBeInTheDocument();
  });
});

describe('StrategyComparisonTable & StrategyComparisonChart', () => {
  const strategyComparison = {
    none: { precision: 0.88, recall: 0.62, f1: 0.73, pr_auc: 0.74, stability: 0.71 },
    class_weight: { precision: 0.91, recall: 0.84, f1: 0.87, pr_auc: 0.89, stability: 0.88 },
    smote_enn: { precision: 0.85, recall: 0.87, f1: 0.86, pr_auc: 0.87, stability: 0.80 },
  };

  it('renders comparison rows and labels class_weight as Deployed', () => {
    render(<StrategyComparisonTable strategyComparison={strategyComparison} />);
    expect(screen.getByTestId('strategy-row-none')).toHaveTextContent('No Resampling (Baseline)');
    expect(screen.getByTestId('strategy-row-class_weight')).toHaveTextContent('Class Weighting');
    expect(screen.getByText('Deployed')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-row-smote_enn')).toHaveTextContent('SMOTE-ENN Hybrid');
  });

  it('renders accessible SVG chart with strategy bars', () => {
    render(<StrategyComparisonChart strategyComparison={strategyComparison} />);
    const chart = screen.getByRole('img', { name: /strategy performance comparison/i });
    expect(chart).toBeInTheDocument();
    expect(screen.getByTestId('bar-precision-class_weight')).toBeInTheDocument();
    expect(screen.getByTestId('bar-recall-smote_enn')).toBeInTheDocument();
  });
});

describe('MetricsView Integration', () => {
  const availableResponse = {
    total_transactions: 1450,
    decision_distribution: { 'Low Risk': 1280, 'High Risk': 95, 'Analyst Review': 75 },
    explanation_stability_ratio: 0.88,
    average_risk_score: 0.14,
    latency_p50_ms: 13.4,
    latency_p95_ms: 24.8,
    ml_metrics_status: 'available',
    ml_metrics_reason: null,
    ml_metrics: {
      strategy_comparison: {
        none: { precision: 0.88, recall: 0.62, f1: 0.73, pr_auc: 0.74, stability: 0.71 },
        class_weight: { precision: 0.91, recall: 0.84, f1: 0.87, pr_auc: 0.89, stability: 0.88 },
        smote_enn: { precision: 0.85, recall: 0.87, f1: 0.86, pr_auc: 0.87, stability: 0.80 },
      },
      evaluation_metrics: { precision: 0.91, recall: 0.84, f1: 0.87, pr_auc: 0.89, mcc: 0.86 },
      confusion_matrix: { true_positives: 1551, false_positives: 94, true_negatives: 116410, false_negatives: 91 },
    },
  };

  const unavailableResponse = {
    total_transactions: 1450,
    decision_distribution: { 'Low Risk': 1280, 'High Risk': 95, 'Analyst Review': 75 },
    explanation_stability_ratio: 0.88,
    average_risk_score: 0.14,
    latency_p50_ms: 13.4,
    latency_p95_ms: 24.8,
    ml_metrics_status: 'unavailable',
    ml_metrics_reason: 'ML benchmark report not found at ml/reports/metrics.json.',
    ml_metrics: null,
  };

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders full operational and ML sections when status is available', async () => {
    vi.spyOn(client, 'getMetrics').mockResolvedValue(availableResponse);

    render(<MetricsView />);
    await waitFor(() => {
      expect(screen.getByTestId('operational-metrics')).toBeInTheDocument();
    });

    expect(screen.getByTestId('decision-distribution-card')).toBeInTheDocument();
    expect(screen.getByTestId('ml-metrics-section')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-comparison-table')).toBeInTheDocument();
    expect(screen.getByTestId('confusion-matrix-grid')).toBeInTheDocument();
    expect(screen.queryByTestId('ml-metrics-unavailable-banner')).not.toBeInTheDocument();
  });

  it('renders operational metrics and unavailable notice without ML sections when status is unavailable', async () => {
    vi.spyOn(client, 'getMetrics').mockResolvedValue(unavailableResponse);

    render(<MetricsView />);
    await waitFor(() => {
      expect(screen.getByTestId('operational-metrics')).toBeInTheDocument();
    });

    expect(screen.getByTestId('decision-distribution-card')).toBeInTheDocument();
    expect(screen.getByTestId('ml-metrics-unavailable-banner')).toBeInTheDocument();
    expect(screen.getByText('ML benchmark report not found at ml/reports/metrics.json.')).toBeInTheDocument();
    expect(screen.queryByTestId('ml-metrics-section')).not.toBeInTheDocument();
    expect(screen.queryByTestId('confusion-matrix-grid')).not.toBeInTheDocument();
  });

  it('renders error message when getMetrics rejects', async () => {
    vi.spyOn(client, 'getMetrics').mockRejectedValue(new Error('Network error loading metrics'));

    render(<MetricsView />);
    await waitFor(() => {
      expect(screen.getByText('Network error loading metrics')).toBeInTheDocument();
    });
  });

  it('handles nested shap_stability objects and filters metadata keys without rendering NaN% or non-strategy rows', async () => {
    const liveLikeResponse = {
      total_transactions: 5007,
      decision_distribution: { 'Low Risk': 3973, 'High Risk': 0, 'Analyst Review': 1034 },
      explanation_stability_ratio: 0.836,
      average_risk_score: 0.18,
      latency_p50_ms: 12.5,
      latency_p95_ms: 22.1,
      ml_metrics_status: 'available',
      ml_metrics_reason: null,
      ml_metrics: {
        strategy_comparison: {
          class_weight: {
            label: 'class_weight (deployed)',
            metrics_at_high: {
              precision: 0.9429,
              recall: 0.9446,
              f1: 0.9437,
              pr_auc: 0.9907,
              confusion_matrix: {
                true_negatives: 116410,
                false_positives: 94,
                false_negatives: 91,
                true_positives: 1551,
              },
            },
            shap_stability: {
              fraud_cohort: {
                primary_k5: {
                  stability_rate: 0.8276,
                },
              },
              across_10_sets: {
                fraud_cohort_stability: {
                  mean: 0.8359,
                },
              },
            },
          },
          none: {
            label: 'none (10:1 undersampled baseline)',
            metrics_at_high: {
              precision: 0.95,
              recall: 0.9367,
              f1: 0.9433,
              pr_auc: 0.99,
            },
            shap_stability: {
              across_10_sets: {
                fraud_cohort_stability: {
                  mean: 0.8149,
                },
              },
            },
          },
          smote_enn: {
            label: 'smote_enn',
            metrics_at_high: {
              precision: 0.9538,
              recall: 0.905,
              f1: 0.9287,
              pr_auc: 0.9853,
            },
            shap_stability: {
              across_10_sets: {
                fraud_cohort_stability: {
                  mean: 0.8623,
                },
              },
            },
          },
          rules_baseline: {
            label: 'Rules baseline (backend rule function)',
            union_all_rules: {
              precision: 0.0445,
              recall: 0.9951,
              f1: 0.0851,
              pr_auc: 0.0443,
            },
          },
          variant_A_reference: {
            label: 'Variant A reference (15 features with balance errors/deltas)',
            precision: 0.9832,
            recall: 0.9994,
            f1: 0.9912,
            pr_auc: 0.9999,
            shap_stability: {
              primary_k5: {
                stability_rate: 0.8307,
              },
            },
          },
          cross_strategy_shap_agreement: {
            n_samples: 1699,
            k: 5,
            chance_baseline: 0.4545,
            pairwise_comparisons: {
              class_weight_vs_none: { mean_top_k_overlap: 0.8281 },
            },
          },
          informative_nature_of_unstable: {
            chi2_contingency: { p_value: 0.001 },
          },
        },
      },
    };

    vi.spyOn(client, 'getMetrics').mockResolvedValue(liveLikeResponse);

    render(<MetricsView />);
    await waitFor(() => {
      expect(screen.getByTestId('strategy-comparison-table')).toBeInTheDocument();
    });

    // Verify valid strategies render correctly
    expect(screen.getByTestId('strategy-row-class_weight')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-row-none')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-row-smote_enn')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-row-rules_baseline')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-row-variant_A_reference')).toBeInTheDocument();

    // Verify non-strategy metadata keys are completely filtered out
    expect(screen.queryByTestId('strategy-row-cross_strategy_shap_agreement')).not.toBeInTheDocument();
    expect(screen.queryByTestId('strategy-row-informative_nature_of_unstable')).not.toBeInTheDocument();

    // Check specific rendered stability percentages inside their strategy rows
    const classWeightRow = screen.getByTestId('strategy-row-class_weight');
    expect(within(classWeightRow).getByText('83.6%')).toBeInTheDocument();

    const noneRow = screen.getByTestId('strategy-row-none');
    expect(within(noneRow).getByText('81.5%')).toBeInTheDocument();

    const smoteRow = screen.getByTestId('strategy-row-smote_enn');
    expect(within(smoteRow).getByText('86.2%')).toBeInTheDocument();

    const variantARow = screen.getByTestId('strategy-row-variant_A_reference');
    expect(within(variantARow).getByText('83.1%')).toBeInTheDocument();

    const rulesRow = screen.getByTestId('strategy-row-rules_baseline');
    expect(within(rulesRow).getByText('4.5%')).toBeInTheDocument(); // Precision
    expect(within(rulesRow).getByText('99.5%')).toBeInTheDocument(); // Recall

    // Assert that zero NaN% strings appear anywhere in the document body
    expect(document.body.textContent).not.toContain('NaN%');
    expect(document.body.textContent).not.toContain('NaN');
  });
});
