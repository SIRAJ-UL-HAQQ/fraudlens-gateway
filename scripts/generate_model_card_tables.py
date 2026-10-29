"""Script to generate Markdown tables for MODEL_CARD.md from canonical JSON reports.

Reads:
- ml/reports/eda.json (contains rules_performance_train_val on TRAIN and VALIDATION)
- ml/reports/metrics.json (contains model metrics, paths, and keys)
- ml/reports/stability_bootstrap_comparison.json (contains multi-set bootstrap metrics)

Usage:
    python scripts/generate_model_card_tables.py [--check] [--update]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def format_rules_table(split_data: dict[str, Any], split_name: str) -> str:
    """Generate Markdown table for a split's rule performance."""
    lines = [
        f"### {split_name.capitalize()} Split: Heuristic Rules Evaluation",
        "",
        "| Rule Name | Flag Rate | Fraud Recall | Precision | FPR | Total Triggers | TP | FP |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    rule_display_names = {
        "ORIGIN_ACCOUNT_DRAINED": "Origin Account Drained (>= 90%)",
        "LARGE_TRANSFER_AMOUNT": "Large Transfer Amount (>= 200k)",
        "DEST_BALANCE_INCONSISTENCY": "Destination Balance Inconsistency",
        "VERY_HIGH_AMOUNT_FOR_TYPE": "Very High Amount for Type",
        "union_drain_or_amount": "**Union: Drain OR Large Transfer (SRS FR-7)**",
        "union_all_rules": "**Union: All 4 Backend Rules**",
    }
    for r_key, label in rule_display_names.items():
        if r_key not in split_data:
            continue
        stats = split_data[r_key]
        fr = f"{stats['flag_rate'] * 100:.3f}%"
        rec = f"{stats['recall'] * 100:.2f}%"
        prec = f"{stats['precision'] * 100:.2f}%"
        fpr = f"{stats['fpr'] * 100:.3f}%"
        trig = f"{stats['total_triggers']:,}"
        tp = f"{stats['true_positives']:,}"
        fp = f"{stats['false_positives']:,}"
        lines.append(f"| {label} | {fr} | {rec} | {prec} | {fpr} | {trig} | {tp} | {fp} |")
    return "\n".join(lines)


def format_type_breakdown_table(split_data: dict[str, Any], split_name: str) -> str:
    """Generate Markdown table for per-type breakdown on a split."""
    lines = [
        f"### {split_name.capitalize()} Split: Rule Performance Breakdown by Transaction Type",
        "",
        "| Rule Name | Type | Volume | Frauds | Triggers | Flag Rate | Recall | Precision | FPR |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    rule_display_names = {
        "ORIGIN_ACCOUNT_DRAINED": "Origin Drained",
        "LARGE_TRANSFER_AMOUNT": "Large Transfer",
        "DEST_BALANCE_INCONSISTENCY": "Dest Inconsistency",
        "VERY_HIGH_AMOUNT_FOR_TYPE": "Very High Amount",
        "union_drain_or_amount": "Union (FR-7)",
        "union_all_rules": "Union (All Rules)",
    }
    for r_key, label in rule_display_names.items():
        if r_key not in split_data:
            continue
        by_type = split_data[r_key].get("by_type", {})
        for t_name in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]:
            t_stats = by_type.get(t_name)
            if not t_stats or t_stats["total_rows"] == 0:
                continue
            vol = f"{t_stats['total_rows']:,}"
            f_count = f"{t_stats['frauds']:,}"
            trig = f"{t_stats['triggers']:,}"
            fr = f"{t_stats['flag_rate'] * 100:.3f}%"
            rec = f"{t_stats['recall'] * 100:.2f}%" if t_stats['frauds'] > 0 else "-"
            prec = f"{t_stats['precision'] * 100:.2f}%" if t_stats['triggers'] > 0 else "-"
            fpr = f"{t_stats['fpr'] * 100:.3f}%"
            lines.append(f"| {label} | {t_name} | {vol} | {f_count} | {trig} | {fr} | {rec} | {prec} | {fpr} |")
    return "\n".join(lines)


def generate_all_rules_markdown(eda_path: Path) -> str:
    """Read eda.json and generate complete rules section markdown."""
    with open(eda_path, encoding="utf-8") as f:
        data = json.load(f)

    rules_perf = data.get("rules_performance_train_val", {})
    if not rules_perf:
        raise ValueError(f"No 'rules_performance_train_val' found in {eda_path}")

    train_tbl = format_rules_table(rules_perf["train"], "train")
    val_tbl = format_rules_table(rules_perf["validation"], "validation")
    val_type_tbl = format_type_breakdown_table(rules_perf["validation"], "validation")

    header = "## Heuristic Safety Rules Performance (Train & Validation Splits)"
    desc = (
        "Evaluated strictly on **TRAIN** (steps 1..520) and **VALIDATION** (steps 521..595) splits; "
        "test numbers are excluded from heuristic calibration. "
        "All rules implement the exact logic in `backend.app.services.rules`."
    )
    return f"{header}\n\n{desc}\n\n{train_tbl}\n\n{val_tbl}\n\n{val_type_tbl}\n"


def generate_strategy_comparison_table(metrics_path: Path) -> str:
    """Read metrics.json and generate Model Strategy Results Comparison table."""
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    strat_comp = data.get("strategy_comparison", {})
    lines = [
        "## Model & Strategy Results Comparison (Held-Out Test Split)",
        "",
        "Source files: `ml/reports/metrics.json` (key: `strategy_comparison`) and `ml/reports/comparison.csv`.",
        "",
        "| Model / Strategy | Val PR-AUC | Test PR-AUC | Recall @ High | Precision @ High | F1 @ High | FPR @ High | Stable Rate ($k=5$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    order = [
        ("class_weight", "class_weight (deployed)"),
        ("none", "none (10:1 undersampled baseline)"),
        ("smote_enn", "smote_enn"),
        ("logistic_regression", "Logistic Regression baseline"),
        ("random_forest", "Random Forest baseline"),
        ("rules_baseline", "Rules baseline (FR-7 union)"),
        ("full_train_none", "Full-train reference (none, 6.08M)"),
        ("full_train_class_weight", "Full-train reference (class_weight, 6.08M)"),
        ("variant_A_reference", "Variant A reference (15 features)"),
    ]

    for key, label in order:
        if key not in strat_comp:
            continue
        row = strat_comp[key]
        val_auc = f"{row.get('validation_pr_auc', '-'):.4f}" if isinstance(row.get('validation_pr_auc'), (float, int)) else str(row.get('validation_pr_auc', '-'))
        test_auc = f"{row.get('test_pr_auc', row.get('pr_auc', '-')):.4f}" if isinstance(row.get('test_pr_auc', row.get('pr_auc')), (float, int)) else "-"
        
        # Recall & Precision at High
        if "metrics_at_high" in row:
            m_high = row["metrics_at_high"]
            rec = f"{m_high.get('recall', 0.0) * 100:.2f}%"
            prec = f"{m_high.get('precision', 0.0) * 100:.2f}%"
            f1 = f"{m_high.get('f1', 0.0):.4f}"
            fpr = f"{m_high.get('fpr', 0.0) * 100:.4f}%"
        elif "recall" in row:
            rec = f"{row.get('recall', 0.0) * 100:.2f}%"
            prec = f"{row.get('precision', 0.0) * 100:.2f}%"
            f1 = f"{row.get('f1', 0.0):.4f}"
            fpr = f"{row.get('fpr', 0.0) * 100:.4f}%"
        else:
            rec = prec = f1 = fpr = "-"

        # Stability rate
        stab = row.get("shap_stability", {})
        if "fraud_cohort" in stab and "primary_k5" in stab["fraud_cohort"]:
            sr = f"{stab['fraud_cohort']['primary_k5'].get('stability_rate', 0.0) * 100:.2f}%"
        elif "primary_k5" in stab:
            sr = f"{stab['primary_k5'].get('stability_rate', 0.0) * 100:.2f}%"
        else:
            sr = "-"

        lines.append(f"| {label} | {val_auc} | {test_auc} | {rec} | {prec} | {f1} | {fpr} | {sr} |")

    return "\n".join(lines)


def generate_gateway_evaluation_markdown(gateway_json_path: Path) -> str:
    """Read gateway_evaluation.json and generate complete Gateway Evaluation section markdown."""
    with open(gateway_json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    val_data = data["variant_evaluations"]["validation"]
    test_data = data["variant_evaluations"]["test"]
    sr = data["selection_rule_validation"]
    unstable_info = data["informative_analysis_of_unstable"]
    rule_dist = data["rule_only_frauds_score_distribution"]

    lines = [
        "## Full Gateway Decision Evaluation & Variant Selection (Step 3)",
        "",
        f"Source file: `ml/reports/gateway_evaluation.json`. Evaluates the deployed `{meta['model_version']}` artifact "
        f"($t_{{\\text{{review}}}} = {meta['threshold_review']:.6f}, t_{{\\text{{high}}}} = {meta['threshold_high']:.6f}$, "
        f"$k={meta['k']}, \\text{{min\\_overlap}}={meta['min_overlap']:.2f}$) using read-only backend `route_decision` "
        f"and 5 replicate models.",
        "",
        "### Validation Split: Routing Variants & Selection Rule (75 Simulated Hours, Budget $\\le 3,000$ Alerts)",
        "",
        "| Routing Variant | Total Review Queue | Queue Rate (/hr) | High-Only Recall | Combined Recall (High+Review) | 95% Wilson CI | Legit in Review (%) | Selection Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    variant_names = {
        "variant_A": "Variant A (Current Backend: Unconstrained Rules)",
        "variant_B": "Variant B (Score-Gated Rules: Rule Never Routes Alone)",
        "variant_C": "Variant C (Train-Recalibrated Rules: $\\le 0.5\\%$ cap)",
        "variant_D": "Variant D (Pure ML Gateway: No Rules)",
        "variant_E": "Variant E (Exploratory Post-Hoc: Tuned Floor $f=0.75$)",
    }

    for v_key in ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]:
        v_res = val_data[v_key]
        m = v_res["metrics"]
        q_cnt = f"{m['analyst_review_queue_total']:,}"
        q_rate = f"{m['analyst_review_queue_rate_per_hour']:.2f}"
        ho_rec = f"{m['high_only_recall_pct']:.2f}%"
        comb_rec = f"{m['combined_recall_pct']:.2f}%"
        ci = f"[{m['combined_recall_wilson_95ci_pct'][0]:.2f}%, {m['combined_recall_wilson_95ci_pct'][1]:.2f}%]"
        legit_pct = f"{m['legit_in_review_rate_pct']:.3f}%"

        if v_key == "variant_A":
            status = "Ineligible (Queue $> 3,000$)"
        elif v_key == sr["smallest_queue_variant"]:
            status = "**Selected (Smallest Queue within 1pp)**"
        elif v_key in [x["variant"] for x in sr.get("qualifying_variants_within_1pp", [])]:
            status = "Eligible (Qualifying within 1pp)"
        else:
            status = "Exploratory (Post-hoc tuned)"
        lines.append(f"| {variant_names[v_key]} | {q_cnt} | {q_rate} | {ho_rec} | {comb_rec} | {ci} | {legit_pct} | {status} |")

    lines.extend([
        "",
        "### Held-Out Test Split: Full Routing Variant Comparison (148 Simulated Hours, 118,146 Rows, 1,642 Frauds)",
        "",
        "| Routing Variant | High Risk Count | High Risk Recall | Review Queue Count | Queue Rate (/hr) | Combined Recall (High+Review) | 95% Wilson CI | Legit in Review (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for v_key in ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]:
        t_res = test_data[v_key]
        m = t_res["metrics"]
        d = t_res["decisions"]
        hr_cnt = f"{d['high_risk']['count']:,}"
        ho_rec = f"{m['high_only_recall_pct']:.2f}%"
        q_cnt = f"{m['analyst_review_queue_total']:,}"
        q_rate = f"{m['analyst_review_queue_rate_per_hour']:.2f}"
        comb_rec = f"{m['combined_recall_pct']:.2f}%"
        ci = f"[{m['combined_recall_wilson_95ci_pct'][0]:.2f}%, {m['combined_recall_wilson_95ci_pct'][1]:.2f}%]"
        legit_pct = f"{m['legit_in_review_rate_pct']:.3f}%"
        lines.append(f"| {variant_names[v_key]} | {hr_cnt} | {ho_rec} | {q_cnt} | {q_rate} | {comb_rec} | {ci} | {legit_pct} |")

    lines.extend([
        "",
        "### Held-Out Test Split: Analyst Review Queue Breakdown by Cause",
        "",
        "| Routing Variant | Score Band ($t_{\\text{rev}} \\le s < t_{\\text{high}}$) | Unstable Explanation | Rule Flag Alone | Total Queue | Queue Frauds (TP) | Queue Legit (FP) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for v_key in ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]:
        qc = test_data[v_key]["queue_by_cause"]
        sb = f"{qc['score_band']['total']:,} ({qc['score_band']['frauds']}/{qc['score_band']['legitimate']})"
        ue = f"{qc['unstable_explanation']['total']:,} ({qc['unstable_explanation']['frauds']}/{qc['unstable_explanation']['legitimate']})"
        rf = f"{qc['rule_flag']['total']:,} ({qc['rule_flag']['frauds']}/{qc['rule_flag']['legitimate']})"
        tq = f"{qc['total_queue']['total']:,}"
        q_tp = f"{qc['total_queue']['frauds']:,}"
        q_fp = f"{qc['total_queue']['legitimate']:,}"
        lines.append(f"| {variant_names[v_key]} | {sb} | {ue} | {rf} | {tq} | {q_tp} | {q_fp} |")

    lines.extend([
        "",
        "### Candidate High Risk Flow Table (Validation & Held-Out Test Splits)",
        "",
        "| Split | Candidate High Total | Routed to High Risk (Stable) | Routed to Review (Unstable) | Stability Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for s_name in ["validation", "test"]:
        cf = data["variant_evaluations"][s_name]["variant_B"]["candidate_high_flow"]
        tot = f"{cf['candidate_high_total']['total']:,} ({cf['candidate_high_total']['frauds']} F, {cf['candidate_high_total']['legitimate']} L)"
        st = f"{cf['routed_high_risk_stable']['total']:,} ({cf['routed_high_risk_stable']['frauds']} F, {cf['routed_high_risk_stable']['legitimate']} L)"
        un = f"{cf['routed_analyst_review_unstable']['total']:,} ({cf['routed_analyst_review_unstable']['frauds']} F, {cf['routed_analyst_review_unstable']['legitimate']} L)"
        rate = f"{cf['routed_high_risk_stable']['total'] / cf['candidate_high_total']['total'] * 100.0:.2f}%"
        lines.append(f"| {s_name.capitalize()} Split | {tot} | {st} | {un} | {rate} |")

    lines.extend([
        "",
        "### Informative Analysis of Unstable Explanations (Candidate High Risk Rows)",
        "",
        "| Split | Cohort | $N$ | Frauds (TP) | Legit (FP) | Precision (95% Wilson CI) | FP Share (95% Wilson CI) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s_name in ["validation", "test"]:
        info = unstable_info[s_name]
        st = info["stable_cohort"]
        un = info["unstable_cohort"]
        st_p_ci = f"[{st['precision_wilson_95ci'][0]*100:.2f}%, {st['precision_wilson_95ci'][1]*100:.2f}%]"
        st_fp_ci = f"[{st['false_positive_share_wilson_95ci_pct'][0]:.2f}%, {st['false_positive_share_wilson_95ci_pct'][1]:.2f}%]"
        un_p_ci = f"[{un['precision_wilson_95ci'][0]*100:.2f}%, {un['precision_wilson_95ci'][1]*100:.2f}%]"
        un_fp_ci = f"[{un['false_positive_share_wilson_95ci_pct'][0]:.2f}%, {un['false_positive_share_wilson_95ci_pct'][1]:.2f}%]"

        lines.append(f"| {s_name.capitalize()} | Stable ($k=5$) | {st['n']:,} | {st['true_positives']:,} | {st['false_positives']:,} | {st['precision_pct']:.2f}% {st_p_ci} | {st['false_positive_share_pct']:.2f}% {st_fp_ci} |")
        lines.append(f"| {s_name.capitalize()} | Unstable ($k=5$) | {un['n']:,} | {un['true_positives']:,} | {un['false_positives']:,} | {un['precision_pct']:.2f}% {un_p_ci} | {un['false_positive_share_pct']:.2f}% {un_fp_ci} |")

    t_info = unstable_info["test"]
    v_info = unstable_info["validation"]
    lines.extend([
        "",
        "#### Pre-Specified Conclusion on Informative Nature of Unstable:",
        f"- **Validation Split**: AUROC = {v_info['auroc_mean_overlap_predicting_fp']['auroc']:.4f} "
        f"(95% bootstrap CI: [{v_info['auroc_mean_overlap_predicting_fp']['bootstrap_95ci'][0]:.4f}, {v_info['auroc_mean_overlap_predicting_fp']['bootstrap_95ci'][1]:.4f}]). "
        f"Validation meets both conditions (AUROC excludes 0.50, and Unstable FP share lower CI > Stable upper CI).",
        f"- **Held-Out Test Split**: AUROC = {t_info['auroc_mean_overlap_predicting_fp']['auroc']:.4f} "
        f"(95% bootstrap CI: [{t_info['auroc_mean_overlap_predicting_fp']['bootstrap_95ci'][0]:.4f}, {t_info['auroc_mean_overlap_predicting_fp']['bootstrap_95ci'][1]:.4f}]). "
        f"Test fails both conditions (AUROC CI includes chance 0.50, and Unstable FP share CI [{t_info['unstable_cohort']['false_positive_share_wilson_95ci_pct'][0]:.2f}%, {t_info['unstable_cohort']['false_positive_share_wilson_95ci_pct'][1]:.2f}%] overlaps Stable [{t_info['stable_cohort']['false_positive_share_wilson_95ci_pct'][0]:.2f}%, {t_info['stable_cohort']['false_positive_share_wilson_95ci_pct'][1]:.2f}%]).",
        f"- **Scientific Evaluation Note**: `docs/analysis-plan.md` does not pre-specify whether Validation or Held-Out Test is the governing split. However, because generalizability to unseen data is required, the pre-specified conclusion on test is: **\"{t_info['pre_specified_conclusion']['verdict']}\"**.",
        "",
        "#### Score-Margin Control vs Unstable Explanation Re-routing:",
        f"- **Restricted Score-Margin Control** (strictly candidate rows with score $\\ge t_{{\\text{{high}}}}$; $N=284$ rows): re-routes {t_info['score_margin_control_restricted_thigh']['false_positives']} FPs "
        f"({t_info['score_margin_control_restricted_thigh']['false_positive_share_pct']:.2f}% FP share) vs {t_info['score_margin_control_restricted_thigh']['unstable_fp_within_thigh']} FPs "
        f"for Unstable within $s \\ge t_{{\\text{{high}}}}$ (+{t_info['score_margin_control_restricted_thigh']['fp_gain_vs_unstable']} FPs caught, with {t_info['score_margin_control_restricted_thigh']['fp_gain_vs_unstable']} fewer frauds delayed).",
        f"- **Unrestricted Score-Margin Control** (all candidate rows including rule-boosted rows below $t_{{\\text{{high}}}}$; $N=299$ rows): re-routes {t_info['score_margin_control_unrestricted_labeled']['false_positives']} FPs "
        f"({t_info['score_margin_control_unrestricted_labeled']['false_positive_share_pct']:.2f}% FP share). *Labeled unrestricted: includes lower-scoring rule-boosted band rows with higher FP prevalence.*",
        "",
        "### Exploratory Post-Hoc Variant E: Score Floor Grid Analysis",
        "",
        "| Score Floor $f$ | Validation Queue | Val Queue Rate (/hr) | Val Combined Recall | Val Legit Share (%) | Test Queue | Test Queue Rate (/hr) | Test Combined Recall | Test Legit Share (%) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    ve_grid = data.get("variant_E_exploratory_grid", {}).get("grid", [])
    for row in ve_grid:
        lines.append(
            f"| $f = {row['floor_f']:.2f}$ | {row['validation']['total_queue']:,} | {row['validation']['queue_rate_per_hour']:.2f} | "
            f"{row['validation']['combined_recall_pct']:.2f}% | {row['validation']['legit_rate_pct']:.3f}% | "
            f"{row['test']['total_queue']:,} | {row['test']['queue_rate_per_hour']:.2f} | "
            f"{row['test']['combined_recall_pct']:.2f}% | {row['test']['legit_rate_pct']:.3f}% |"
        )

    lines.extend([
        "",
        "### ML Score Distribution of Frauds Flagged Only by Rules ($s < t_{\\text{review}}$)",
        "",
        "| Split | Fraud Count | Min Score | 25th % | Median Score | 75th % | Max Score | Mean Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Validation | {rule_dist['validation']['count']:,} | {rule_dist['validation']['min']:.4f} | {rule_dist['validation']['p25']:.4f} | {rule_dist['validation']['median']:.4f} | {rule_dist['validation']['p75']:.4f} | {rule_dist['validation']['max']:.4f} | {rule_dist['validation']['mean']:.4f} |",
        f"| Held-Out Test | {rule_dist['test']['count']:,} | {rule_dist['test']['min']:.4f} | {rule_dist['test']['p25']:.4f} | {rule_dist['test']['median']:.4f} | {rule_dist['test']['p75']:.4f} | {rule_dist['test']['max']:.4f} | {rule_dist['test']['mean']:.4f} |",
    ])

    return "\n".join(lines)


def update_model_card(
    model_card_path: Path,
    eda_path: Path,
    metrics_path: Path,
    gateway_path: Path | None = None,
) -> bool:
    """Inject/update the generated tables into MODEL_CARD.md."""
    new_rules_md = generate_all_rules_markdown(eda_path)
    content = model_card_path.read_text(encoding="utf-8")
    updated = content

    # 1. Update rules section
    pattern_rules = re.compile(
        r"## Heuristic Safety Rules Performance \(Train & Validation Splits\).*?(?=\n## |\Z)",
        re.DOTALL,
    )
    if pattern_rules.search(updated):
        updated = pattern_rules.sub(lambda _: new_rules_md.strip(), updated)
    else:
        updated = updated + "\n\n" + new_rules_md

    # 2. Update gateway section if gateway_path provided
    if gateway_path and gateway_path.exists():
        new_gw_md = generate_gateway_evaluation_markdown(gateway_path)
        pattern_gw = re.compile(
            r"## (?:Full Gateway Decision Evaluation.*?|Gateway Routing Evaluation.*?)(?=\n## |\Z)",
            re.DOTALL,
        )
        if pattern_gw.search(updated):
            updated = pattern_gw.sub(lambda _: new_gw_md.strip(), updated)
        else:
            updated = updated + "\n\n" + new_gw_md

    if updated != content:
        model_card_path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and sync MODEL_CARD markdown tables from JSON.")
    parser.add_argument("--eda-path", default="ml/reports/eda.json", help="Path to eda.json")
    parser.add_argument("--metrics-path", default="ml/reports/metrics.json", help="Path to metrics.json")
    parser.add_argument("--gateway-path", default="ml/reports/gateway_evaluation.json", help="Path to gateway_evaluation.json")
    parser.add_argument("--model-card", default="ml/reports/MODEL_CARD.md", help="Path to MODEL_CARD.md")
    parser.add_argument("--update", action="store_true", help="Update MODEL_CARD.md in place")
    parser.add_argument("--check", action="store_true", help="Check whether MODEL_CARD.md tables are in sync")
    args = parser.parse_args()

    eda_p = Path(args.eda_path)
    metrics_p = Path(args.metrics_path)
    gw_p = Path(args.gateway_path)
    card_p = Path(args.model_card)

    if not eda_p.exists():
        print(f"Error: {eda_p} not found.", file=sys.stderr)
        sys.exit(1)
    if not card_p.exists():
        print(f"Error: {card_p} not found.", file=sys.stderr)
        sys.exit(1)

    expected_rules_md = generate_all_rules_markdown(eda_p).strip()
    has_gw = gw_p.exists()
    expected_gw_md = generate_gateway_evaluation_markdown(gw_p).strip() if has_gw else ""

    if args.check:
        actual_content = card_p.read_text(encoding="utf-8")
        if expected_rules_md not in actual_content:
            print("ERROR: MODEL_CARD.md rules tables do not match JSON data!", file=sys.stderr)
            sys.exit(1)
        if has_gw and expected_gw_md not in actual_content:
            print("ERROR: MODEL_CARD.md gateway evaluation tables do not match JSON data!", file=sys.stderr)
            sys.exit(1)
        print("SUCCESS: MODEL_CARD.md tables are in sync with JSON reports.")
        sys.exit(0)

    if args.update or not (args.check or args.update):
        changed = update_model_card(card_p, eda_p, metrics_p, gateway_path=gw_p if has_gw else None)
        if changed:
            print(f"Updated {card_p} with latest tables from canonical JSON reports.")
        else:
            print(f"{card_p} is already up-to-date.")


if __name__ == "__main__":
    main()
