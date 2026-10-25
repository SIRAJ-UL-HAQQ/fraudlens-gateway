"""Script to generate Markdown tables for MODEL_CARD.md from canonical JSON reports.

Reads:
- ml/reports/eda.json (contains rules_performance_train_val on TRAIN and VALIDATION)
- ml/reports/metrics.json (contains model metrics)

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


def update_model_card(model_card_path: Path, eda_path: Path) -> bool:
    """Inject/update the generated rules section into MODEL_CARD.md."""
    new_rules_md = generate_all_rules_markdown(eda_path)
    content = model_card_path.read_text(encoding="utf-8")

    # Match from section_header to next level-2 header
    pattern = re.compile(
        r"## Heuristic Safety Rules Performance \(Train & Validation Splits\).*?(?=\n## |\Z)",
        re.DOTALL,
    )
    if pattern.search(content):
        updated_content = pattern.sub(new_rules_md.strip(), content)
    else:
        # Append before Limitations or at end
        lim_pattern = re.compile(r"(## Limitations & Operational Constraints)")
        if lim_pattern.search(content):
            updated_content = lim_pattern.sub(f"{new_rules_md}\n\n\\1", content)
        else:
            updated_content = content + "\n\n" + new_rules_md

    if updated_content != content:
        model_card_path.write_text(updated_content, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and sync MODEL_CARD markdown tables from JSON.")
    parser.add_argument("--eda-path", default="ml/reports/eda.json", help="Path to eda.json")
    parser.add_argument("--model-card", default="ml/reports/MODEL_CARD.md", help="Path to MODEL_CARD.md")
    parser.add_argument("--update", action="store_true", help="Update MODEL_CARD.md in place")
    parser.add_argument("--check", action="store_true", help="Check whether MODEL_CARD.md tables are in sync")
    args = parser.parse_args()

    eda_p = Path(args.eda_path)
    card_p = Path(args.model_card)

    if not eda_p.exists():
        print(f"Error: {eda_p} not found.", file=sys.stderr)
        sys.exit(1)
    if not card_p.exists():
        print(f"Error: {card_p} not found.", file=sys.stderr)
        sys.exit(1)

    expected_md = generate_all_rules_markdown(eda_p).strip()

    if args.check:
        actual_content = card_p.read_text(encoding="utf-8")
        if expected_md not in actual_content:
            print("ERROR: MODEL_CARD.md rules tables do not match JSON data!", file=sys.stderr)
            sys.exit(1)
        print("SUCCESS: MODEL_CARD.md tables are in sync with JSON reports.")
        sys.exit(0)

    if args.update or not (args.check or args.update):
        changed = update_model_card(card_p, eda_p)
        if changed:
            print(f"Updated {card_p} with latest rules tables from {eda_p}.")
        else:
            print(f"{card_p} is already up-to-date.")


if __name__ == "__main__":
    main()
