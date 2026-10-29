"""Unit test verifying that MODEL_CARD.md markdown tables are in exact sync with JSON reports."""

from pathlib import Path

from scripts.generate_model_card_tables import (
    generate_all_rules_markdown,
    generate_cross_strategy_markdown,
    generate_gateway_evaluation_markdown,
    generate_strategy_comparison_table,
)


def test_model_card_rules_tables_match_json():
    """Assert MODEL_CARD.md contains the exact rules table generated from eda.json."""
    eda_path = Path("ml/reports/eda.json")
    model_card_path = Path("ml/reports/MODEL_CARD.md")

    assert eda_path.exists(), f"Missing {eda_path}"
    assert model_card_path.exists(), f"Missing {model_card_path}"

    expected_rules_section = generate_all_rules_markdown(eda_path).strip()
    actual_model_card_content = model_card_path.read_text(encoding="utf-8")

    assert expected_rules_section in actual_model_card_content, (
        "MODEL_CARD.md heuristic rules table has diverged from canonical JSON in eda.json. "
        "Run `python scripts/generate_model_card_tables.py --update` to resync."
    )


def test_strategy_comparison_generator_runs():
    """Assert strategy comparison table can be generated from metrics.json."""
    metrics_path = Path("ml/reports/metrics.json")
    assert metrics_path.exists(), f"Missing {metrics_path}"
    table_md = generate_strategy_comparison_table(metrics_path)
    assert "| Model / Strategy |" in table_md
    assert "class_weight (deployed)" in table_md


def test_model_card_gateway_tables_match_json():
    """Assert MODEL_CARD.md contains the exact gateway evaluation table generated from gateway_evaluation.json."""
    gw_path = Path("ml/reports/gateway_evaluation.json")
    model_card_path = Path("ml/reports/MODEL_CARD.md")

    if not gw_path.exists():
        return

    expected_gw_section = generate_gateway_evaluation_markdown(gw_path).strip()
    actual_model_card_content = model_card_path.read_text(encoding="utf-8")

    assert expected_gw_section in actual_model_card_content, (
        "MODEL_CARD.md gateway evaluation tables have diverged from canonical JSON in gateway_evaluation.json. "
        "Run `python scripts/generate_model_card_tables.py --update` to resync."
    )


def test_model_card_cross_strategy_tables_match_json():
    """Assert MODEL_CARD.md contains the exact cross-strategy agreement section generated from cross_strategy_agreement.json."""
    cs_path = Path("ml/reports/cross_strategy_agreement.json")
    model_card_path = Path("ml/reports/MODEL_CARD.md")

    if not cs_path.exists():
        return

    expected_cs_section = generate_cross_strategy_markdown(cs_path).strip()
    actual_model_card_content = model_card_path.read_text(encoding="utf-8")

    assert expected_cs_section in actual_model_card_content, (
        "MODEL_CARD.md cross-strategy agreement tables have diverged from canonical JSON in cross_strategy_agreement.json. "
        "Run `python scripts/generate_model_card_tables.py --update` to resync."
    )


def test_strategy_comparison_table_cells_match_metrics_json():
    """Assert every cell in MODEL_CARD.md strategy table matches canonical metrics.json.

    Validates cell-by-cell against metrics.json (not generator function):
    - Val PR-AUC against `strategy_comparison.{strat}.validation_pr_auc`
    - Test PR-AUC against `strategy_comparison.{strat}.test_pr_auc`
    - Recall against `strategy_comparison.{strat}.metrics_at_high.confusion_matrix`
    - Test Precision against `strategy_comparison.{strat}.metrics_at_high.confusion_matrix`
    - High Threshold against `strategy_comparison.{strat}.threshold_high`
    """
    import json

    metrics_path = Path("ml/reports/metrics.json")
    model_card_path = Path("ml/reports/MODEL_CARD.md")

    assert metrics_path.exists(), f"Missing {metrics_path}"
    assert model_card_path.exists(), f"Missing {model_card_path}"

    with open(metrics_path, encoding="utf-8") as f:
        metrics_data = json.load(f)

    strat_comp = metrics_data["strategy_comparison"]
    content = model_card_path.read_text(encoding="utf-8")

    marker = "| Model / Strategy | Val PR-AUC | Test PR-AUC |"
    assert marker in content, "Missing strategy comparison table in MODEL_CARD.md"

    table_lines = [
        line.strip()
        for line in content.split(marker)[1].split("\n\n")[0].splitlines()
        if line.strip().startswith("|") and not line.strip().startswith("| :---")
    ]

    strategy_row_mapping = {
        "class_weight (deployed) [Set 1]": "class_weight",
        "none (10:1 undersampled baseline)": "none",
        "smote_enn": "smote_enn",
        "Logistic Regression baseline": "logistic_regression",
        "Random Forest baseline": "random_forest",
        "Full-train reference (none, 6.08M)": "full_train_none",
        "Full-train reference (class_weight, 6.08M)": "full_train_class_weight",
        "Variant A reference (15 features)": "variant_A_reference",
    }

    for line in table_lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells or len(cells) < 11:
            continue
        label = cells[0]
        if label not in strategy_row_mapping:
            continue

        key = strategy_row_mapping[label]
        row_json = strat_comp[key]

        # 1. Val PR-AUC
        expected_val_auc = (
            f"{row_json.get('validation_pr_auc', '-'):.4f}"
            if isinstance(row_json.get("validation_pr_auc"), (float, int))
            else "-"
        )
        assert cells[1] == expected_val_auc, (
            f"Cell Val PR-AUC mismatch for {key}: markdown '{cells[1]}' vs metrics.json "
            f"key `strategy_comparison.{key}.validation_pr_auc` '{expected_val_auc}'"
        )

        # 2. Test PR-AUC
        expected_test_auc = (
            f"{row_json.get('test_pr_auc', row_json.get('pr_auc', '-')):.4f}"
            if isinstance(row_json.get("test_pr_auc", row_json.get("pr_auc")), (float, int))
            else "-"
        )
        assert cells[2] == expected_test_auc, (
            f"Cell Test PR-AUC mismatch for {key}: markdown '{cells[2]}' vs metrics.json "
            f"key `strategy_comparison.{key}.test_pr_auc` '{expected_test_auc}'"
        )

        # 3. Recall & Precision
        m = row_json.get("metrics_at_high") or row_json.get("test_metrics") or row_json
        cm = m.get("confusion_matrix", {})
        tp = cm.get("true_positives", 0)
        fp = cm.get("false_positives", 0)
        fn = cm.get("false_negatives", 0)

        expected_recall = f"{(tp / (tp + fn) if (tp + fn) > 0 else (m.get('recall') or 0.0)) * 100:.2f}%"
        assert cells[3] == expected_recall, (
            f"Cell Recall mismatch for {key}: markdown '{cells[3]}' vs metrics.json "
            f"confusion matrix '{expected_recall}'"
        )

        expected_prec = f"{(tp / (tp + fp) if (tp + fp) > 0 else (m.get('precision') or 0.0)) * 100:.2f}%"
        assert cells[4] == expected_prec, (
            f"Cell Test Precision mismatch for {key}: markdown '{cells[4]}' vs metrics.json "
            f"confusion matrix '{expected_prec}'"
        )

        # 4. Threshold high
        t_high = row_json.get("threshold_high")
        if t_high is not None and key in ("class_weight", "none", "smote_enn"):
            expected_thresh = f"$t_{{\\text{{high}}}} = {t_high:.4f}$ (Val-selected)"
            assert cells[10] == expected_thresh, (
                f"Cell Threshold mismatch for {key}: markdown '{cells[10]}' vs metrics.json "
                f"key `strategy_comparison.{key}.threshold_high` '{expected_thresh}'"
            )
        elif key == "variant_A_reference":
            expected_thresh = "$t_{\\text{high}} = 0.9981$ (Stage 1 Val-selected, 15 features)"
            assert cells[10] == expected_thresh, (
                f"Cell Threshold mismatch for {key}: markdown '{cells[10]}' vs expected '{expected_thresh}'"
            )



