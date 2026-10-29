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



