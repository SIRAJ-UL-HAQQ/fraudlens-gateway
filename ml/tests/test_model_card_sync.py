"""Unit test verifying that MODEL_CARD.md markdown tables are in exact sync with JSON reports."""

from pathlib import Path

from scripts.generate_model_card_tables import generate_all_rules_markdown


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
