"""Test verifying that docs/latency-benchmark.md is in sync and conforms to SRS citation rules.

Requirements:
- Cites SRS NFR-1, NFR-4, NFR-5 by ID.
- No pass/fail verdict wording (observations only).
- Matches the output of scripts/generate_latency_table.py --check.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_FILE = REPO_ROOT / "docs" / "latency-benchmark.md"
SCRIPT_FILE = REPO_ROOT / "scripts" / "generate_latency_table.py"


def test_latency_docs_in_sync():
    """Verify that docs/latency-benchmark.md is in sync with ml/reports/latency_benchmark.json."""
    benchmark_report = REPO_ROOT / "ml" / "reports" / "latency_benchmark.json"
    assert benchmark_report.exists(), (
        f"Missing canonical benchmark report: {benchmark_report}. "
        "docs/latency-benchmark.md must only be generated from this committed artifact."
    )
    assert DOCS_FILE.exists(), f"Missing documentation file: {DOCS_FILE}"
    cmd = [sys.executable, str(SCRIPT_FILE), "--check"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"generate_latency_table.py --check failed:\n{result.stderr}\n{result.stdout}"


def test_latency_docs_srs_citations_by_id():
    """Verify that official SRS requirement IDs NFR-1, NFR-4, and NFR-5 are explicitly cited."""
    assert DOCS_FILE.exists(), f"Missing documentation file: {DOCS_FILE}"
    content = DOCS_FILE.read_text(encoding="utf-8")
    assert "NFR-1" in content, "Missing official SRS citation for NFR-1"
    assert "NFR-4" in content, "Missing official SRS citation for NFR-4"
    assert "NFR-5" in content, "Missing official SRS citation for NFR-5"


def test_latency_docs_no_pass_fail_verdicts():
    """Verify that the documentation contains no pass/fail verdict wording."""
    assert DOCS_FILE.exists(), f"Missing documentation file: {DOCS_FILE}"
    content = DOCS_FILE.read_text(encoding="utf-8")

    # Forbidden verdict patterns
    forbidden_verdict_patterns = [
        r"\bPASS\b",
        r"\bFAIL\b",
        r"\bPASSED\b",
        r"\bFAILED\b",
        r"\bVerdict\s*:\s*(Pass|Fail)\b",
    ]

    for pattern in forbidden_verdict_patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        # Filter out false positives if any in file names or standard markdown
        real_matches = [m for m in matches if m.upper() in ("PASS", "FAIL", "PASSED", "FAILED")]
        assert not real_matches, f"Found forbidden pass/fail verdict wording '{pattern}': {real_matches}"
