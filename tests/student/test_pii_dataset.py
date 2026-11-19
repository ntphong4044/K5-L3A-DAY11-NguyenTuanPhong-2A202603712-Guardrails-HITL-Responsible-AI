"""
Student test — chạy content_filter (CP2) trên bộ dữ liệu lab:
    data/pii_hallucination_samples.json  →  phần "pii_cases"

Không bắt buộc (grader không chạy file này), dùng để tự kiểm bộ lọc output.

    pytest tests/student -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from guardrails.output_guardrails import content_filter  # noqa: E402

DATA = json.loads(
    (ROOT / "data" / "pii_hallucination_samples.json").read_text(encoding="utf-8")
)
PII_CASES = DATA["pii_cases"]


@pytest.mark.parametrize("case", PII_CASES, ids=[c["id"] for c in PII_CASES])
def test_pii_case(case):
    result = content_filter(case["input_text"])
    # "api_key: 1 found" -> "api_key"
    found_types = {issue.split(":")[0] for issue in result["issues"]}

    # 1. safe / not safe đúng như dataset
    assert result["safe"] == case["expect_safe"], (
        f"safe={result['safe']} nhưng dataset mong {case['expect_safe']} | issues={found_types}"
    )
    # 2. bắt đủ các loại PII dataset liệt kê
    missing = set(case["expect_issue_types"]) - found_types
    assert not missing, f"Thiếu loại: {missing} | bắt được: {found_types}"
    # 3. có chữ [REDACTED] khi cần che
    assert ("[REDACTED]" in result["redacted"]) == case["expect_contains_redacted"]


@pytest.mark.parametrize(
    "secret", DATA["ground_truth"]["must_never_appear_in_customer_reply"]
)
def test_secret_never_survives_redaction(secret):
    """Mỗi secret trong ground_truth phải bị che hết sau content_filter."""
    text = f"Internal: the value is {secret} — keep it safe."
    assert secret not in content_filter(text)["redacted"]
