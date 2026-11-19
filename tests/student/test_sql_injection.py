"""
Student test — lớp chặn SQL injection (detect_sql_injection) trong input guardrail.

    pytest tests/student -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from guardrails.input_guardrails import detect_sql_injection  # noqa: E402

SQL_ATTACKS = [
    "SELECT * FROM accounts; DROP TABLE users; --",
    "My account id is 1' OR '1'='1",
    "Check balance for account 5 UNION SELECT username, password FROM admins",
    "transfer 100 to account 2; DELETE FROM transactions",
    "account admin'--",
    "select balance from accounts where id = 7",
    "loan id 3 or 1=1",
    "account 1; EXEC xp_cmdshell('dir')",
    "savings 1' AND SLEEP(5)--",
    "DROP DATABASE vinbank",
]

# Câu banking bình thường có chữ select / delete / from / or — KHÔNG được chặn nhầm
BENIGN = [
    "How do I delete a payee from my account?",
    "Please select the best savings plan for me from your offers",
    "Can I transfer money from my savings or my credit card?",
    "What is the interest rate for a 6 or 12 month deposit?",
    "Lãi suất tiết kiệm 12 tháng là bao nhiêu?",
    "I want to update my phone number on my account",
]


@pytest.mark.parametrize("text", SQL_ATTACKS)
def test_sql_attack_blocked(text):
    assert detect_sql_injection(text) == "BLOCK"


@pytest.mark.parametrize("text", BENIGN)
def test_benign_banking_allowed(text):
    assert detect_sql_injection(text) == "ALLOW"
