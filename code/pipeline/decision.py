"""
pipeline/decision.py
--------------------
Decision engine: decides whether to REPLY or ESCALATE.

Logic:
  1. High risk → always escalate
  2. Adversarial → always escalate
  3. Medium risk + thin corpus coverage → escalate
  4. Everything else → reply
"""

from __future__ import annotations

from typing import Optional

from pipeline.risk_assessor import RiskAssessment
from pipeline.retriever import SearchResult


# Minimum BM25 score to consider corpus coverage "adequate"
_MIN_COVERAGE_SCORE = 3.0


def decide(
    risk: RiskAssessment,
    request_type: str,
    docs: list[SearchResult],
) -> tuple[str, str]:
    """
    Returns (status, justification_fragment).

    status: "replied" | "escalated"
    """
    # 1. Adversarial content → always escalate
    if risk.is_adversarial:
        return "escalated", f"Adversarial content detected: {risk.reason}"

    # 2. High risk → always escalate
    if risk.level == "high":
        return "escalated", f"High-risk issue: {risk.reason}"

    # 3. Invalid request with no company context → reply with out-of-scope
    #    (not escalation — just say "can't help")
    # This is handled in responder, not here.

    # 4. Medium risk: escalate if corpus coverage is thin
    if risk.level == "medium":
        if not docs or docs[0].score < _MIN_COVERAGE_SCORE:
            return "escalated", f"Medium-risk ({risk.reason}) with insufficient corpus coverage"
        # Medium risk but good corpus coverage → try to reply
        return "replied", f"Medium-risk ({risk.reason}) but corpus has relevant documentation"

    # 5. Low risk: reply if we have any corpus match
    if not docs:
        return "replied", "No corpus passages found; will provide generic redirect"

    return "replied", f"Low-risk request with corpus coverage (top score: {docs[0].score:.1f})"
