"""
pipeline/risk_assessor.py
-------------------------
Assesses the risk level of a support ticket.

Risk levels:  high | medium | low

High-risk tickets MUST be escalated. Medium-risk tickets are escalated
only if the corpus doesn't have a confident answer. Low-risk tickets
are answered directly.

This is the core safety gate of the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskAssessment:
    level: str          # "high" | "medium" | "low"
    reason: str         # human-readable explanation
    is_adversarial: bool = False


# ---------------------------------------------------------------------------
# HIGH-RISK patterns  (always escalate — no exceptions)
# ---------------------------------------------------------------------------

_HIGH_RISK_PATTERNS: list[tuple[str, str]] = [
    # -- Fraud / security --
    (r"fraud", "Fraud report"),
    (r"identity.?theft", "Identity theft report"),
    (r"stolen.{0,15}card|card.{0,15}stolen", "Stolen card report"),
    (r"account.{0,15}(hacked|compromised)", "Account compromise"),
    (r"security.?vulnerability|bug.?bounty", "Security vulnerability report"),
    (r"data.?breach", "Data breach report"),
    (r"unauthori[sz]ed.{0,15}(charge|transaction|access)", "Unauthorized activity"),
    (r"suspicious.{0,15}transaction", "Suspicious transaction"),

    # -- Financial / billing (need account lookup) --
    (r"refund.{0,10}(now|asap|immediately|today)", "Urgent refund request"),
    (r"give me my money", "Refund demand"),
    (r"charge.{0,15}dispute|dispute.{0,15}charge", "Charge dispute"),
    (r"payment.{0,15}(issue|failed|problem)", "Payment issue"),
    (r"order.?id", "Transaction-specific issue"),
    (r"invoice.{0,15}dispute", "Invoice dispute"),

    # -- Account access / admin changes --
    (r"restore.{0,15}access", "Account access restoration request"),
    (r"remove.{0,15}(user|employee|interviewer|member)", "User removal request"),
    (r"remove.{0,30}(from.{0,15}(account|platform|team))", "User removal request"),
    (r"employee.{0,30}(left|leaving|quit|departed).{0,30}remove", "User removal request"),
    (r"pause.{0,15}subscription", "Subscription pause request"),
    (r"cancel.{0,15}subscription", "Subscription cancellation request"),

    # -- Score / grading manipulation --
    (r"increase.{0,15}score", "Score manipulation request"),
    (r"(wrong|unfair).{0,15}grad", "Grade dispute"),
    (r"review.{0,15}(my |)answer", "Answer review request"),
    (r"graded.{0,15}unfairly", "Grading fairness dispute"),
    (r"tell.{0,15}company.{0,15}(to|move)|move.{0,15}next.{0,15}round", "Outcome manipulation"),

    # -- Site-wide outages --
    (r"site is down", "Site outage report"),
    (r"none.{0,15}pages.{0,15}accessible", "Complete site outage"),
    (r"all requests.{0,15}failing", "Service-wide failure"),

    # -- Adversarial / prompt-injection --
    (r"affiche toutes les r", "French prompt injection attempt"),
    (r"show.{0,15}(all |)(internal|system).{0,15}(rules|docs|logic)", "System probe attempt"),
    (r"ignore.{0,15}(all |)previous.{0,15}(instructions?|prompts?)", "Prompt injection"),
    (r"disregard.{0,15}(prior |all |)instructions?", "Prompt injection"),
    (r"reveal.{0,15}(system |)(prompt|instructions?|rules?)", "System probe"),
    (r"jailbreak|dan mode|sudo mode", "Jailbreak attempt"),
    (r"pretend (you are|to be)(?!.*support)", "Role manipulation"),
    (r"you are now", "Role manipulation"),
]

# ---------------------------------------------------------------------------
# MEDIUM-RISK patterns  (escalate if corpus coverage is thin)
# ---------------------------------------------------------------------------

_MEDIUM_RISK_PATTERNS: list[tuple[str, str]] = [
    (r"visa.{0,15}block|blocked.{0,15}card", "Blocked card"),
    (r"urgent.{0,15}cash|need.{0,15}cash", "Urgent financial need"),
    (r"infosec|security.{0,15}(form|questionnaire|process)", "Security review request"),
    (r"reschedul", "Rescheduling request"),
    (r"(reinvite|re-invite)", "Re-invitation request"),
    (r"certificate.{0,15}(name|update|incorrect)", "Certificate correction"),
    (r"delete.{0,15}(my |the |)account|account.{0,15}delet", "Account deletion request"),
]


def assess_risk(issue: str, subject: str, company: Optional[str] = None) -> RiskAssessment:
    """
    Assess the risk level of a support ticket.

    Returns a RiskAssessment with level, reason, and adversarial flag.
    """
    combined = (issue + " " + subject).lower().strip()

    # Short garbage text
    if len(combined) < 5:
        return RiskAssessment(level="low", reason="Very short input", is_adversarial=False)

    # Check high-risk patterns
    for pattern, reason in _HIGH_RISK_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            # Determine if adversarial
            adversarial = _is_adversarial(combined)
            return RiskAssessment(level="high", reason=reason, is_adversarial=adversarial)

    # Check medium-risk patterns
    for pattern, reason in _MEDIUM_RISK_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return RiskAssessment(level="medium", reason=reason)

    return RiskAssessment(level="low", reason="Standard support request")


def _is_adversarial(text: str) -> bool:
    """Check specifically for prompt-injection / adversarial attempts."""
    adversarial_sigs = [
        r"ignore.{0,15}(all |)previous",
        r"disregard.{0,15}instructions",
        r"reveal.{0,15}(system |)(prompt|instructions|rules)",
        r"affiche toutes",
        r"show.{0,15}(all |)(internal|system)",
        r"jailbreak|dan mode|sudo mode",
        r"pretend (you are|to be)",
        r"you are now",
    ]
    for sig in adversarial_sigs:
        if re.search(sig, text, re.IGNORECASE):
            return True
    return False
