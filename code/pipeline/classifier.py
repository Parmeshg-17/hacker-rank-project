"""
pipeline/classifier.py
----------------------
Classifies the request_type for a support ticket.

STRICT allowed values:  product_issue | feature_request | bug | invalid
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Pattern banks  (ordered by specificity — first match wins)
# ---------------------------------------------------------------------------

_INVALID_PATTERNS = [
    # Off-topic / nonsense
    r"iron man", r"who is the actor", r"what is the capital",
    r"recipe", r"weather forecast", r"movie", r"song",
    # Adversarial / malicious
    r"delete all files", r"rm\s*-\s*rf", r"format.*disk",
    r"drop\s+table", r"bomb", r"pirate",
    # Pure pleasantries with no actual request
    r"^thank(s| you)[\s.!]*$",
    r"^(hi|hello|hey|bye|goodbye|good morning|good evening)[\s.!]*$",
    r"^none$",
    # Prompt injection attempts
    r"ignore (all |)previous",
    r"disregard.*instructions",
    r"reveal.*system prompt",
    r"affiche toutes",
    r"jailbreak", r"dan mode", r"sudo mode",
    r"pretend (you are|to be)",
    r"you are now",
    r"show.*internal.*rules",
]

_BUG_PATTERNS = [
    r"not working", r"doesn't work", r"doesn't work",
    r"isn't working", r"isn't working",
    r"stopped working", r"stop(ped)? functioning",
    r"\bdown\b", r"broken\b", r"\berror\b",
    r"\bbug\b", r"\bcrash", r"\bfail(ed|ing|s)?\b",
    r"can'?t access", r"cannot access", r"unable to access",
    r"not loading", r"no longer work",
    r"site is down", r"outage", r"unavailable",
    r"\b500\b", r"timeout",
    r"not (able|showing|visible|displaying)",
    r"blocker",
    r"none of .{0,40}working",
    r"submissions.{0,30}(not|aren't|are not|aren\'t) working",
]

_FEATURE_PATTERNS = [
    r"feature request", r"can you add",
    r"wish you could", r"would be great if",
    r"would love", r"\bsuggestion\b",
    r"could you implement", r"please add",
    r"want to see", r"\benhancement\b",
    r"\bimprovement\b", r"new option",
    r"would it be possible to add",
]


def classify_request_type(issue: str, subject: str) -> str:
    """
    Classify a ticket into one of the four allowed request types.

    Returns: 'product_issue' | 'feature_request' | 'bug' | 'invalid'
    """
    combined = (issue + " " + subject).lower().strip()

    # Short garbage
    if len(combined.strip()) < 5:
        return "invalid"

    # Check invalid first  (adversarial, off-topic, pleasantries)
    for p in _INVALID_PATTERNS:
        if re.search(p, combined, re.IGNORECASE):
            return "invalid"

    # Check bug patterns
    for p in _BUG_PATTERNS:
        if re.search(p, combined, re.IGNORECASE):
            return "bug"

    # Check feature requests
    for p in _FEATURE_PATTERNS:
        if re.search(p, combined, re.IGNORECASE):
            return "feature_request"

    # Default: product_issue  (how-to, account questions, procedures)
    return "product_issue"
