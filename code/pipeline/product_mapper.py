"""
pipeline/product_mapper.py
--------------------------
Maps a support ticket to its product area and detects company.

Product area is derived from:
  1. The explicit company field
  2. Keyword analysis of the ticket text
  3. The top retrieval result's corpus path
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Company detection keywords
# ---------------------------------------------------------------------------

_COMPANY_KEYWORDS: dict[str, list[str]] = {
    "hackerrank": [
        "hackerrank", "hacker rank", "assessment", "recruiter", "candidate",
        "coding test", "coding challenge", "proctoring", "interview platform",
        "skillup", "chakra", "screen", "mock interview", "certified assessment",
        "resume builder", "hiring", "certificate", "submission", "hackerrank for work",
        "apply tab", "inactivity", "test invite", "test settings",
        "community", "leaderboard",
    ],
    "claude": [
        "claude", "anthropic", "claude.ai", "claude api", "lti",
        "bedrock", "amazon bedrock", "team workspace", "claude code",
        "claude desktop", "conversation", "data crawl", "claude mobile",
        "security vulnerability", "bug bounty", "claude lti",
        "pro plan", "max plan", "claude team", "claude enterprise",
    ],
    "visa": [
        "visa", "visa card", "visa debit", "visa credit",
        "card stolen", "card blocked", "traveller", "merchant",
        "minimum spend", "refund", "chargeback", "dispute",
        "identity theft", "fraud", "cash advance", "cardholder",
        "atm", "virgin islands", "transaction",
    ],
}


def detect_company(issue: str, subject: str, company_field: str) -> Optional[str]:
    """
    Return normalised company: 'hackerrank' | 'claude' | 'visa' | None.
    Prefers the explicit CSV field, then falls back to keyword matching.
    """
    field = (company_field or "").strip().lower()

    # Direct field match
    if field in ("hackerrank",):
        return "hackerrank"
    if field in ("claude",):
        return "claude"
    if field in ("visa",):
        return "visa"

    # Keyword matching
    combined = (issue + " " + subject).lower()
    scores: dict[str, int] = {k: 0 for k in _COMPANY_KEYWORDS}
    for company, kws in _COMPANY_KEYWORDS.items():
        for kw in kws:
            if kw in combined:
                scores[company] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best
    return None


# ---------------------------------------------------------------------------
# Product area mapping from corpus paths
# ---------------------------------------------------------------------------

_AREA_MAP: dict[str, str] = {
    # HackerRank
    "screen": "screen",
    "interviews": "interviews",
    "interview": "interviews",
    "library": "library",
    "settings": "settings",
    "integrations": "integrations",
    "skillup": "skillup",
    "chakra": "chakra",
    "engage": "engage",
    "general_help": "general_support",
    "general-help": "general_support",
    "hackerrank_community": "community",
    "uncategorized": "general_support",
    # Claude
    "claude": "claude_general",
    "claude_api_and_console": "claude_api",
    "claude-api-and-console": "claude_api",
    "amazon_bedrock": "amazon_bedrock",
    "amazon-bedrock": "amazon_bedrock",
    "claude_code": "claude_code",
    "claude-code": "claude_code",
    "claude_desktop": "claude_desktop",
    "claude-desktop": "claude_desktop",
    "claude_for_education": "claude_education",
    "claude-for-education": "claude_education",
    "claude_for_government": "claude_government",
    "claude-for-government": "claude_government",
    "claude_for_nonprofits": "claude_nonprofits",
    "claude-for-nonprofits": "claude_nonprofits",
    "claude_mobile_apps": "claude_mobile",
    "claude-mobile-apps": "claude_mobile",
    "claude-in-chrome": "claude_chrome",
    "pro_and_max_plans": "billing",
    "pro-and-max-plans": "billing",
    "team_and_enterprise_plans": "team_enterprise",
    "team-and-enterprise-plans": "team_enterprise",
    "privacy_and_legal": "privacy",
    "privacy-and-legal": "privacy",
    "safeguards": "safeguards",
    "connectors": "connectors",
    "identity_management_sso_jit_scim": "identity_management",
    "identity-management-sso-jit-scim": "identity_management",
    # Visa
    "support": "general_support",
    "consumer": "consumer_support",
    "small-business": "business_support",
}


def map_product_area(corpus_path: Optional[str], company: Optional[str]) -> str:
    """
    Derive a product_area label from the corpus path of the top retrieval hit.
    Falls back to keyword-based inference if no corpus path is available.
    """
    if not corpus_path:
        return _infer_area_from_company(company)

    # Extract the sub-directory name from the corpus path
    # e.g. "hackerrank/screen/..." -> "screen"
    parts = corpus_path.replace("\\", "/").split("/")
    # parts[0] is the company dir; parts[1] is the category
    if len(parts) >= 2:
        raw = parts[1].strip()
        # Handle files directly under company root (e.g. "visa/support.md")
        if raw.endswith(".md"):
            raw = raw[:-3]
        # Check direct mapping
        if raw in _AREA_MAP:
            return _AREA_MAP[raw]
        # Try normalised form
        normalised = raw.replace("-", "_").replace(" ", "_").lower()
        if normalised in _AREA_MAP:
            return _AREA_MAP[normalised]
        # Try deeper path segment if available (e.g. visa/support/consumer/*)
        if len(parts) >= 3:
            deeper = parts[2].strip()
            if deeper.endswith(".md"):
                deeper = deeper[:-3]
            if deeper in _AREA_MAP:
                return _AREA_MAP[deeper]
            deeper_n = deeper.replace("-", "_").replace(" ", "_").lower()
            if deeper_n in _AREA_MAP:
                return _AREA_MAP[deeper_n]
        return normalised if normalised else _infer_area_from_company(company)

    return _infer_area_from_company(company)


def _infer_area_from_company(company: Optional[str]) -> str:
    """Fallback when no corpus path is available."""
    if company == "hackerrank":
        return "general_support"
    if company == "claude":
        return "claude_general"
    if company == "visa":
        return "general_support"
    return "general_support"


def infer_product_area_from_text(issue: str, subject: str, company: Optional[str]) -> str:
    """
    Keyword-based product area detection for when retrieval results
    aren't available (e.g. escalation before retrieval).
    """
    text = (issue + " " + subject).lower()

    if company == "hackerrank":
        if any(k in text for k in ["test", "assessment", "candidate", "invite", "proctor"]):
            return "screen"
        if any(k in text for k in ["interview", "interviewer", "lobby"]):
            return "interviews"
        if any(k in text for k in ["subscription", "billing", "payment", "invoice"]):
            return "billing"
        if any(k in text for k in ["resume", "certificate", "community", "practice"]):
            return "community"
        if any(k in text for k in ["team", "user", "role", "admin", "employee"]):
            return "settings"
        if any(k in text for k in ["integration", "sso", "api"]):
            return "integrations"
        return "general_support"

    if company == "claude":
        if any(k in text for k in ["api", "console", "token"]):
            return "claude_api"
        if any(k in text for k in ["bedrock", "aws"]):
            return "amazon_bedrock"
        if any(k in text for k in ["privacy", "data", "crawl"]):
            return "privacy"
        if any(k in text for k in ["workspace", "team", "seat", "admin"]):
            return "team_enterprise"
        if any(k in text for k in ["desktop"]):
            return "claude_desktop"
        if any(k in text for k in ["education", "lti", "student", "professor"]):
            return "claude_education"
        if any(k in text for k in ["security", "vulnerability", "bug bounty"]):
            return "safeguards"
        return "claude_general"

    if company == "visa":
        if any(k in text for k in ["travel", "abroad", "foreign"]):
            return "travel_support"
        if any(k in text for k in ["stolen", "lost", "block", "fraud", "identity"]):
            return "fraud_support"
        if any(k in text for k in ["dispute", "charge", "refund"]):
            return "dispute_support"
        if any(k in text for k in ["cash", "atm", "emergency"]):
            return "emergency_support"
        return "general_support"

    return "general_support"
