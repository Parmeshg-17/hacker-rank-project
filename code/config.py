"""
config.py
---------
Central configuration for the Support Triage Agent.
All paths, constants, and allowed values in one place.
"""

from __future__ import annotations

import os
import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_ROOT = REPO_ROOT / "data"
TICKETS_DIR = REPO_ROOT / "support_tickets"
DEFAULT_INPUT = TICKETS_DIR / "support_tickets.csv"
DEFAULT_OUTPUT = TICKETS_DIR / "output.csv"

LOG_DIR = pathlib.Path.home() / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"

# ---------------------------------------------------------------------------
# Allowed output values  (from problem statement — strict!)
# ---------------------------------------------------------------------------

ALLOWED_STATUSES = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

COMPANIES = {"hackerrank", "claude", "visa"}

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

BM25_TOP_K = 5          # passages to retrieve
BM25_EXPAND_K = 30      # pre-filter candidate pool
BM25_K1 = 1.5
BM25_B = 0.75

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 60        # seconds

# Provider priority order
LLM_PROVIDERS = ["gemini", "openai", "anthropic"]

# ---------------------------------------------------------------------------
# API key helpers  (reads from env — never hardcoded)
# ---------------------------------------------------------------------------

def get_gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

def get_openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")

def get_anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")

def has_any_llm_key() -> bool:
    return bool(get_gemini_key() or get_openai_key() or get_anthropic_key())
