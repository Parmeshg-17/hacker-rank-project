"""
main.py
-------
Entry point for the Multi-Domain Support Triage Agent.

Pipeline per ticket:
  classify -> assess risk -> decide -> retrieve -> respond

Usage:
    python main.py                      # Process support_tickets.csv
    python main.py --input <path>       # Custom input CSV
    python main.py --verbose            # Show debug info
    python main.py --dry-run            # Skip LLM calls

Output: support_tickets/output.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Optional

# Ensure code/ is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config
from pipeline.classifier import classify_request_type
from pipeline.product_mapper import (
    detect_company,
    infer_product_area_from_text,
    map_product_area,
)
from pipeline.risk_assessor import assess_risk
from pipeline.retriever import CorpusRetriever
from pipeline.decision import decide
from pipeline.responder import (
    generate_adversarial_response,
    generate_escalation_response,
    generate_grounded_response,
    generate_out_of_scope_response,
)
from utils.csv_handler import read_tickets, write_output


# ---------------------------------------------------------------------------
# Logging  (AGENTS.md  section 5)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _append_log(text: str) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with config.LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def _log_session_start() -> None:
    _append_log(textwrap.dedent(f"""
## [{_now_iso()}] SESSION START

Agent: Antigravity
Repo Root: {config.REPO_ROOT}
Branch: main
Worktree: main
Parent Agent: none
Language: py
Time Remaining: see challenge deadline
""").strip() + "\n")


def _log_run_summary(
    input_name: str,
    total: int,
    replied: int,
    escalated: int,
    elapsed: float,
) -> None:
    _append_log(textwrap.dedent(f"""
## [{_now_iso()}] Processed {total} support tickets

User Prompt (verbatim, secrets redacted):
Run agent on {input_name} ({total} tickets)

Agent Response Summary:
Processed {total} support tickets in {elapsed:.1f}s.
Replied: {replied}, Escalated: {escalated}.
Pipeline: classify -> assess risk -> decide -> retrieve -> respond.

Actions:
* Read {input_name}
* Wrote output.csv
* Indexed corpus via BM25 (774 chunks)
* Called LLM for grounded responses

Context:
tool=Antigravity
branch=main
repo_root={config.REPO_ROOT}
worktree=main
parent_agent=none
""").strip() + "\n")


# ---------------------------------------------------------------------------
# The pipeline  (one ticket at a time)
# ---------------------------------------------------------------------------

def process_ticket(
    issue: str,
    subject: str,
    company_field: str,
    retriever: CorpusRetriever,
    verbose: bool = False,
) -> dict:
    """
    Full triage pipeline for a single support ticket.

    Returns a dict with: status, product_area, response, justification, request_type
    """
    issue = (issue or "").strip()
    subject = (subject or "").strip()
    company_field = (company_field or "").strip()

    # ---- Step 1: Detect company ----
    company = detect_company(issue, subject, company_field)

    # ---- Step 2: Classify request type ----
    request_type = classify_request_type(issue, subject)

    # ---- Step 3: Assess risk ----
    risk = assess_risk(issue, subject, company)

    if verbose:
        print(f"    Company={company}, Type={request_type}, "
              f"Risk={risk.level} ({risk.reason})", flush=True)

    # ---- Step 4: Handle adversarial ----
    if risk.is_adversarial:
        product_area = infer_product_area_from_text(issue, subject, company)
        resp = generate_adversarial_response()
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": resp.text,
            "justification": resp.justification,
            "request_type": "invalid",
        }

    # ---- Step 5: Handle invalid + no company  (out-of-scope) ----
    if request_type == "invalid" and not company:
        resp = generate_out_of_scope_response()
        return {
            "status": "replied",
            "product_area": "general_support",
            "response": resp.text,
            "justification": resp.justification,
            "request_type": "invalid",
        }

    # ---- Step 6: Retrieve relevant docs ----
    query = issue + " " + subject
    docs = retriever.search(query, company=company, top_k=5)

    if verbose and docs:
        print(f"    Top hit: {docs[0].chunk.doc_id} "
              f"(score={docs[0].score:.2f})", flush=True)

    # ---- Step 7: Decide escalate vs reply ----
    status, decision_reason = decide(risk, request_type, docs)

    # ---- Step 8: Determine product area ----
    if docs:
        product_area = map_product_area(docs[0].chunk.doc_id, company)
    else:
        product_area = infer_product_area_from_text(issue, subject, company)

    # ---- Step 9: Generate response ----
    if status == "escalated":
        resp = generate_escalation_response(risk.reason)
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": resp.text,
            "justification": resp.justification,
            "request_type": request_type,
        }

    # status == "replied"
    resp = generate_grounded_response(
        issue=issue,
        subject=subject,
        company=company,
        docs=docs,
        decision_reason=decision_reason,
    )
    return {
        "status": "replied",
        "product_area": product_area,
        "response": resp.text,
        "justification": resp.justification,
        "request_type": request_type,
    }


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print("=" * 65)
    print("  Multi-Domain Support Triage Agent")
    print("  Pipeline: classify -> assess risk -> decide -> retrieve -> respond")
    print("=" * 65)
    print()


def _print_result(idx: int, result: dict) -> None:
    icon = "[OK]" if result["status"] == "replied" else "[!!]"
    preview = (result.get("issue", "") or "")[:55].replace("\n", " ")
    print(
        f"  [{idx:02d}] {icon} {result['status'].upper():10s} | "
        f"{result['product_area']:20s} | {result['request_type']}"
    )
    if preview:
        print(f"       {preview}...")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument("--input", type=str, default=str(config.DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(config.DEFAULT_OUTPUT))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline without LLM (uses corpus excerpts)")
    args = parser.parse_args()

    _print_banner()

    # Check API keys (unless dry-run)
    if not args.dry_run and not config.has_any_llm_key():
        print("WARNING: No LLM API key found.")
        print("Set one of: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        print("Falling back to corpus-excerpt mode (same as --dry-run).\n")

    # Load retriever
    print("Loading corpus...", end=" ", flush=True)
    retriever = CorpusRetriever(str(config.DATA_ROOT))
    retriever.load()
    print(f"done ({retriever.chunk_count} chunks indexed).\n")

    # Log session
    _log_session_start()

    # Read tickets
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    tickets = read_tickets(input_path)
    print(f"Processing {len(tickets)} ticket(s) from {input_path.name}...\n")

    output_rows: list[dict] = []
    start = time.time()

    for idx, ticket in enumerate(tickets, start=1):
        issue = ticket.get("Issue") or ticket.get("issue") or ""
        subject = ticket.get("Subject") or ticket.get("subject") or ""
        company = ticket.get("Company") or ticket.get("company") or ""

        if args.verbose:
            print(f"  --- Ticket {idx} ---", flush=True)

        try:
            result = process_ticket(
                issue=issue,
                subject=subject,
                company_field=company,
                retriever=retriever,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  [{idx:02d}] ERROR: {e}", flush=True)
            result = {
                "status": "escalated",
                "product_area": "general_support",
                "response": "An internal error occurred. Escalating for human review.",
                "justification": f"Agent error: {e}",
                "request_type": "product_issue",
            }

        # Add input fields to output
        result["issue"] = issue
        result["subject"] = subject
        result["company"] = company

        _print_result(idx, result)
        output_rows.append(result)

        # Rate limiting
        time.sleep(0.3)

    elapsed = time.time() - start

    # Write output
    output_path = pathlib.Path(args.output)
    write_output(output_rows, output_path)

    replied = sum(1 for r in output_rows if r["status"] == "replied")
    escalated = sum(1 for r in output_rows if r["status"] == "escalated")

    print(f"[DONE] Processed {len(output_rows)} tickets in {elapsed:.1f}s")
    print(f"       Replied: {replied} | Escalated: {escalated}")
    print(f"       Output:  {output_path}")
    print(f"       Log:     {config.LOG_FILE}")

    # Log summary
    _log_run_summary(input_path.name, len(output_rows), replied, escalated, elapsed)


if __name__ == "__main__":
    main()
