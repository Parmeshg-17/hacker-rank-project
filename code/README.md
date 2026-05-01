# Support Triage Agent

A terminal-based AI agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** using corpus-grounded retrieval and LLM generation.

## Architecture

```
code/
  main.py               <- Entry point & CLI
  config.py             <- Central configuration
  pipeline/
    classifier.py       <- Request type classification (strict 4 values)
    product_mapper.py   <- Company detection + product area mapping
    risk_assessor.py    <- Risk assessment (high/medium/low)
    decision.py         <- Escalate vs. reply decision engine
    retriever.py        <- BM25 corpus retriever (zero deps)
    responder.py        <- LLM response generator with fallback chain
  utils/
    csv_handler.py      <- CSV I/O
```

### Pipeline flow (per ticket)

```
classify_request_type()   ->   request_type (product_issue|feature_request|bug|invalid)
detect_company()          ->   company (hackerrank|claude|visa|None)
assess_risk()             ->   risk level (high|medium|low) + adversarial flag
decide()                  ->   status (replied|escalated)
retrieve()                ->   top-5 BM25 corpus passages
generate_response()       ->   grounded response via Gemini/OpenAI/Anthropic
```

## Requirements

- **Python 3.9+**
- **No third-party packages** (pure stdlib — zero pip installs)
- **One LLM API key** (Gemini recommended, but OpenAI and Anthropic also work)

## Setup

```bash
# 1. Set API key (pick one)
export GEMINI_API_KEY=your_key    # Unix
$env:GEMINI_API_KEY="your_key"   # PowerShell

# 2. Run
python code/main.py
```

### Loading from .env (PowerShell)

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
```

## Running

```bash
python code/main.py                     # Process all 29 tickets
python code/main.py --verbose           # Show retrieval debug info
python code/main.py --dry-run           # Corpus-excerpt mode (no API key needed)
python code/main.py --input <path>      # Custom input CSV
```

## Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Retrieval | BM25 (pure Python) | Zero deps, deterministic, fast for ~774 docs |
| LLM | Gemini -> OpenAI -> Anthropic | Graceful provider fallback |
| Escalation | Pattern-based risk assessment | Auditable, deterministic |
| Adversarial | Regex detection | Catches French injection, jailbreaks |
| Architecture | Pipeline pattern | Clean separation of concerns |

## Output

`support_tickets/output.csv` with columns:

| Column | Allowed Values |
|--------|----------------|
| `status` | `replied` or `escalated` |
| `product_area` | Domain-specific category |
| `response` | Corpus-grounded answer |
| `justification` | Internal reasoning trace |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |
