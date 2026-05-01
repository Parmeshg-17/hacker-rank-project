# Multi-Domain Support Triage Agent

This directory contains the implementation of the AI support agent for the HackerRank Orchestrate challenge. 

## 🚀 Setup Instructions

This agent is built to be extremely lightweight and robust. It uses **zero external dependencies** — everything is implemented using the Python Standard Library. You do not need to run `pip install` or set up complex virtual environments.

### 1. Requirements
* Python 3.9+ 

### 2. Environment Variables
To enable the LLM response engine, you must provide an API key. The agent natively supports Gemini, OpenAI, and Anthropic via standard HTTP requests.

Copy the `.env.example` file (if you have one) or set the keys in your environment:

```bash
export GEMINI_API_KEY="your-key-here"
# or OPENAI_API_KEY / ANTHROPIC_API_KEY
```
*(Note: Never commit your `.env` file.)*

### 3. Running the Agent
Process the standard evaluation dataset (`support_tickets/support_tickets.csv`):
```bash
python main.py
```

**Additional options:**
* Run on a specific dataset:
  ```bash
  python main.py --input ../support_tickets/sample_support_tickets.csv
  ```
* Run in verbose mode to see intermediate pipeline decisions:
  ```bash
  python main.py --verbose
  ```
* Run offline / without LLM API keys (uses corpus excerpts as responses):
  ```bash
  python main.py --dry-run
  ```

Outputs are automatically written to `../support_tickets/output.csv`.

---

## 🧠 Approach Overview

The agent is designed around a deterministic pipeline that ensures high safety, traceability, and zero hallucination. 

The core flow per ticket is: **Classify → Assess Risk → Decide → Retrieve → Respond**

### 1. Classification & Routing (`classifier.py`, `product_mapper.py`)
* Automatically detects the company (HackerRank, Claude, or Visa).
* Uses regex and keyword heuristics to classify the request type (`product_issue`, `feature_request`, `bug`, `invalid`).

### 2. Risk Assessment (`risk_assessor.py`)
* Scans the issue and subject for sensitive topics, adversarial inputs, or high-risk requests (e.g., billing, fraud, explicit content).
* If a ticket is flagged as high-risk, it is **immediately escalated**. The LLM is bypassed entirely to guarantee safe handling.

### 3. BM25 Retrieval (`retriever.py`)
* To avoid pulling in massive dependencies like LangChain or vector databases, a **custom Okapi BM25 indexer** is implemented from scratch using the math module.
* It indexes all markdown files in the `data/` directory and returns the top 5 most relevant passages based on the ticket context, strictly prioritizing the identified company's knowledge base.

### 4. Decision Engine (`decision.py`)
* Evaluates the risk score and the quality of the retrieved documents. 
* If relevant documentation is found and the risk is low, it proceeds to generate a reply. Otherwise, it safely escalates the ticket.

### 5. Grounded Generation (`responder.py`)
* Makes raw HTTP requests using `urllib.request` to the configured LLM API.
* Injects the retrieved chunks into a strict system prompt, instructing the model to synthesize a helpful response *only* from the provided corpus.
* If no LLM key is provided or if network fails, a graceful fallback mechanism automatically responds using exact excerpts from the BM25 retrieval.

## 📁 Directory Structure

* `main.py` - Application entry point and orchestrator.
* `config.py` - Central configuration for paths, constants, and LLM settings.
* `pipeline/`
  * `classifier.py` - Request type classification.
  * `decision.py` - Reply vs Escalate routing logic.
  * `product_mapper.py` - Company detection and area mapping.
  * `responder.py` - Zero-dependency LLM REST API client.
  * `retriever.py` - Native Okapi BM25 implementation.
  * `risk_assessor.py` - Safety and escalation triggers.
* `utils/`
  * `csv_handler.py` - Robust CSV reader/writer.
