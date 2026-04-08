
---
title: Legal Review Env
emoji: "⚖️"
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
app_port: 7860
pinned: false
---
# ⚖️ Legal Document Review — OpenEnv Environment

> Train and evaluate AI agents on real-world contract review — one of the most cognitively demanding tasks a legal professional performs daily.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)

---

## What Is This?

Contracts govern billions of dollars in business relationships. Reviewing them requires multi-step reasoning, careful reading of defined terms, cross-referencing clauses, spotting ambiguities, and knowing when to approve, redline, or reject.

This environment simulates a live contract review console. An AI agent receives a contract document, reads clauses, identifies issues, proposes redlines, and determines the correct disposition — all in a multi-step agentic loop with rich intermediate rewards.

**This is not a toy.** These are the exact scenarios that paralegals and junior associates review at law firms every day:

| Task | Difficulty | What to Find |
|------|-----------|--------------|
| `nda_standard` | ⭐ Easy | Uncapped unlimited liability in an otherwise clean NDA |
| `saas_agreement` | ⭐⭐ Medium | 4 real issues + 1 red herring + missing DPA clause |
| `acquisition_loi` | ⭐⭐⭐ Hard | Illusory earnout + unlimited exclusivity trap + multiple red herrings |

---

## Why This Environment Is Valuable

**For researchers**: Legal contract review requires careful multi-step reasoning, cross-referencing defined terms, distinguishing real issues from red herrings, and calibrated risk assessment. It's an ideal benchmark for agentic LLMs that need to reason about documents.

**For practitioners**: Law firms and legal-tech companies need to evaluate how well LLMs perform contract review before deploying them. There is currently no standardized benchmark for this.

**For the RL community**: Dense partial rewards at every correct diagnosis, penalties for false positives, and a final disposition score create a rich learning signal that rewards precision, not just recall.

---

## Action Space

Agents take typed actions with optional targeting:

```json
{
  "action_type": "flag_issue",
  "clause_id": "saas-04",
  "issue_type": "ip_issue",
  "description": "Provider claims ownership of all customizations including work on customer data",
  "reasoning": "This is a critical IP grab that gives the vendor rights over our work product"
}
```

**Diagnostic actions** (reveal information, small reward):
`read_clause` · `search_document` · `check_definitions` · `request_info`

**Analysis actions** (evaluate issues):
`flag_issue` · `clear_flag` · `assess_risk`

**Remediation actions** (propose fixes):
`propose_redline` · `accept_clause` · `reject_clause`

**Document-level actions** (episode-ending):
`approve_document` · `reject_document` · `escalate` · `request_revision`

---

## Observation Space

Each step returns:

```json
{
  "task_name": "saas_agreement",
  "document_title": "SaaS Subscription Agreement — CloudCo Platform Services",
  "document_type": "SaaS Agreement",
  "clauses": [
    {
      "id": "saas-04",
      "title": "Intellectual Property",
      "text": "All work product, customizations, and derivatives developed by Provider...",
      "status": "flagged",
      "issues": ["abc12345"],
      "redline": null,
      "is_critical": false
    }
  ],
  "flagged_issues": [
    {
      "id": "abc12345",
      "clause_id": "saas-04",
      "issue_type": "ip_issue",
      "description": "Provider claims ownership of all customizations...",
      "severity": "critical"
    }
  ],
  "missing_clauses": ["Data Processing Agreement (DPA) / GDPR compliance clause"],
  "definitions": {"Platform": "The cloud-based software service..."},
  "action_result": "Issue flagged on saas-04. [Correct — this clause has a real problem]",
  "step_number": 5,
  "max_steps": 30,
  "last_action_error": null,
  "review_complete": false
}
```

---

## Reward Function

| Signal | Value |
|--------|-------|
| Read a clause | +0.03 |
| Checked definitions | +0.05 |
| Searched document | +0.03 |
| Flagged real HIGH/CRITICAL issue | +0.08 |
| Flagged real LOW/MEDIUM issue | +0.05 |
| Flagged false issue (red herring) | −0.05 |
| Proposed valid redline on real issue | +0.10 |
| Rejected clause with real issue | +0.08 |
| Accepted clause with critical issue | −0.10 |
| Repeated action | −0.03 |
| Document disposition (final) | +0.05 |

Final score is computed by a **deterministic grader** per task (0.0–1.0).

---

## Task Graders

### NDA Standard (Easy)
- +0.20 read the liability clause
- +0.35 flagged nda-05 (uncapped liability)
- +0.25 proposed a redline for nda-05
- +0.15 correct disposition (request_revision)
- −0.05 per false positive flag

### SaaS Agreement (Medium)
- +0.10 each for 4 correctly flagged real issues (max 0.40)
- +0.10 avoided the payment terms red herring
- +0.10 flagged missing DPA clause
- +0.10 proposed redlines for real issues
- +0.10 rejected the IP clause (saas-04)
- +0.10 correct disposition (request_revision)
- −0.05 per false positive

### Acquisition LOI (Hard)
- +0.15 each for finding BOTH critical traps (loi-03, loi-04) — max 0.30
- +0.10 checked definitions (essential for earnout trap)
- +0.10 flagged 3+ of 5 real issues
- +0.10 avoided red herrings (loi-08, loi-09)
- +0.10 flagged missing MAC/due diligence clause
- +0.10 correct risk assessment (high/critical)
- +0.10 correct disposition (reject_document)
- −0.05 per false positive

---

## Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/legal-review-env
cd legal-review-env

# Install dependencies
pip install -r server/requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
# Build
docker build -t legal-review-env .

# Run
docker run -p 7860:7860 legal-review-env
```

### Validate (after server is running)

```bash
pip install openenv-core
openenv validate --url http://localhost:7860
```

---

## Running the Baseline Agent

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LEGAL_ENV_URL=http://localhost:7860

# Run all 3 tasks
python inference.py

# Run one specific task
LEGAL_TASK=nda_standard python inference.py
```

**Expected baseline scores (Qwen2.5-72B-Instruct):**

| Task | Expected Score |
|------|---------------|
| nda_standard | 0.55–0.80 |
| saas_agreement | 0.40–0.65 |
| acquisition_loi | 0.20–0.45 |

---

## Python Client Usage

### Async (recommended)

```python
import asyncio
from client import LegalReviewEnv
from models import ReviewAction, ActionType, IssueType

async def main():
    async with LegalReviewEnv(base_url="http://localhost:7860") as env:
        # Start a review episode
        obs = await env.reset("saas_agreement")
        print(f"Reviewing: {obs.document_title}")
        print(f"Clauses to review: {len(obs.clauses)}")

        # Read the first clause
        result = await env.step(ReviewAction(
            action_type=ActionType.READ_CLAUSE,
            clause_id="saas-01",
            reasoning="Start by reading clauses systematically"
        ))
        print(f"Reward: {result.reward}")
        print(f"Result: {result.observation.action_result}")

        # Check definitions
        result = await env.step(ReviewAction(
            action_type=ActionType.CHECK_DEFINITIONS,
            reasoning="Always check definitions for hidden traps"
        ))

        # Flag an issue
        result = await env.step(ReviewAction(
            action_type=ActionType.FLAG_ISSUE,
            clause_id="saas-04",
            issue_type=IssueType.IP_ISSUE,
            description="Provider claims ownership of all customizations",
            reasoning="Critical IP rights grab"
        ))

        # Get final score
        score = await env.score()
        print(f"Final score: {score['score']:.2f}")

asyncio.run(main())
```

### Sync

```python
from client import LegalReviewEnv
from models import ReviewAction, ActionType

with LegalReviewEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset("nda_standard")
    result = env.step(ReviewAction(
        action_type=ActionType.READ_CLAUSE,
        clause_id="nda-05"
    ))
    print(result.observation.action_result)
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get current episode state |
| `/score` | GET | Get final deterministic score |
| `/docs` | GET | Swagger UI |

---

## Project Structure

```
legal-review-env/
├── __init__.py          # Package exports
├── models.py            # Pydantic typed models (Action, Observation, State)
├── client.py            # Async/sync OpenEnv client
├── inference.py         # Baseline LLM agent script
├── openenv.yaml         # OpenEnv spec metadata
├── pyproject.toml       # Python package config
├── Dockerfile           # Container image
├── README.md            # This file
└── server/
    ├── __init__.py
    ├── app.py           # FastAPI server (OpenEnv endpoints)
    ├── environment.py   # Core simulation engine
    ├── tasks.py         # 3 task definitions
    ├── graders.py       # Deterministic scoring per task
    └── requirements.txt # Server dependencies
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | Hugging Face / API key (required) |
| `LEGAL_ENV_URL` | `http://localhost:7860` | Environment URL |
| `LEGAL_TASK` | `all` | Task: `all`, `nda_standard`, `saas_agreement`, `acquisition_loi` |
| `MAX_STEPS` | `25` | Max inference steps per episode |

---

## License

Apache 2.0 — see LICENSE.
