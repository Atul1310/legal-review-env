"""
inference.py — Legal Document Review OpenEnv Baseline Inference Script

Uses an LLM (via OpenAI-compatible API) to act as a legal reviewer agent.
The agent reads contract clauses, identifies issues, proposes redlines,
and determines the correct document disposition.

Environment variables:
  API_BASE_URL  — API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — Hugging Face / API key (REQUIRED)
  LEGAL_ENV_URL — Environment server URL (default: http://localhost:7860)
  LEGAL_TASK    — Task to run: nda_standard | saas_agreement | acquisition_loi | all

STDOUT FORMAT (strictly followed for evaluation):
  [START] task=<task_name> env=legal-review model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────
# Environment variables (with required defaults)
# ──────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LEGAL_ENV_URL = os.getenv("LEGAL_ENV_URL", "http://localhost:7860")
LEGAL_TASK   = os.getenv("LEGAL_TASK", "all")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "25"))

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ──────────────────────────────────────────────────────
# OpenAI client
# ──────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["nda_standard", "saas_agreement", "acquisition_loi"]

# ──────────────────────────────────────────────────────
# System prompt for legal review agent
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert legal contract reviewer at a top-tier law firm.
Your job is to review contracts, identify problematic clauses, and determine
whether documents should be approved, sent for revision, or rejected.

You must respond with ONLY a JSON object (no markdown, no explanation) like:
{
  "action_type": "<one of the valid action types>",
  "clause_id": "<clause id or null>",
  "issue_type": "<issue type or null>",
  "description": "<your analysis or null>",
  "proposed_text": "<redline text or null>",
  "search_query": "<search term or null>",
  "reasoning": "<your step-by-step reasoning>"
}

Valid action_types:
  Diagnostic: read_clause, search_document, check_definitions, request_info
  Analysis:   flag_issue, clear_flag, assess_risk
  Remediation: propose_redline, accept_clause, reject_clause
  Document:   approve_document, reject_document, escalate, request_revision

Valid issue_types: liability_cap, missing_clause, ambiguous_term, unfavorable_term,
                   compliance_risk, ip_issue, payment_terms, termination, governing_law

Strategy:
1. Start by reading ALL clauses systematically
2. Check definitions (critical for spotting traps in defined terms)
3. Flag real issues with specific issue_type
4. Propose redlines for fixable issues
5. Assess overall risk level
6. Choose disposition: approve_document, request_revision, reject_document, or escalate
   - approve_document: minor or no issues
   - request_revision: real issues that can be fixed
   - reject_document: critical unfixable issues
   - escalate: beyond your authority

IMPORTANT: Do not flag standard/market terms as issues. Be accurate, not paranoid.
""").strip()


def build_user_prompt(obs: dict, step: int) -> str:
    """Build the user prompt from the current observation."""
    clauses_summary = []
    for c in obs.get("clauses", []):
        clauses_summary.append(
            f"  [{c['id']}] {c['title']} (status: {c['status']})"
        )

    flagged = obs.get("flagged_issues", [])
    flagged_summary = [
        f"  [{f['clause_id']}] {f['issue_type']}: {f['description'][:60]}..."
        for f in flagged
    ] or ["  (none)"]

    missing = obs.get("missing_clauses", []) or ["(none)"]

    prompt = f"""=== LEGAL DOCUMENT REVIEW ===
Document: {obs.get('document_title', 'Unknown')}
Type: {obs.get('document_type', 'Unknown')}
Step: {step} / {obs.get('max_steps', 25)}

CLAUSES:
{chr(10).join(clauses_summary)}

CURRENTLY FLAGGED ISSUES:
{chr(10).join(flagged_summary)}

MISSING CLAUSES TO CONSIDER:
{chr(10).join(missing)}

LAST ACTION RESULT:
{obs.get('action_result', 'None')}

ERROR (if any): {obs.get('last_action_error') or 'null'}

Review complete: {obs.get('review_complete', False)}

Based on your analysis so far, what is your next action?
Respond with ONLY the JSON action object."""

    return prompt


def call_llm(messages: list) -> str:
    """Call the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> dict:
    """Parse LLM output to action dict. Falls back to read_clause on error."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    # Fallback action
    return {"action_type": "read_clause", "clause_id": None, "reasoning": "fallback"}


def run_task(task_name: str) -> dict:
    """
    Run a single task episode and return results.
    Returns dict with steps, rewards, success, score.
    """
    base = LEGAL_ENV_URL.rstrip("/")
    rewards: List[float] = []
    done = False
    steps = 0
    success = False
    score = 0.0
    episode_id = None
    last_error = None

    try:
        # Reset
        resp = httpx.post(f"{base}/reset", json={"task_name": task_name}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        episode_id = data["episode_id"]
        obs = data["observation"]

        print(f"[START] task={task_name} env=legal-review model={MODEL_NAME}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Track clauses we've already read to build a smarter strategy
        unread_clauses = [c["id"] for c in obs.get("clauses", [])]
        read_clauses = set()

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build prompt
            user_msg = build_user_prompt(obs, step_num)
            messages.append({"role": "user", "content": user_msg})

            # Get LLM action
            raw_action = call_llm(messages)
            messages.append({"role": "assistant", "content": raw_action})

            action_dict = parse_action(raw_action)

            # Smart override: if LLM hasn't read all clauses yet, prioritize reading
            action_type = action_dict.get("action_type", "read_clause")
            clause_id = action_dict.get("clause_id")

            # Track read clauses
            if action_type == "read_clause" and clause_id:
                read_clauses.add(clause_id)
                if clause_id in unread_clauses:
                    unread_clauses.remove(clause_id)

            # Step
            step_payload = {
                "episode_id": episode_id,
                "action": {
                    "action_type": action_type,
                    "clause_id": action_dict.get("clause_id"),
                    "issue_type": action_dict.get("issue_type"),
                    "description": action_dict.get("description"),
                    "proposed_text": action_dict.get("proposed_text"),
                    "search_query": action_dict.get("search_query"),
                    "reasoning": action_dict.get("reasoning"),
                }
            }

            step_resp = httpx.post(f"{base}/step", json=step_payload, timeout=30)
            step_resp.raise_for_status()
            step_data = step_resp.json()

            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            last_error = obs.get("last_action_error")

            rewards.append(reward)
            steps = step_num

            action_str = f"{action_type}({clause_id or ''})"
            error_str = last_error if last_error else "null"
            done_str = "true" if done else "false"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")

    except Exception as e:
        last_error = str(e)
        print(f"[STEP] step={steps} action=error reward=0.00 done=true error={last_error}", file=sys.stderr)

    # Get final score
    try:
        if episode_id:
            score_resp = httpx.get(f"{base}/score", params={"episode_id": episode_id}, timeout=15)
            if score_resp.status_code == 200:
                score_data = score_resp.json()
                score = score_data.get("score", 0.0)
    except Exception:
        pass

    success = score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")

    return {
        "task": task_name,
        "steps": steps,
        "rewards": rewards,
        "score": score,
        "success": success,
    }


def main():
    if LEGAL_TASK == "all":
        tasks_to_run = TASKS
    elif LEGAL_TASK in TASKS:
        tasks_to_run = [LEGAL_TASK]
    else:
        print(f"Unknown task '{LEGAL_TASK}'. Valid: {TASKS} or 'all'", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for task in tasks_to_run:
        result = run_task(task)
        all_results.append(result)

    # Summary
    print("\n=== INFERENCE SUMMARY ===")
    total_score = 0.0
    for r in all_results:
        print(f"  {r['task']}: score={r['score']:.2f} success={r['success']} steps={r['steps']}")
        total_score += r["score"]
    if all_results:
        print(f"  Average score: {total_score / len(all_results):.2f}")


if __name__ == "__main__":
    main()
