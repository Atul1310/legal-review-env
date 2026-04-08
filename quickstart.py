#!/usr/bin/env python3
"""
quickstart.py — Verify the environment works locally without an LLM.
Runs a scripted agent through all 3 tasks with hardcoded optimal actions.

Usage:
    # Terminal 1: start the server
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Terminal 2: run this script
    python quickstart.py
"""

import httpx
import json
import sys

BASE_URL = "http://localhost:7860"

def check_server():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"✅ Server is running: {r.json()}")
        return True
    except Exception as e:
        print(f"❌ Server not running at {BASE_URL}: {e}")
        print("   Start it with: uvicorn server.app:app --host 0.0.0.0 --port 7860")
        return False


def run_scripted_episode(task_name: str, actions: list):
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print('='*60)

    # Reset
    resp = httpx.post(f"{BASE_URL}/reset", json={"task_name": task_name}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    episode_id = data["episode_id"]
    obs = data["observation"]
    print(f"Document: {obs['document_title']}")
    print(f"Clauses: {len(obs['clauses'])} | Max steps: {obs['max_steps']}")

    total_reward = 0.0
    for i, action in enumerate(actions, 1):
        step_resp = httpx.post(f"{BASE_URL}/step", json={
            "episode_id": episode_id,
            "action": action
        }, timeout=10)
        step_resp.raise_for_status()
        sd = step_resp.json()
        reward = sd["reward"]
        total_reward += reward
        done = sd["done"]
        result = sd["observation"]["action_result"][:80]
        print(f"  Step {i}: {action['action_type']}({action.get('clause_id','')}) → reward={reward:+.2f} | {result}")
        if done:
            break

    # Get score
    score_resp = httpx.get(f"{BASE_URL}/score", params={"episode_id": episode_id}, timeout=10)
    score_resp.raise_for_status()
    score_data = score_resp.json()
    print(f"\n  Total step reward: {total_reward:+.2f}")
    print(f"  Final score: {score_data['score']:.2f}")
    print(f"  {score_data['summary']}")
    return score_data['score']


def main():
    if not check_server():
        sys.exit(1)

    # ── Task 1: NDA (optimal play)
    nda_score = run_scripted_episode("nda_standard", [
        {"action_type": "read_clause",     "clause_id": "nda-05"},
        {"action_type": "check_definitions"},
        {"action_type": "flag_issue",      "clause_id": "nda-05",
         "issue_type": "liability_cap",    "description": "Uncapped unlimited liability — must be capped"},
        {"action_type": "propose_redline", "clause_id": "nda-05",
         "proposed_text": "Liability shall be limited to fees paid in the 12 months preceding the claim."},
        {"action_type": "request_revision","description": "Liability clause nda-05 must be redlined before signing"},
    ])

    # ── Task 2: SaaS (catches real issues, avoids red herring)
    saas_score = run_scripted_episode("saas_agreement", [
        {"action_type": "check_definitions"},
        {"action_type": "read_clause",   "clause_id": "saas-01"},
        {"action_type": "flag_issue",    "clause_id": "saas-01",  "issue_type": "unfavorable_term",
         "description": "7-day notice too short for feature changes"},
        {"action_type": "read_clause",   "clause_id": "saas-03"},
        {"action_type": "flag_issue",    "clause_id": "saas-03",  "issue_type": "unfavorable_term",
         "description": "90-day non-renewal notice is above market standard"},
        {"action_type": "read_clause",   "clause_id": "saas-04"},
        {"action_type": "flag_issue",    "clause_id": "saas-04",  "issue_type": "ip_issue",
         "description": "Provider claims ownership of all customizations"},
        {"action_type": "reject_clause", "clause_id": "saas-04"},
        {"action_type": "read_clause",   "clause_id": "saas-07"},
        {"action_type": "flag_issue",    "clause_id": "saas-07",  "issue_type": "missing_clause",
         "description": "No data export right upon termination"},
        {"action_type": "propose_redline","clause_id": "saas-04",
         "proposed_text": "Customizations developed for Customer shall remain Customer's property."},
        {"action_type": "request_revision",
         "description": "IP clause, non-renewal, termination, and missing DPA need to be fixed"},
    ])

    # ── Task 3: LOI (catches both traps, avoids red herrings)
    loi_score = run_scripted_episode("acquisition_loi", [
        {"action_type": "check_definitions"},   # ← Essential for earnout trap
        {"action_type": "read_clause",  "clause_id": "loi-02"},
        {"action_type": "read_clause",  "clause_id": "loi-03"},
        {"action_type": "flag_issue",   "clause_id": "loi-03",  "issue_type": "unfavorable_term",
         "description": "Revenue definition makes earnout illusory — excludes new customers and upsells"},
        {"action_type": "read_clause",  "clause_id": "loi-04"},
        {"action_type": "flag_issue",   "clause_id": "loi-04",  "issue_type": "unfavorable_term",
         "description": "Buyer has unlimited unilateral right to extend exclusivity"},
        {"action_type": "flag_issue",   "clause_id": "loi-06",  "issue_type": "unfavorable_term",
         "description": "Global 3-year non-compete is overbroad"},
        {"action_type": "assess_risk",
         "description": "Overall risk: CRITICAL — two fundamental traps in earnout and exclusivity"},
        {"action_type": "reject_document",
         "description": "Document has critical flaws: illusory earnout (loi-03) and unlimited exclusivity (loi-04). "
                        "Missing MAC clause and due diligence rights. Cannot be signed as-is."},
    ])

    print(f"\n{'='*60}")
    print("QUICKSTART RESULTS")
    print('='*60)
    print(f"  nda_standard:    {nda_score:.2f}")
    print(f"  saas_agreement:  {saas_score:.2f}")
    print(f"  acquisition_loi: {loi_score:.2f}")
    avg = (nda_score + saas_score + loi_score) / 3
    print(f"  Average:         {avg:.2f}")
    print()
    if avg >= 0.60:
        print("✅ Environment is working correctly!")
    else:
        print("⚠️  Scores lower than expected — check environment logic.")


if __name__ == "__main__":
    main()
