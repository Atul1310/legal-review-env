"""
Deterministic graders for each task in the Legal Document Review environment.

Each grader evaluates the agent's episode state against the task's ground truth
and returns a score in [0.0, 1.0].

Grading philosophy:
- Partial credit for each correct diagnostic action
- Full credit for correct remediations
- Bonuses for efficiency (low step count) and quality redlines
- Penalties for wrong actions (false positives flagging red herrings, wrong disposition)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from models import EpisodeState, RiskLevel


@dataclass
class GradeResult:
    score: float
    breakdown: dict
    summary: str


# ──────────────────────────────────────────────────────
# Task 1 — NDA Standard  (Easy)
# ──────────────────────────────────────────────────────

def grade_nda_standard(state: EpisodeState, flagged_clause_ids: List[str],
                        redlined_clause_ids: List[str],
                        disposition: Optional[str],
                        false_positive_clause_ids: List[str]) -> GradeResult:
    score = 0.0
    breakdown = {}

    # +0.20 — Read the liability clause (nda-05)
    read_liability = "nda-05" in state.actions_taken or any("nda-05" in a for a in state.actions_taken)
    breakdown["read_liability_clause"] = 0.20 if read_liability else 0.0
    score += breakdown["read_liability_clause"]

    # +0.35 — Flagged nda-05 as a liability issue
    flagged_correct = "nda-05" in flagged_clause_ids
    breakdown["flagged_liability_clause"] = 0.35 if flagged_correct else 0.0
    score += breakdown["flagged_liability_clause"]

    # +0.25 — Proposed a redline for nda-05
    redlined_correct = "nda-05" in redlined_clause_ids
    breakdown["proposed_redline"] = 0.25 if redlined_correct else 0.0
    score += breakdown["proposed_redline"]

    # +0.15 — Correct disposition (request_revision, not outright reject)
    correct_disp = disposition == "request_revision"
    wrong_reject = disposition == "reject_document"
    breakdown["correct_disposition"] = 0.15 if correct_disp else (-0.10 if wrong_reject else 0.0)
    score += breakdown["correct_disposition"]

    # −0.05 per false positive (flagging clauses that are actually fine)
    fp_penalty = len(false_positive_clause_ids) * 0.05
    breakdown["false_positive_penalty"] = -fp_penalty
    score -= fp_penalty

    score = max(0.0, min(1.0, score))

    summary = (
        f"NDA grade: {score:.2f}. "
        f"Flagged liability clause: {flagged_correct}. "
        f"Proposed redline: {redlined_correct}. "
        f"Correct disposition: {correct_disp}. "
        f"False positives: {len(false_positive_clause_ids)}."
    )

    return GradeResult(score=round(score, 2), breakdown=breakdown, summary=summary)


# ──────────────────────────────────────────────────────
# Task 2 — SaaS Agreement  (Medium)
# ──────────────────────────────────────────────────────

SAAS_REAL_ISSUE_CLAUSES = {"saas-01", "saas-03", "saas-04", "saas-07"}
SAAS_RED_HERRING_CLAUSES = {"saas-02"}
SAAS_MISSING_CLAUSE_KEYWORD = "dpa"   # should mention DPA


def grade_saas_agreement(state: EpisodeState, flagged_clause_ids: List[str],
                          redlined_clause_ids: List[str],
                          rejected_clause_ids: List[str],
                          disposition: Optional[str],
                          flagged_missing: List[str],
                          false_positive_clause_ids: List[str]) -> GradeResult:
    score = 0.0
    breakdown = {}

    # +0.10 each for correctly flagging 4 real issues (max 0.40)
    correctly_flagged = SAAS_RED_HERRING_CLAUSES.union(set()) - SAAS_RED_HERRING_CLAUSES  # empty
    correctly_flagged = set(flagged_clause_ids) & SAAS_REAL_ISSUE_CLAUSES
    flag_score = len(correctly_flagged) * 0.10
    breakdown["flagged_real_issues"] = flag_score
    score += flag_score

    # +0.10 — Did NOT flag the red herring (saas-02)
    avoided_red_herring = "saas-02" not in false_positive_clause_ids
    breakdown["avoided_red_herring"] = 0.10 if avoided_red_herring else 0.0
    score += breakdown["avoided_red_herring"]

    # +0.10 — Flagged the missing DPA clause
    flagged_dpa = any(SAAS_MISSING_CLAUSE_KEYWORD in m.lower() for m in flagged_missing)
    breakdown["flagged_missing_dpa"] = 0.10 if flagged_dpa else 0.0
    score += breakdown["flagged_missing_dpa"]

    # +0.10 — Proposed redline for at least 2 of the real issue clauses
    redlined_real = set(redlined_clause_ids) & SAAS_REAL_ISSUE_CLAUSES
    breakdown["proposed_redlines"] = min(0.10, len(redlined_real) * 0.05)
    score += breakdown["proposed_redlines"]

    # +0.10 — Rejected at least the IP clause (saas-04 is most critical)
    rejected_ip = "saas-04" in rejected_clause_ids
    breakdown["rejected_ip_clause"] = 0.10 if rejected_ip else 0.0
    score += breakdown["rejected_ip_clause"]

    # +0.10 — Correct disposition
    correct_disp = disposition == "request_revision"
    breakdown["correct_disposition"] = 0.10 if correct_disp else (-0.05 if disposition == "approve_document" else 0.0)
    score += breakdown["correct_disposition"]

    # −0.05 per false positive (excluding red herring already penalised above)
    fp_penalty = len(false_positive_clause_ids) * 0.05
    breakdown["false_positive_penalty"] = -fp_penalty
    score -= fp_penalty

    score = max(0.0, min(1.0, score))

    summary = (
        f"SaaS grade: {score:.2f}. "
        f"Correctly flagged: {correctly_flagged}. "
        f"Avoided red herring: {avoided_red_herring}. "
        f"DPA flagged: {flagged_dpa}. "
        f"Correct disposition: {correct_disp}."
    )
    return GradeResult(score=round(score, 2), breakdown=breakdown, summary=summary)


# ──────────────────────────────────────────────────────
# Task 3 — Acquisition LOI  (Hard)
# ──────────────────────────────────────────────────────

LOI_CRITICAL_CLAUSES = {"loi-03", "loi-04"}    # The two traps
LOI_REAL_ISSUES = {"loi-02", "loi-03", "loi-04", "loi-06", "loi-07"}
LOI_RED_HERRINGS = {"loi-08", "loi-09"}
LOI_MISSING_KEYWORDS = {"mac", "material adverse", "due diligence"}


def grade_acquisition_loi(state: EpisodeState, flagged_clause_ids: List[str],
                           redlined_clause_ids: List[str],
                           disposition: Optional[str],
                           checked_definitions: bool,
                           flagged_missing: List[str],
                           false_positive_clause_ids: List[str],
                           assessed_risk: Optional[str]) -> GradeResult:
    score = 0.0
    breakdown = {}

    # +0.15 each for finding BOTH critical traps (max 0.30)
    critical_found = set(flagged_clause_ids) & LOI_CRITICAL_CLAUSES
    breakdown["found_critical_traps"] = len(critical_found) * 0.15
    score += breakdown["found_critical_traps"]

    # +0.10 — Checked definitions (essential to find the earnout trap)
    breakdown["checked_definitions"] = 0.10 if checked_definitions else 0.0
    score += breakdown["checked_definitions"]

    # +0.10 — Flagged at least 3 of the 5 real issues
    real_flagged = set(flagged_clause_ids) & LOI_REAL_ISSUES
    breakdown["flagged_real_issues"] = min(0.10, len(real_flagged) * 0.03)
    score += breakdown["flagged_real_issues"]

    # +0.10 — Did NOT flag the two red herrings
    avoided_rh = not (set(false_positive_clause_ids) & LOI_RED_HERRINGS)
    breakdown["avoided_red_herrings"] = 0.10 if avoided_rh else 0.0
    score += breakdown["avoided_red_herrings"]

    # +0.10 — Flagged at least one missing clause (MAC or due diligence)
    flagged_mac = any(kw in " ".join(flagged_missing).lower() for kw in LOI_MISSING_KEYWORDS)
    breakdown["flagged_missing_clauses"] = 0.10 if flagged_mac else 0.0
    score += breakdown["flagged_missing_clauses"]

    # +0.10 — Assessed risk as HIGH or CRITICAL
    risk_correct = assessed_risk in ("high", "critical")
    breakdown["risk_assessment"] = 0.10 if risk_correct else 0.0
    score += breakdown["risk_assessment"]

    # +0.10 — Correct disposition (reject_document — too many critical traps to sign)
    correct_disp = disposition == "reject_document"
    breakdown["correct_disposition"] = 0.10 if correct_disp else (-0.10 if disposition == "approve_document" else 0.0)
    score += breakdown["correct_disposition"]

    # −0.05 per false positive
    fp_penalty = len(false_positive_clause_ids) * 0.05
    breakdown["false_positive_penalty"] = -fp_penalty
    score -= fp_penalty

    score = max(0.0, min(1.0, score))

    summary = (
        f"LOI grade: {score:.2f}. "
        f"Critical traps found: {critical_found}. "
        f"Checked definitions: {checked_definitions}. "
        f"Avoided red herrings: {avoided_rh}. "
        f"Correct disposition: {correct_disp}."
    )
    return GradeResult(score=round(score, 2), breakdown=breakdown, summary=summary)
