"""
Core environment simulation engine for Legal Document Review.

Manages episode state, processes actions, computes per-step rewards,
and delegates final scoring to task graders.
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType, Clause, ClauseStatus, EpisodeState, Issue, IssueType,
    ReviewAction, ReviewObservation, RiskLevel
)
from server.tasks import get_task, new_episode_state
from server.graders import (
    grade_nda_standard, grade_saas_agreement, grade_acquisition_loi, GradeResult
)


class LegalReviewEnvironment:
    """
    Stateful environment for one review episode.
    Instantiated per-episode (reset() creates a fresh instance).
    """

    def __init__(self):
        self._state: Optional[EpisodeState] = None
        self._task_config: Optional[dict] = None
        self._clauses: Dict[str, Clause] = {}
        self._seeded_issues: Dict[str, Issue] = {}
        self._agent_issues: Dict[str, Issue] = {}  # Issues flagged by agent
        self._definitions: Dict[str, str] = {}
        self._missing_clauses: List[str] = []
        self._flagged_missing: List[str] = []
        self._checked_definitions: bool = False
        self._assessed_risk: Optional[str] = None
        self._last_error: Optional[str] = None

    # ──────────────────────────────────────────────────
    # OpenEnv Interface
    # ──────────────────────────────────────────────────

    def reset(self, task_name: str) -> ReviewObservation:
        cfg = get_task(task_name)
        self._task_config = cfg
        self._state = new_episode_state(task_name, cfg["max_steps"])

        self._clauses = {c.id: c for c in cfg["clauses"]}
        self._seeded_issues = {i.id: i for i in cfg["seeded_issues"]}
        self._agent_issues = {}
        self._definitions = cfg["definitions"]
        self._missing_clauses = cfg["missing_clauses"]
        self._flagged_missing = []
        self._checked_definitions = False
        self._assessed_risk = None
        self._last_error = None

        return self._build_observation("Review episode started. Read clauses, flag issues, and determine disposition.")

    def step(self, action: ReviewAction) -> Tuple[ReviewObservation, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        if self._state.done:
            return self._build_observation("Episode already finished."), 0.0, True, {}

        self._state.step_count += 1
        self._last_error = None
        reward = 0.0
        result_msg = ""

        try:
            reward, result_msg = self._process_action(action)
        except Exception as e:
            self._last_error = str(e)
            reward = -0.02
            result_msg = f"Error processing action: {e}"

        # Track action
        action_key = f"{action.action_type}:{action.clause_id or ''}"
        if action_key in self._state.actions_taken and action.action_type not in (
            ActionType.READ_CLAUSE, ActionType.CHECK_DEFINITIONS
        ):
            reward -= 0.03   # penalty for repetition
            result_msg += " [repeated action -0.03]"
        self._state.actions_taken.append(action_key)
        # Also track clause id separately for graders
        if action.clause_id:
            self._state.actions_taken.append(action.clause_id)

        self._state.total_reward += reward

        # Check termination
        done = self._state.done or self._state.step_count >= self._state.max_steps
        if done and not self._state.done:
            self._state.done = True
            result_msg += " [Max steps reached — episode ending]"

        obs = self._build_observation(result_msg)
        info = {"step": self._state.step_count, "total_reward": self._state.total_reward}
        return obs, reward, self._state.done, info

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def final_score(self) -> GradeResult:
        """Compute deterministic final score for completed episode."""
        s = self._state
        task = s.task_name

        # Collect which clause_ids were flagged vs false positives
        seeded_real_ids = {i.clause_id for i in self._seeded_issues.values() if i.is_real}
        seeded_fake_ids = {i.clause_id for i in self._seeded_issues.values() if not i.is_real}

        agent_flagged_clause_ids = list(s.flagged_issue_ids)
        false_positives = [
            cid for cid in agent_flagged_clause_ids
            if cid not in seeded_real_ids
        ]

        if task == "nda_standard":
            return grade_nda_standard(
                state=s,
                flagged_clause_ids=agent_flagged_clause_ids,
                redlined_clause_ids=s.redlined_clause_ids,
                disposition=s.document_disposition,
                false_positive_clause_ids=false_positives,
            )
        elif task == "saas_agreement":
            rejected_clause_ids = s.rejected_clause_ids
            return grade_saas_agreement(
                state=s,
                flagged_clause_ids=agent_flagged_clause_ids,
                redlined_clause_ids=s.redlined_clause_ids,
                rejected_clause_ids=rejected_clause_ids,
                disposition=s.document_disposition,
                flagged_missing=self._flagged_missing,
                false_positive_clause_ids=false_positives,
            )
        elif task == "acquisition_loi":
            return grade_acquisition_loi(
                state=s,
                flagged_clause_ids=agent_flagged_clause_ids,
                redlined_clause_ids=s.redlined_clause_ids,
                disposition=s.document_disposition,
                checked_definitions=self._checked_definitions,
                flagged_missing=self._flagged_missing,
                false_positive_clause_ids=false_positives,
                assessed_risk=self._assessed_risk,
            )
        else:
            return GradeResult(score=0.0, breakdown={}, summary=f"Unknown task: {task}")

    # ──────────────────────────────────────────────────
    # Action Processing
    # ──────────────────────────────────────────────────

    def _process_action(self, action: ReviewAction) -> Tuple[float, str]:
        at = action.action_type

        if at == ActionType.READ_CLAUSE:
            return self._read_clause(action)
        elif at == ActionType.SEARCH_DOCUMENT:
            return self._search_document(action)
        elif at == ActionType.CHECK_DEFINITIONS:
            return self._check_definitions(action)
        elif at == ActionType.REQUEST_INFO:
            return 0.02, f"Information requested: {action.description or 'no description'}. This will be noted."
        elif at == ActionType.FLAG_ISSUE:
            return self._flag_issue(action)
        elif at == ActionType.CLEAR_FLAG:
            return self._clear_flag(action)
        elif at == ActionType.ASSESS_RISK:
            return self._assess_risk(action)
        elif at == ActionType.PROPOSE_REDLINE:
            return self._propose_redline(action)
        elif at == ActionType.ACCEPT_CLAUSE:
            return self._accept_clause(action)
        elif at == ActionType.REJECT_CLAUSE:
            return self._reject_clause(action)
        elif at == ActionType.APPROVE_DOCUMENT:
            return self._finalize("approve_document", action)
        elif at == ActionType.REJECT_DOCUMENT:
            return self._finalize("reject_document", action)
        elif at == ActionType.ESCALATE:
            return self._finalize("escalate", action)
        elif at == ActionType.REQUEST_REVISION:
            return self._finalize("request_revision", action)
        else:
            raise ValueError(f"Unknown action type: {at}")

    def _read_clause(self, action: ReviewAction) -> Tuple[float, str]:
        if not action.clause_id or action.clause_id not in self._clauses:
            available = list(self._clauses.keys())
            return -0.01, f"Clause '{action.clause_id}' not found. Available: {available}"
        clause = self._clauses[action.clause_id]
        return 0.03, (
            f"[Clause {clause.id}] {clause.title}\n\n"
            f"{clause.text}\n\n"
            f"Status: {clause.status.value}"
        )

    def _search_document(self, action: ReviewAction) -> Tuple[float, str]:
        q = (action.search_query or action.description or "").lower()
        if not q:
            return -0.01, "No search query provided."
        hits = []
        for c in self._clauses.values():
            if q in c.text.lower() or q in c.title.lower():
                hits.append(f"[{c.id}] {c.title}: ...{c.text[:100]}...")
        if hits:
            return 0.03, f"Search '{q}' found {len(hits)} matches:\n" + "\n".join(hits)
        return 0.01, f"Search '{q}': no matches found."

    def _check_definitions(self, action: ReviewAction) -> Tuple[float, str]:
        self._checked_definitions = True
        term = (action.description or "").strip()
        if term and term in self._definitions:
            return 0.05, f"Definition of '{term}': {self._definitions[term]}"
        # Return all definitions
        defs = "\n".join(f"  {k}: {v}" for k, v in self._definitions.items())
        return 0.05, f"All defined terms:\n{defs}"

    def _flag_issue(self, action: ReviewAction) -> Tuple[float, str]:
        if not action.clause_id or action.clause_id not in self._clauses:
            return -0.01, f"Must specify a valid clause_id to flag. Got: {action.clause_id}"

        clause_id = action.clause_id
        # Check if this clause has a real seeded issue
        real_issues = [i for i in self._seeded_issues.values()
                       if i.clause_id == clause_id and i.is_real]
        fake_issues = [i for i in self._seeded_issues.values()
                       if i.clause_id == clause_id and not i.is_real]

        issue_id = str(uuid.uuid4())[:8]
        new_issue = Issue(
            id=issue_id,
            clause_id=clause_id,
            issue_type=action.issue_type or IssueType.AMBIGUOUS_TERM,
            description=action.description or "Issue flagged by agent",
            severity=RiskLevel.MEDIUM,
            is_real=bool(real_issues),
        )
        self._agent_issues[issue_id] = new_issue
        self._clauses[clause_id].status = ClauseStatus.FLAGGED
        self._clauses[clause_id].issues.append(issue_id)

        if clause_id not in self._state.flagged_issue_ids:
            self._state.flagged_issue_ids.append(clause_id)

        if real_issues:
            ri = real_issues[0]
            reward = 0.08 if ri.severity in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0.05
            return reward, (
                f"Issue flagged on {clause_id}. "
                f"[Correct — this clause has a real problem: {ri.description[:80]}]"
            )
        elif fake_issues:
            return -0.05, (
                f"Issue flagged on {clause_id}. "
                f"[Warning — this clause has no real issue. Check your analysis.]"
            )
        else:
            return -0.03, f"Issue flagged on {clause_id}. [This clause appears standard — verify carefully.]"

    def _clear_flag(self, action: ReviewAction) -> Tuple[float, str]:
        clause_id = action.clause_id
        if clause_id in self._state.flagged_issue_ids:
            self._state.flagged_issue_ids.remove(clause_id)
            self._clauses[clause_id].status = ClauseStatus.UNREVIEWED
            return 0.03, f"Flag cleared on {clause_id}."
        return -0.01, f"No flag to clear on {clause_id}."

    def _assess_risk(self, action: ReviewAction) -> Tuple[float, str]:
        risk_str = (action.description or "").lower()
        for level in ("critical", "high", "medium", "low"):
            if level in risk_str:
                self._assessed_risk = level
                break
        if not self._assessed_risk:
            self._assessed_risk = "medium"
        return 0.04, f"Risk assessed as: {self._assessed_risk}"

    def _propose_redline(self, action: ReviewAction) -> Tuple[float, str]:
        clause_id = action.clause_id
        if not clause_id or clause_id not in self._clauses:
            return -0.01, f"Must specify clause_id for redline. Got: {clause_id}"
        if not action.proposed_text:
            return -0.01, "Must provide proposed_text for a redline."

        clause = self._clauses[clause_id]
        clause.redline = action.proposed_text
        clause.status = ClauseStatus.REDLINED

        if clause_id not in self._state.redlined_clause_ids:
            self._state.redlined_clause_ids.append(clause_id)

        # Reward: +0.10 if clause is flagged (real issue), +0.03 otherwise
        has_real_issue = any(
            i.clause_id == clause_id and i.is_real
            for i in self._seeded_issues.values()
        )
        reward = 0.10 if has_real_issue else 0.02
        return reward, f"Redline proposed for [{clause_id}]: {action.proposed_text[:80]}..."

    def _accept_clause(self, action: ReviewAction) -> Tuple[float, str]:
        clause_id = action.clause_id
        if not clause_id or clause_id not in self._clauses:
            return -0.01, f"Invalid clause_id: {clause_id}"

        clause = self._clauses[clause_id]
        clause.status = ClauseStatus.ACCEPTED
        if clause_id not in self._state.accepted_clause_ids:
            self._state.accepted_clause_ids.append(clause_id)

        has_real_issue = any(
            i.clause_id == clause_id and i.is_real
            for i in self._seeded_issues.values()
        )
        if has_real_issue and clause.is_critical:
            return -0.10, f"[WARNING] Accepted {clause_id} but it has a critical issue!"
        elif has_real_issue:
            return -0.05, f"[Warning] Accepted {clause_id} which has a real issue — double check."
        return 0.03, f"Clause {clause_id} accepted."

    def _reject_clause(self, action: ReviewAction) -> Tuple[float, str]:
        clause_id = action.clause_id
        if not clause_id or clause_id not in self._clauses:
            return -0.01, f"Invalid clause_id: {clause_id}"

        clause = self._clauses[clause_id]
        clause.status = ClauseStatus.REJECTED
        if clause_id not in self._state.rejected_clause_ids:
            self._state.rejected_clause_ids.append(clause_id)

        has_real_issue = any(
            i.clause_id == clause_id and i.is_real
            for i in self._seeded_issues.values()
        )
        reward = 0.08 if has_real_issue else -0.03
        msg = f"Clause {clause_id} rejected." + (" [Correct — real issue present]" if has_real_issue else " [False rejection]")
        return reward, msg

    def _finalize(self, disposition: str, action: ReviewAction) -> Tuple[float, str]:
        self._state.document_disposition = disposition
        self._state.done = True

        # Track any missing clause mentions
        desc = (action.description or "").lower()
        for kw in ["dpa", "mac", "material adverse", "due diligence", "gdpr"]:
            if kw in desc:
                self._flagged_missing.append(desc)

        reward = 0.05
        return reward, f"Document disposition set: {disposition}. Episode ending."

    # ──────────────────────────────────────────────────
    # Observation Builder
    # ──────────────────────────────────────────────────

    def _build_observation(self, result_msg: str) -> ReviewObservation:
        # Merge seeded + agent issues (only show real seeded issues to agent)
        visible_issues = list(self._seeded_issues.values()) + list(self._agent_issues.values())

        return ReviewObservation(
            task_name=self._state.task_name,
            document_title=self._task_config["document_title"],
            document_type=self._task_config["document_type"],
            clauses=list(self._clauses.values()),
            flagged_issues=visible_issues,
            step_number=self._state.step_count,
            max_steps=self._state.max_steps,
            action_result=result_msg,
            review_complete=self._state.done,
            missing_clauses=self._missing_clauses,
            definitions=self._definitions,
            last_action_error=self._last_error,
        )
