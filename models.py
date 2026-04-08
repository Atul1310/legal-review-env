"""
Pydantic models for the Legal Document Review OpenEnv environment.

Action space: agents review contracts for issues, flag clauses, request info,
              propose redlines, and accept/reject/escalate documents.
Observation space: document content, flagged issues, review state, history.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


# ──────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────

class ActionType(str, Enum):
    # Diagnostic / reading actions
    READ_CLAUSE      = "read_clause"        # Read a specific clause by id
    SEARCH_DOCUMENT  = "search_document"    # Search for terms/phrases
    CHECK_DEFINITIONS= "check_definitions"  # Look up defined terms in the contract
    REQUEST_INFO     = "request_info"       # Request clarification/missing info

    # Analysis actions
    FLAG_ISSUE       = "flag_issue"         # Flag a clause as problematic
    CLEAR_FLAG       = "clear_flag"         # Remove an incorrect flag
    ASSESS_RISK      = "assess_risk"        # Assess overall risk level

    # Remediation actions
    PROPOSE_REDLINE  = "propose_redline"    # Suggest revised clause language
    ACCEPT_CLAUSE    = "accept_clause"      # Mark a clause as acceptable
    REJECT_CLAUSE    = "reject_clause"      # Mark clause as must-fix

    # Document-level actions
    APPROVE_DOCUMENT = "approve_document"   # Approve entire document
    REJECT_DOCUMENT  = "reject_document"    # Reject document (requires reason)
    ESCALATE         = "escalate"           # Escalate to senior counsel
    REQUEST_REVISION = "request_revision"   # Send back for revision


class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class IssueType(str, Enum):
    LIABILITY_CAP     = "liability_cap"
    MISSING_CLAUSE    = "missing_clause"
    AMBIGUOUS_TERM    = "ambiguous_term"
    UNFAVORABLE_TERM  = "unfavorable_term"
    COMPLIANCE_RISK   = "compliance_risk"
    IP_ISSUE          = "ip_issue"
    PAYMENT_TERMS     = "payment_terms"
    TERMINATION       = "termination"
    GOVERNING_LAW     = "governing_law"


class ClauseStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED   = "accepted"
    FLAGGED    = "flagged"
    REJECTED   = "rejected"
    REDLINED   = "redlined"


# ──────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────

class Clause(BaseModel):
    id: str
    title: str
    text: str
    status: ClauseStatus = ClauseStatus.UNREVIEWED
    issues: List[str] = Field(default_factory=list)   # issue IDs
    redline: Optional[str] = None
    is_critical: bool = False    # hidden from agent; used by grader


class Issue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    clause_id: str
    issue_type: IssueType
    description: str
    severity: RiskLevel
    is_real: bool = True         # False = red herring; hidden from agent


class ReviewAction(BaseModel):
    """Action taken by the agent during document review."""
    action_type: ActionType
    clause_id: Optional[str] = None        # Target clause (if applicable)
    issue_type: Optional[IssueType] = None # For flag_issue
    description: Optional[str] = None     # Free-text description/reasoning
    proposed_text: Optional[str] = None   # For propose_redline
    search_query: Optional[str] = None    # For search_document
    reasoning: Optional[str] = None       # Agent's reasoning (always encouraged)


class ReviewObservation(BaseModel):
    """What the agent sees after each step."""
    task_name: str
    document_title: str
    document_type: str
    clauses: List[Clause]
    flagged_issues: List[Issue]
    step_number: int
    max_steps: int
    action_result: str           # Human-readable result of last action
    review_complete: bool = False
    overall_risk: Optional[RiskLevel] = None
    missing_clauses: List[str] = Field(default_factory=list)
    definitions: Dict[str, str] = Field(default_factory=dict)
    search_results: Optional[str] = None
    last_action_error: Optional[str] = None


class EpisodeState(BaseModel):
    """Full internal episode state (returned by state() endpoint)."""
    episode_id: str
    task_name: str
    step_count: int
    max_steps: int
    done: bool
    total_reward: float
    actions_taken: List[str]
    flagged_issue_ids: List[str]
    accepted_clause_ids: List[str]
    rejected_clause_ids: List[str]
    redlined_clause_ids: List[str]
    document_disposition: Optional[str] = None  # approved/rejected/escalated/revision
