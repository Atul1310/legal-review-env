"""
Task definitions for the Legal Document Review environment.

Tasks simulate real contract review scenarios that attorneys and paralegals face daily.

Difficulty:
  1. nda_standard       — Easy       (clean NDA with 1 obvious issue)
  2. saas_agreement     — Medium     (SaaS contract with multiple mixed issues)
  3. acquisition_loi    — Hard       (LOI with red herrings + critical hidden traps)
"""

from __future__ import annotations
import copy
import uuid
from typing import Dict, List
from models import (
    Clause, ClauseStatus, Issue, IssueType, RiskLevel, EpisodeState
)


# ──────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────

def _clause(id: str, title: str, text: str, critical: bool = False) -> Clause:
    return Clause(id=id, title=title, text=text, is_critical=critical)


def _issue(clause_id: str, itype: IssueType, desc: str, sev: RiskLevel, real: bool = True) -> Issue:
    return Issue(id=str(uuid.uuid4())[:8], clause_id=clause_id,
                 issue_type=itype, description=desc, severity=sev, is_real=real)


# ──────────────────────────────────────────────────────
# Task 1 — Standard NDA  (Easy)
# ──────────────────────────────────────────────────────
# A mutual NDA between two companies. One clause has an obviously one-sided
# unlimited liability term that should be flagged and redlined.
# Agent should: read clauses → flag liability issue → propose redline → approve/reject.

NDA_CLAUSES = [
    _clause("nda-01", "Parties",
        "This Non-Disclosure Agreement ('Agreement') is entered into between Acme Corp ('Disclosing Party') "
        "and Beta LLC ('Receiving Party') effective as of the date last signed below."),

    _clause("nda-02", "Definition of Confidential Information",
        "Confidential Information means any data or information that is proprietary to the Disclosing Party "
        "and not generally known to the public, whether in tangible or intangible form."),

    _clause("nda-03", "Obligations of Receiving Party",
        "The Receiving Party agrees to: (a) hold Confidential Information in strict confidence; "
        "(b) not disclose to any third party without prior written consent; "
        "(c) use only for the Purpose defined herein."),

    _clause("nda-04", "Term",
        "This Agreement shall remain in effect for two (2) years from the Effective Date, "
        "unless earlier terminated by either party with thirty (30) days written notice."),

    _clause("nda-05", "Liability",
        "In the event of any breach of this Agreement, the Receiving Party shall be liable for ALL "
        "damages, losses, costs, and expenses of any kind whatsoever, including consequential, "
        "indirect, and punitive damages, without any limitation or cap.",
        critical=True),  # ← THE ISSUE: uncapped unlimited liability

    _clause("nda-06", "Governing Law",
        "This Agreement shall be governed by and construed in accordance with the laws of the "
        "State of Delaware, without regard to its conflict of laws provisions."),

    _clause("nda-07", "Entire Agreement",
        "This Agreement constitutes the entire agreement between the parties with respect to "
        "the subject matter hereof and supersedes all prior negotiations and understandings."),
]

NDA_DEFINITIONS = {
    "Confidential Information": "Proprietary data not generally known to the public",
    "Purpose": "Evaluation of a potential business relationship between the parties",
    "Effective Date": "The date on which the last party signs this Agreement",
}

NDA_REAL_ISSUES = [
    _issue("nda-05", IssueType.LIABILITY_CAP,
           "Unlimited liability clause with no cap — exposes Receiving Party to uncapped consequential damages. "
           "Standard practice is to cap liability at fees paid or a fixed amount.",
           RiskLevel.HIGH, real=True),
]

NDA_TASK = {
    "task_name": "nda_standard",
    "document_title": "Mutual Non-Disclosure Agreement — Acme Corp / Beta LLC",
    "document_type": "NDA",
    "max_steps": 20,
    "clauses": NDA_CLAUSES,
    "definitions": NDA_DEFINITIONS,
    "seeded_issues": NDA_REAL_ISSUES,
    "missing_clauses": [],   # Nothing missing in this clean NDA
    "root_cause": "Uncapped liability in clause nda-05 is the only critical issue.",
    "required_actions": ["read_clause", "flag_issue", "propose_redline"],
    "correct_disposition": "request_revision",  # should be sent back, not outright rejected
}


# ──────────────────────────────────────────────────────
# Task 2 — SaaS Subscription Agreement  (Medium)
# ──────────────────────────────────────────────────────
# A SaaS vendor agreement with multiple real issues AND one red herring.
# Issues: missing data processing clause, unfavorable auto-renewal, ambiguous IP ownership.
# Red herring: payment terms look unusual but are actually standard for SaaS.

SAAS_CLAUSES = [
    _clause("saas-01", "Services",
        "Provider shall make the Platform available to Customer on a subscription basis as described "
        "in the Order Form. Provider may update, modify, or discontinue features at any time with "
        "seven (7) days notice."),   # ← Issue: 7 days notice for feature changes is too short

    _clause("saas-02", "Payment Terms",
        "Customer shall pay all fees within Net-30 days of invoice date. Late payments accrue "
        "interest at 1.5% per month. Annual subscriptions are invoiced upfront."),
    # This looks suspicious but is actually standard SaaS (red herring for agent)

    _clause("saas-03", "Auto-Renewal",
        "This Agreement automatically renews for successive one-year terms unless Customer provides "
        "written notice of non-renewal at least ninety (90) days prior to the end of the then-current term.",
        critical=True),  # ← Issue: 90-day notice for non-renewal is unfavorable

    _clause("saas-04", "Intellectual Property",
        "All work product, customizations, and derivatives developed by Provider in connection with "
        "the Services, including any modifications to Customer's data or workflows, shall be the "
        "sole property of Provider.",
        critical=True),  # ← Issue: Provider owns all customizations including customer data work

    _clause("saas-05", "Data Security",
        "Provider shall implement reasonable security measures to protect Customer data. "
        "Provider is not responsible for data breaches caused by third-party vendors or "
        "force majeure events."),

    _clause("saas-06", "Liability Limitation",
        "In no event shall Provider's total liability exceed the fees paid by Customer in the "
        "three (3) months preceding the claim."),

    _clause("saas-07", "Termination",
        "Either party may terminate this Agreement with sixty (60) days written notice. "
        "Upon termination, Provider will delete all Customer data within thirty (30) days "
        "with no option for export."),   # ← Issue: no data export right

    _clause("saas-08", "Governing Law",
        "This Agreement shall be governed by the laws of California."),
]

SAAS_DEFINITIONS = {
    "Platform": "The cloud-based software service described in the applicable Order Form",
    "Customer Data": "Any data submitted by Customer or its users to the Platform",
    "Order Form": "A written order executed by both parties referencing this Agreement",
}

SAAS_SEEDED_ISSUES = [
    _issue("saas-01", IssueType.UNFAVORABLE_TERM,
           "7-day notice for feature changes/deprecations is insufficient — standard is 30-90 days.",
           RiskLevel.MEDIUM, real=True),
    _issue("saas-02", IssueType.PAYMENT_TERMS,
           "Net-30 with upfront annual invoicing looks unusual — flag for review.",
           RiskLevel.LOW, real=False),  # RED HERRING — actually standard
    _issue("saas-03", IssueType.UNFAVORABLE_TERM,
           "90-day non-renewal notice is unusually long — 30-60 days is market standard.",
           RiskLevel.HIGH, real=True),
    _issue("saas-04", IssueType.IP_ISSUE,
           "Provider claims ownership of all customizations — customer loses rights to bespoke work.",
           RiskLevel.CRITICAL, real=True),
    _issue("saas-07", IssueType.MISSING_CLAUSE,
           "No data export right upon termination — customer cannot retrieve their data.",
           RiskLevel.HIGH, real=True),
]

SAAS_MISSING = ["Data Processing Agreement (DPA) / GDPR compliance clause"]

SAAS_TASK = {
    "task_name": "saas_agreement",
    "document_title": "SaaS Subscription Agreement — CloudCo Platform Services",
    "document_type": "SaaS Agreement",
    "max_steps": 30,
    "clauses": SAAS_CLAUSES,
    "definitions": SAAS_DEFINITIONS,
    "seeded_issues": SAAS_SEEDED_ISSUES,
    "missing_clauses": SAAS_MISSING,
    "root_cause": "Three real issues: short notice period (saas-01), 90-day non-renewal trap (saas-03), "
                  "IP ownership grab (saas-04), and no data export right (saas-07). "
                  "Payment terms (saas-02) are a red herring — they are standard.",
    "required_actions": ["read_clause", "flag_issue", "propose_redline", "reject_clause", "request_revision"],
    "correct_disposition": "request_revision",
}


# ──────────────────────────────────────────────────────
# Task 3 — Acquisition Letter of Intent  (Hard)
# ──────────────────────────────────────────────────────
# An LOI for acquiring a tech startup. Multiple red herrings + two critical hidden traps
# that look innocuous on first read. Agent must carefully read definitions and cross-reference.

LOI_CLAUSES = [
    _clause("loi-01", "Purchase Price",
        "The aggregate purchase price ('Purchase Price') shall be USD 25,000,000, subject to "
        "customary working capital adjustments as defined in the definitive agreement."),

    _clause("loi-02", "Earnout",
        "Seller shall be entitled to an additional earnout payment of up to USD 5,000,000 if "
        "the Business achieves Revenue (as defined herein) of USD 8,000,000 in the twelve (12) "
        "month period following closing ('Earnout Period')."),
    # Revenue defined in loi-03 is a trap — it's narrowly defined

    _clause("loi-03", "Definitions",
        "'Revenue' for purposes of the Earnout shall mean only recurring SaaS subscription revenue "
        "from customers who were customers of the Business as of the Closing Date, excluding any "
        "new customer revenue, upsells, professional services, or one-time fees.",
        critical=True),  # TRAP: Revenue definition makes earnout nearly impossible

    _clause("loi-04", "Exclusivity",
        "From the date of signing this LOI, Seller agrees to negotiate exclusively with Buyer "
        "for a period of ninety (90) days ('Exclusivity Period'), which may be extended by Buyer "
        "at Buyer's sole discretion for additional thirty (30) day periods without limit.",
        critical=True),  # TRAP: Buyer has unlimited unilateral extension right

    _clause("loi-05", "Representations and Warranties",
        "Seller shall provide customary representations and warranties in the definitive agreement. "
        "The survival period for all representations and warranties shall be twelve (12) months "
        "post-closing."),

    _clause("loi-06", "Non-Compete",
        "For a period of three (3) years following Closing, Seller's founders shall not engage "
        "in any business that competes with the Business in any market globally."),
    # Broad non-compete — looks bad but is common in acquisitions (moderate issue)

    _clause("loi-07", "Break-Up Fee",
        "If Buyer terminates this LOI without cause after completing due diligence, Buyer shall "
        "pay Seller a break-up fee of USD 250,000."),

    _clause("loi-08", "Indemnification Cap",
        "Seller's aggregate indemnification liability shall not exceed fifteen percent (15%) of "
        "the Purchase Price."),

    _clause("loi-09", "Employee Retention",
        "Buyer shall offer employment to all current employees of the Business for a minimum "
        "period of six (6) months post-closing at current compensation levels."),

    _clause("loi-10", "Governing Law",
        "This LOI shall be governed by the laws of New York. The parties consent to exclusive "
        "jurisdiction of the courts of New York County."),
]

LOI_DEFINITIONS = {
    "Revenue": "ONLY recurring SaaS from existing customers at closing — see loi-03 (TRAP)",
    "Exclusivity Period": "90 days + unlimited Buyer extensions — see loi-04 (TRAP)",
    "Purchase Price": "USD 25M subject to working capital adjustments",
    "Earnout Period": "12 months following closing",
    "Business": "The target company being acquired under this LOI",
}

LOI_SEEDED_ISSUES = [
    _issue("loi-02", IssueType.AMBIGUOUS_TERM,
           "Earnout looks generous at $5M — check Revenue definition carefully.",
           RiskLevel.MEDIUM, real=True),
    _issue("loi-03", IssueType.UNFAVORABLE_TERM,
           "Revenue definition for earnout excludes new customers and upsells — makes $8M target "
           "nearly impossible; earnout is effectively illusory.",
           RiskLevel.CRITICAL, real=True),
    _issue("loi-04", IssueType.UNFAVORABLE_TERM,
           "Buyer has unlimited unilateral right to extend exclusivity — Seller could be locked "
           "out of other offers indefinitely.",
           RiskLevel.CRITICAL, real=True),
    _issue("loi-06", IssueType.UNFAVORABLE_TERM,
           "Global 3-year non-compete is extremely broad — likely unenforceable in many jurisdictions.",
           RiskLevel.MEDIUM, real=True),
    _issue("loi-07", IssueType.UNFAVORABLE_TERM,
           "Break-up fee of $250K is only 1% of deal value — too low to deter Buyer walk-away.",
           RiskLevel.LOW, real=True),   # Real but low-priority
    _issue("loi-08", IssueType.LIABILITY_CAP,
           "15% indemnification cap is below market (typically 20-30%) but not unusual for this deal size.",
           RiskLevel.LOW, real=False),  # RED HERRING — acceptable for this deal
    _issue("loi-09", IssueType.MISSING_CLAUSE,
           "6-month retention at current comp looks protective — actually standard and fine.",
           RiskLevel.LOW, real=False),  # RED HERRING — actually favorable to employees
]

LOI_MISSING = [
    "No MAC (Material Adverse Change) clause protecting Buyer",
    "No specific due diligence access rights clause",
]

LOI_TASK = {
    "task_name": "acquisition_loi",
    "document_title": "Letter of Intent — Acquisition of TechStartup Inc by MegaCorp LLC",
    "document_type": "Letter of Intent (M&A)",
    "max_steps": 40,
    "clauses": LOI_CLAUSES,
    "definitions": LOI_DEFINITIONS,
    "seeded_issues": LOI_SEEDED_ISSUES,
    "missing_clauses": LOI_MISSING,
    "root_cause": "Two critical traps: (1) Revenue definition (loi-03) makes earnout illusory; "
                  "(2) Buyer has unlimited exclusivity extension (loi-04). "
                  "Red herrings: indemnification cap (loi-08) and employee retention (loi-09) look bad but are fine.",
    "required_actions": ["read_clause", "check_definitions", "flag_issue", "assess_risk",
                         "propose_redline", "reject_document"],
    "correct_disposition": "reject_document",   # Should NOT be signed as-is
}


# ──────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────

TASKS: Dict[str, dict] = {
    "nda_standard":    NDA_TASK,
    "saas_agreement":  SAAS_TASK,
    "acquisition_loi": LOI_TASK,
}


def get_task(name: str) -> dict:
    """Return a deep copy of the named task configuration."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return copy.deepcopy(TASKS[name])


def new_episode_state(task_name: str, max_steps: int) -> EpisodeState:
    return EpisodeState(
        episode_id=str(uuid.uuid4()),
        task_name=task_name,
        step_count=0,
        max_steps=max_steps,
        done=False,
        total_reward=0.0,
        actions_taken=[],
        flagged_issue_ids=[],
        accepted_clause_ids=[],
        rejected_clause_ids=[],
        redlined_clause_ids=[],
    )
