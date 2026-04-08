"""
Tests for the Legal Document Review OpenEnv environment.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import ReviewAction, ActionType, IssueType, RiskLevel
from server.environment import LegalReviewEnvironment
from server.tasks import get_task, TASKS


class TestTaskLoading:
    def test_all_tasks_exist(self):
        for name in ["nda_standard", "saas_agreement", "acquisition_loi"]:
            task = get_task(name)
            assert task["task_name"] == name
            assert len(task["clauses"]) >= 3
            assert task["max_steps"] > 0

    def test_tasks_are_independent(self):
        t1 = get_task("nda_standard")
        t2 = get_task("nda_standard")
        t1["clauses"][0].status = "accepted"
        # t2 should not be affected (deep copy)
        assert t2["clauses"][0].status.value == "unreviewed"


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = LegalReviewEnvironment()
        obs = env.reset("nda_standard")
        assert obs.task_name == "nda_standard"
        assert len(obs.clauses) > 0
        assert obs.step_number == 0

    def test_reset_all_tasks(self):
        for task_name in ["nda_standard", "saas_agreement", "acquisition_loi"]:
            env = LegalReviewEnvironment()
            obs = env.reset(task_name)
            assert obs.task_name == task_name

    def test_reset_clears_state(self):
        env = LegalReviewEnvironment()
        env.reset("nda_standard")
        env.step(ReviewAction(action_type=ActionType.READ_CLAUSE, clause_id="nda-01"))
        # Reset again
        obs = env.reset("nda_standard")
        assert obs.step_number == 0
        state = env.state()
        assert state.step_count == 0


class TestEnvironmentStep:
    def setup_method(self):
        self.env = LegalReviewEnvironment()
        self.env.reset("nda_standard")

    def test_read_clause(self):
        obs, reward, done, info = self.env.step(
            ReviewAction(action_type=ActionType.READ_CLAUSE, clause_id="nda-01")
        )
        assert reward > 0
        assert "nda-01" in obs.action_result
        assert not done

    def test_read_invalid_clause(self):
        obs, reward, done, info = self.env.step(
            ReviewAction(action_type=ActionType.READ_CLAUSE, clause_id="nonexistent")
        )
        assert reward < 0

    def test_check_definitions(self):
        obs, reward, done, info = self.env.step(
            ReviewAction(action_type=ActionType.CHECK_DEFINITIONS)
        )
        assert reward > 0
        assert "defined terms" in obs.action_result.lower() or "definition" in obs.action_result.lower()

    def test_flag_real_issue(self):
        obs, reward, done, info = self.env.step(
            ReviewAction(
                action_type=ActionType.FLAG_ISSUE,
                clause_id="nda-05",
                issue_type=IssueType.LIABILITY_CAP,
                description="Uncapped unlimited liability"
            )
        )
        assert reward > 0
        assert "nda-05" in self.env.state().flagged_issue_ids

    def test_flag_good_clause_penalized(self):
        obs, reward, done, info = self.env.step(
            ReviewAction(
                action_type=ActionType.FLAG_ISSUE,
                clause_id="nda-01",
                issue_type=IssueType.AMBIGUOUS_TERM,
                description="Flagging a perfectly fine clause"
            )
        )
        assert reward < 0

    def test_propose_redline(self):
        # First flag the issue
        self.env.step(ReviewAction(
            action_type=ActionType.FLAG_ISSUE,
            clause_id="nda-05",
            issue_type=IssueType.LIABILITY_CAP,
            description="Uncapped liability"
        ))
        # Then propose redline
        obs, reward, done, info = self.env.step(
            ReviewAction(
                action_type=ActionType.PROPOSE_REDLINE,
                clause_id="nda-05",
                proposed_text="Liability shall be capped at fees paid in preceding 12 months.",
                reasoning="Market standard liability cap"
            )
        )
        assert reward > 0
        assert "nda-05" in self.env.state().redlined_clause_ids

    def test_request_revision_ends_episode(self):
        obs, reward, done, info = self.env.step(
            ReviewAction(action_type=ActionType.REQUEST_REVISION,
                         description="Liability clause needs to be redlined")
        )
        assert done
        assert self.env.state().document_disposition == "request_revision"

    def test_max_steps_ends_episode(self):
        env = LegalReviewEnvironment()
        env.reset("nda_standard")
        state = env.state()
        state.max_steps = 3  # Force short episode

        for _ in range(4):
            obs, reward, done, info = env.step(
                ReviewAction(action_type=ActionType.READ_CLAUSE, clause_id="nda-01")
            )
        # Should be done by now
        assert done or env.state().step_count >= 3


class TestGraders:
    def test_nda_perfect_score(self):
        env = LegalReviewEnvironment()
        env.reset("nda_standard")

        # Optimal play
        env.step(ReviewAction(action_type=ActionType.READ_CLAUSE, clause_id="nda-05"))
        env.step(ReviewAction(
            action_type=ActionType.FLAG_ISSUE,
            clause_id="nda-05",
            issue_type=IssueType.LIABILITY_CAP,
            description="Uncapped liability"
        ))
        env.step(ReviewAction(
            action_type=ActionType.PROPOSE_REDLINE,
            clause_id="nda-05",
            proposed_text="Liability capped at fees paid in prior 12 months."
        ))
        env.step(ReviewAction(
            action_type=ActionType.REQUEST_REVISION,
            description="Liability clause must be fixed"
        ))

        result = env.final_score()
        assert result.score >= 0.70, f"Expected >= 0.70, got {result.score}"

    def test_nda_zero_score_approve_bad_doc(self):
        env = LegalReviewEnvironment()
        env.reset("nda_standard")
        env.step(ReviewAction(action_type=ActionType.APPROVE_DOCUMENT))

        result = env.final_score()
        assert result.score < 0.30, f"Approving bad doc should score low, got {result.score}"

    def test_saas_catches_red_herring(self):
        env = LegalReviewEnvironment()
        env.reset("saas_agreement")
        # Flag payment terms (red herring) — should penalize
        env.step(ReviewAction(
            action_type=ActionType.FLAG_ISSUE,
            clause_id="saas-02",
            issue_type=IssueType.PAYMENT_TERMS,
            description="Payment terms seem unusual"
        ))
        result = env.final_score()
        assert result.breakdown.get("avoided_red_herring", 0.10) == 0.0

    def test_loi_requires_definitions_check(self):
        env = LegalReviewEnvironment()
        env.reset("acquisition_loi")
        # Without checking definitions, earnout trap is harder to find
        env.step(ReviewAction(action_type=ActionType.REJECT_DOCUMENT))
        result = env.final_score()
        assert result.breakdown.get("checked_definitions", 1.0) == 0.0


class TestHTTPServer:
    """Integration tests against the running FastAPI server."""

    @pytest.fixture
    def base_url(self):
        return os.getenv("LEGAL_ENV_URL", "http://localhost:7860")

    def test_health_check(self, base_url):
        import httpx
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5)
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except Exception:
            pytest.skip("Server not running — skipping HTTP tests")

    def test_reset_and_step(self, base_url):
        import httpx
        try:
            resp = httpx.post(f"{base_url}/reset", json={"task_name": "nda_standard"}, timeout=10)
        except Exception:
            pytest.skip("Server not running")

        assert resp.status_code == 200
        data = resp.json()
        episode_id = data["episode_id"]
        obs = data["observation"]
        assert obs["task_name"] == "nda_standard"

        # Take a step
        step_resp = httpx.post(f"{base_url}/step", json={
            "episode_id": episode_id,
            "action": {"action_type": "read_clause", "clause_id": "nda-01"}
        }, timeout=10)
        assert step_resp.status_code == 200
        step_data = step_resp.json()
        assert step_data["reward"] > 0
