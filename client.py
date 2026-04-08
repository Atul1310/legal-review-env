"""
OpenEnv client for the Legal Document Review environment.
Supports both async (default) and sync (.sync()) usage.
"""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Any

import httpx

from models import ReviewAction, ReviewObservation, EpisodeState, ActionType


# ──────────────────────────────────────────────────────
# Async Client
# ──────────────────────────────────────────────────────

class LegalReviewEnv:
    """
    Async OpenEnv client for the Legal Document Review environment.

    Usage:
        async with LegalReviewEnv(base_url="http://localhost:7860") as env:
            obs = await env.reset("nda_standard")
            result = await env.step(ReviewAction(action_type=ActionType.READ_CLAUSE, clause_id="nda-01"))
            print(result.observation.action_result)
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._episode_id: Optional[str] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_name: str = "nda_standard") -> ReviewObservation:
        resp = await self._client.post("/reset", json={"task_name": task_name})
        resp.raise_for_status()
        data = resp.json()
        self._episode_id = data["episode_id"]
        return ReviewObservation(**data["observation"])

    async def step(self, action: ReviewAction):
        payload = {
            "episode_id": self._episode_id,
            "action": action.model_dump(),
        }
        resp = await self._client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=ReviewObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )

    async def state(self) -> EpisodeState:
        resp = await self._client.get("/state", params={"episode_id": self._episode_id})
        resp.raise_for_status()
        return EpisodeState(**resp.json())

    async def score(self) -> dict:
        resp = await self._client.get("/score", params={"episode_id": self._episode_id})
        resp.raise_for_status()
        return resp.json()

    def sync(self) -> "SyncLegalReviewEnv":
        return SyncLegalReviewEnv(base_url=self.base_url, timeout=self.timeout)


class StepResult:
    def __init__(self, observation: ReviewObservation, reward: float, done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


# ──────────────────────────────────────────────────────
# Sync Wrapper
# ──────────────────────────────────────────────────────

class SyncLegalReviewEnv:
    """Synchronous wrapper around LegalReviewEnv."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout
        self._async_env: Optional[LegalReviewEnv] = None

    def __enter__(self):
        self._async_env = LegalReviewEnv(self.base_url, self.timeout)
        asyncio.get_event_loop().run_until_complete(self._async_env.__aenter__())
        return self

    def __exit__(self, *args):
        asyncio.get_event_loop().run_until_complete(self._async_env.__aexit__(*args))

    def reset(self, task_name: str = "nda_standard") -> ReviewObservation:
        return asyncio.get_event_loop().run_until_complete(self._async_env.reset(task_name))

    def step(self, action: ReviewAction) -> StepResult:
        return asyncio.get_event_loop().run_until_complete(self._async_env.step(action))

    def state(self) -> EpisodeState:
        return asyncio.get_event_loop().run_until_complete(self._async_env.state())

    def score(self) -> dict:
        return asyncio.get_event_loop().run_until_complete(self._async_env.score())
