"""
FastAPI server for the Legal Document Review OpenEnv environment.
Implements the OpenEnv standard endpoints: /reset, /step, /state, /health
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ReviewAction, ReviewObservation, EpisodeState
from server.environment import LegalReviewEnvironment

app = FastAPI(
    title="Legal Document Review — OpenEnv",
    description=(
        "An OpenEnv environment that simulates real-world legal contract review. "
        "Agents must identify issues, flag problematic clauses, propose redlines, "
        "and determine correct document disposition across NDA, SaaS, and M&A tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory episode store (keyed by episode_id)
_episodes: Dict[str, LegalReviewEnvironment] = {}
_active_episode: Optional[str] = None   # single-session shortcut


# ──────────────────────────────────────────────────────
# Request / Response schemas for HTTP
# ──────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "nda_standard"


class StepRequest(BaseModel):
    episode_id: Optional[str] = None
    action: ReviewAction


class FinalScoreResponse(BaseModel):
    episode_id: str
    score: float
    breakdown: dict
    summary: str


# ──────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "legal-document-review", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "name": "Legal Document Review OpenEnv",
        "tasks": ["nda_standard", "saas_agreement", "acquisition_loi"],
        "endpoints": ["/reset", "/step", "/state", "/score", "/health"],
        "docs": "/docs",
    }


# ✅ FIXED RESET ENDPOINT (important change)
@app.post("/reset")
async def reset(req: Optional[ResetRequest] = Body(default=None)) -> dict:
    global _active_episode

    task_name = "nda_standard"
    if req and req.task_name:
        task_name = req.task_name

    env = LegalReviewEnvironment()
    obs = env.reset(task_name)
    episode_id = str(uuid.uuid4())

    _episodes[episode_id] = env
    _active_episode = episode_id

    return {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
async def step(req: StepRequest) -> dict:
    episode_id = req.episode_id or _active_episode
    if not episode_id or episode_id not in _episodes:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found. Call /reset first.")

    env = _episodes[episode_id]
    obs, reward, done, info = env.step(req.action)

    return {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state(episode_id: Optional[str] = None) -> dict:
    eid = episode_id or _active_episode
    if not eid or eid not in _episodes:
        raise HTTPException(status_code=404, detail="No active episode. Call /reset first.")
    return _episodes[eid].state().model_dump()


@app.get("/score")
async def score(episode_id: Optional[str] = None) -> FinalScoreResponse:
    eid = episode_id or _active_episode
    if not eid or eid not in _episodes:
        raise HTTPException(status_code=404, detail="No active episode.")
    result = _episodes[eid].final_score()
    return FinalScoreResponse(
        episode_id=eid,
        score=result.score,
        breakdown=result.breakdown,
        summary=result.summary,
    )


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"name": "nda_standard",    "difficulty": "easy",      "description": "Standard mutual NDA with one uncapped liability trap"},
            {"name": "saas_agreement",  "difficulty": "medium",    "description": "SaaS contract with mixed real issues + red herrings"},
            {"name": "acquisition_loi", "difficulty": "hard",      "description": "M&A LOI with illusory earnout + unlimited exclusivity trap"},
        ]
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()