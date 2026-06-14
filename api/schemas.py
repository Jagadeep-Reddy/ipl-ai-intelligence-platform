"""
api/schemas.py
──────────────
Pydantic v2 typed schemas for all API request/response contracts.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    message: str = Field(..., description="User's natural language query")
    session_id: str = Field(default="default", description="Conversation session ID")


class QueryResponse(BaseModel):
    answer: str
    backend_used: str  # RAG | SQL | ML | MATCHUP | TEAMVSTEAM
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    cached: bool = False
    latency_ms: int | None = None


class BallEvent(BaseModel):
    """Schema for a single live ball event from the cricket API WebSocket."""
    match_id: str
    batter: str
    bowler: str
    non_striker: str
    over: int
    ball_num: int
    batting_team: str
    bowling_team: str
    # Contextual match state
    cum_runs: int = 0
    wickets_fallen: int = 0
    current_rr: float = 0.0
    required_rr: float = 0.0
    target: int = 0
    # Bowler spell state
    bowler_spell_balls: int = 0
    bowler_spell_runs: int = 0
    bowler_last3_runs: float = 0.0
    # Historical context (pre-loaded from Redis at match start)
    h2h_sr: float = 100.0
    h2h_balls_total: int = 0
    batter_last5_avg: float = 25.0
    bowler_economy: float = 8.0
    bowler_type_enc: int = 0  # 0=pace, 1=spin
    pressure_index: float = 0.0
    # Derived
    phase_enc: int = 1
    inning: int = 1
    season: int = 2025


class IntentVerdict(BaseModel):
    """Schema for real-time intent prediction pushed to frontend."""
    batter: str
    bowler: str
    over: int
    ball_num: int
    intent: str  # aggressive | defensive | neutral | pressure_error
    probabilities: dict[str, float]
    confidence_flag: bool = False
    verdict_text: str
    rag_context_summary: str = ""
    latency_ms: int


class MetricsResponse(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    last_evaluated: str = ""
    total_questions: int = 0
