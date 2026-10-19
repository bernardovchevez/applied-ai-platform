"""Pydantic request/response models for BFF routes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

class MetadataFilters(BaseModel):
    doc_type: str | None = None
    confidentiality: str | None = None
    rights_tier: Literal["cleared", "pending", "denied", "excluded", "unknown"] | None = None
    department: str | None = None


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    filters: MetadataFilters | None = None


class SourceCitation(BaseModel):
    file_name: str
    doc_type: str | None = None
    rights_tier: str | None = None
    sharepoint_path: str | None = None
    chunk_index: int | None = None
    distance: float | None = None


class ChunkHitOut(BaseModel):
    chunk_id: str
    text: str
    source: SourceCitation


class RetrieveResponse(BaseModel):
    question: str
    chunks: list[ChunkHitOut]
    sources: list[SourceCitation]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    filters: MetadataFilters | None = None
    format: Literal["full", "compact"] = "full"


FeedbackRating = Literal["up", "down", "approve", "reject"]


class FeedbackRequest(BaseModel):
    session_id: str
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    rating: FeedbackRating
    sources: list[SourceCitation] | None = None
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: str
    created_at: str


class EvalBackendScore(BaseModel):
    backend: str
    passed: int
    total: int
    note: str | None = None


class PromptfooSummary(BaseModel):
    passed: int
    total: int
    failures: int | None = None
    errors: int | None = None
    eval_id: str | None = None
    timestamp: str | None = None


class EvalSummaryResponse(BaseModel):
    gold_questions: int
    agent_scenarios: int = 0
    keyword_eval: list[EvalBackendScore]
    keyword_eval_timestamp: str | None = None
    ragas_eval: list[EvalBackendScore] = Field(default_factory=list)
    ragas_eval_timestamp: str | None = None
    ragas_judge_model: str | None = None
    agent_eval: EvalBackendScore | None = None
    agent_eval_timestamp: str | None = None
    promptfoo: PromptfooSummary | None = None
    sources: dict[str, str] | None = None


class PgvectorHealthOk(BaseModel):
    status: Literal["ok"]
    chunk_count: int
    document_count: int


class PgvectorHealthFailed(BaseModel):
    status: Literal["unreachable"]
    error: str


PgvectorHealth = Annotated[
    PgvectorHealthOk | PgvectorHealthFailed,
    Field(discriminator="status"),
]


class LlmHealthOk(BaseModel):
    provider: str
    status: Literal["ok"]


class LlmHealthFailed(BaseModel):
    provider: str
    status: Literal["unreachable"]
    error: str


LlmHealth = Annotated[
    LlmHealthOk | LlmHealthFailed,
    Field(discriminator="status"),
]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: Literal["localhost", "stage", "production"] = "localhost"
    state_store: Literal["sqlite", "postgres"] = "sqlite"
    pgvector: PgvectorHealth
    llm: LlmHealth


AgentResumeDecision = Literal["approve", "reject"]


class AgentRunRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1)
    auto_approve: bool = False
    thread_id: str | None = None
    resume: AgentResumeDecision | None = None


class AgentRunResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    tool_trace: list[dict[str, Any]]
    status: Literal["approved", "completed", "rejected"] = "completed"


class AgentPausedResponse(BaseModel):
    thread_id: str
    message: str
    proposed_output: str | None = None
    prompt: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
