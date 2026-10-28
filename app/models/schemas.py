"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_TITLE_CHARS = 200
MAX_DESCRIPTION_CHARS = 5000
MAX_SYSTEM_CHARS = 100
MAX_QUERY_CHARS = 500

Urgency = Literal["low", "medium", "high", "critical"]
Impact = Literal["individual", "team", "department", "organization"]
PriorityLevel = Literal["P1", "P2", "P3", "P4"]
ProcessingMode = Literal["demo", "llm", "demo_fallback"]

TicketId = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9._-]{1,64}$")
]


class TicketRequest(BaseModel):
    """A support ticket submitted for classification and/or analysis."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "title": "Cannot connect to VPN from home",
                "description": "The VPN client times out after I enter my credentials. "
                "Home internet works fine for everything else.",
                "affected_system": "LAPTOP-0142",
                "urgency": "high",
                "impact": "individual",
            }
        },
    )

    ticket_id: TicketId | None = Field(None, description="Optional caller-supplied ID.")
    title: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=MAX_TITLE_CHARS),
    ]
    description: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=10, max_length=MAX_DESCRIPTION_CHARS
        ),
    ]
    affected_system: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=MAX_SYSTEM_CHARS)
    ] | None = None
    urgency: Urgency | None = Field(None, description="Urgency as stated by the requester.")
    impact: Impact | None = Field(None, description="Business impact as stated by the requester.")

    @field_validator("urgency", "impact", mode="before")
    @classmethod
    def _normalise_enum(cls, value: Any) -> Any:
        """Accept 'High', ' high ' and '' (treated as not provided)."""
        if isinstance(value, str):
            return value.strip().lower() or None
        return value

    @field_validator("affected_system", "ticket_id", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=MAX_QUERY_CHARS)
    ]
    top_k: int = Field(3, ge=1, le=10)


class CategoryAlternative(BaseModel):
    category: str
    probability: float


class PriorityPrediction(BaseModel):
    level: PriorityLevel
    label: str
    score: int = Field(description="Heuristic points total that produced the level.")
    reasoning: list[str]
    method: str = "rule-based heuristic (not learned from data)"


class UserProvidedContext(BaseModel):
    """Values exactly as the requester supplied them; never overwritten by predictions."""

    urgency: Urgency | None = None
    impact: Impact | None = None
    affected_system: str | None = None


class ClassificationResponse(BaseModel):
    ticket_id: str
    category: str
    confidence: float = Field(description="Model probability of the top class (0-1).")
    low_confidence: bool
    alternatives: list[CategoryAlternative]
    predicted_priority: PriorityPrediction
    user_provided: UserProvidedContext
    warnings: list[str] = []
    classifier_holdout_macro_f1: float | None = Field(
        None,
        description="Macro-F1 of the classifier on its held-out split of the synthetic "
        "sample dataset. Not a production performance estimate.",
    )


class RetrievedArticle(BaseModel):
    article_id: str
    title: str
    category: str
    score: float = Field(description="Retrieval similarity (0-1); see retrieval_method.")
    matched_section: str
    matched_passage: str
    resolution_steps: list[str]
    escalation_criteria: str | None = None
    tags: list[str] = []


class ResolutionStep(BaseModel):
    text: str
    source_id: str | None = Field(None, description="ID of the retrieved article it came from.")


class Resolution(BaseModel):
    summary: str
    steps: list[ResolutionStep]
    citations: list[str] = Field(description="IDs of retrieved articles actually used.")
    mode: ProcessingMode
    generated_by: str
    fallback_reason: str | None = None
    escalation: str | None = None


class AnalysisResponse(ClassificationResponse):
    retrieved: list[RetrievedArticle]
    kb_match_found: bool
    retrieval_method: str
    resolution: Resolution
    processing_time_ms: float


class SearchResponse(BaseModel):
    query: str
    retrieval_method: str
    min_score: float
    results: list[RetrievedArticle]
    message: str | None = None


class ArticleSummary(BaseModel):
    article_id: str
    title: str
    category: str
    tags: list[str]
    summary: str


class ArticleDetail(ArticleSummary):
    sections: dict[str, str]
    resolution_steps: list[str]


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
