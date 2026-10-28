"""REST endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.config import APP_VERSION
from app.models.schemas import (
    AnalysisResponse,
    ArticleDetail,
    ArticleSummary,
    ClassificationResponse,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    TicketRequest,
)
from app.services.analysis import hit_to_schema
from app.services.container import Container

router = APIRouter()
_STARTED = time.monotonic()

_ERRORS = {
    413: {"model": ErrorResponse, "description": "Request body too large"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    503: {"model": ErrorResponse, "description": "Service not ready"},
}


def get_container(request: Request) -> Container:
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return container


@router.get("/health", tags=["system"], summary="Liveness check")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "uptime_seconds": round(time.monotonic() - _STARTED, 1),
    }


@router.get("/ready", tags=["system"], summary="Readiness and dependency status")
def ready(request: Request) -> JSONResponse:
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        error = getattr(request.app.state, "startup_error", "startup has not completed")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": {"startup": {"ok": False, "detail": error}}},
        )
    kb = container.knowledge_base
    llm = container.generator
    holdout = container.classifier.metrics.get("holdout", {})
    return JSONResponse(
        content={
            "status": "ready",
            "checks": {
                "classifier": {
                    "ok": True,
                    "holdout_macro_f1": holdout.get("macro_f1"),
                    "trained_on": "synthetic sample tickets",
                },
                "knowledge_base": {
                    "ok": True,
                    "articles": len(kb.articles),
                    "retrieval_method": kb.method,
                    "retrieval_description": kb.embedder.description,
                    "neural_embeddings_in_use": kb.method.startswith("sentence-transformers"),
                    "embedding_fallback_reason": kb.fallback_reason,
                },
                "llm": {
                    "ok": True,
                    "provider": llm.provider,
                    "mode": "llm" if llm.configured else "demo",
                    "configured": llm.configured,
                    "note": "Credentials are not verified until a request is made.",
                },
            },
        }
    )


@router.get(
    "/api/v1/knowledge-base",
    response_model=list[ArticleSummary],
    tags=["knowledge-base"],
    summary="List knowledge-base articles",
    responses={503: _ERRORS[503]},
)
def list_articles(request: Request) -> list[ArticleSummary]:
    kb = get_container(request).knowledge_base
    return [
        ArticleSummary(
            article_id=a.article_id, title=a.title, category=a.category,
            tags=list(a.tags), summary=a.summary,
        )
        for a in kb.articles
    ]


@router.get(
    "/api/v1/knowledge-base/{article_id}",
    response_model=ArticleDetail,
    tags=["knowledge-base"],
    summary="Get one knowledge-base article",
    responses={404: {"model": ErrorResponse}, 503: _ERRORS[503]},
)
def get_article(article_id: str, request: Request) -> ArticleDetail:
    article = get_container(request).knowledge_base.get(article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found.")
    return ArticleDetail(
        article_id=article.article_id, title=article.title, category=article.category,
        tags=list(article.tags), summary=article.summary,
        sections=article.sections, resolution_steps=article.steps,
    )


@router.post(
    "/api/v1/knowledge-base/search",
    response_model=SearchResponse,
    tags=["knowledge-base"],
    summary="Search the knowledge base",
    responses={413: _ERRORS[413], 422: _ERRORS[422], 503: _ERRORS[503]},
)
def search_kb(body: SearchRequest, request: Request) -> SearchResponse:
    container = get_container(request)
    min_score = container.settings.retrieval_min_score
    hits = container.knowledge_base.search(body.query, body.top_k, min_score)
    return SearchResponse(
        query=body.query,
        retrieval_method=container.knowledge_base.method,
        min_score=min_score,
        results=[hit_to_schema(h) for h in hits],
        message=None if hits else "No article scored above the relevance threshold.",
    )


@router.post(
    "/api/v1/tickets/classify",
    response_model=ClassificationResponse,
    tags=["tickets"],
    summary="Classify a ticket (category and priority only)",
    responses={413: _ERRORS[413], 422: _ERRORS[422], 503: _ERRORS[503]},
)
def classify_ticket(ticket: TicketRequest, request: Request) -> ClassificationResponse:
    return get_container(request).analysis.classify(ticket)


@router.post(
    "/api/v1/tickets/analyze",
    response_model=AnalysisResponse,
    tags=["tickets"],
    summary="Classify, retrieve knowledge and suggest a grounded resolution",
    responses={413: _ERRORS[413], 422: _ERRORS[422], 503: _ERRORS[503]},
)
def analyze_ticket(ticket: TicketRequest, request: Request) -> AnalysisResponse:
    return get_container(request).analysis.analyze(ticket)


@router.get(
    "/api/v1/model-info",
    tags=["system"],
    summary="Classifier training metadata and hold-out metrics",
    responses={503: _ERRORS[503]},
)
def model_info(request: Request) -> dict:
    return get_container(request).classifier.metrics
