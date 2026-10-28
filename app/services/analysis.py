"""Orchestrates classification, priority prediction, retrieval and resolution."""

from __future__ import annotations

import logging
import time
import uuid

from app.core.config import Settings
from app.models.schemas import (
    AnalysisResponse,
    CategoryAlternative,
    ClassificationResponse,
    RetrievedArticle,
    TicketRequest,
    UserProvidedContext,
)
from app.retrieval.knowledge_base import KnowledgeBase, SearchHit
from app.services.classifier import TicketClassifier, compose_text
from app.services.llm import ResolutionGenerator
from app.services.priority import predict_priority

logger = logging.getLogger(__name__)


def hit_to_schema(hit: SearchHit) -> RetrievedArticle:
    article = hit.article
    return RetrievedArticle(
        article_id=article.article_id,
        title=article.title,
        category=article.category,
        score=hit.score,
        matched_section=hit.matched_section,
        matched_passage=hit.matched_passage,
        resolution_steps=article.steps,
        escalation_criteria=article.escalation,
        tags=list(article.tags),
    )


class TicketAnalysisService:
    def __init__(
        self,
        classifier: TicketClassifier,
        knowledge_base: KnowledgeBase,
        generator: ResolutionGenerator,
        settings: Settings,
    ):
        self._classifier = classifier
        self._kb = knowledge_base
        self._generator = generator
        self._settings = settings

    def classify(self, ticket: TicketRequest) -> ClassificationResponse:
        prediction = self._classifier.predict(ticket.title, ticket.description)
        priority = predict_priority(
            compose_text(ticket.title, ticket.description), ticket.urgency, ticket.impact
        )
        return ClassificationResponse(
            ticket_id=ticket.ticket_id or f"TKT-{uuid.uuid4().hex[:8].upper()}",
            category=prediction.category,
            confidence=prediction.confidence,
            low_confidence=prediction.low_confidence,
            alternatives=[CategoryAlternative(category=c, probability=p) for c, p in prediction.alternatives],
            predicted_priority=priority,
            user_provided=UserProvidedContext(
                urgency=ticket.urgency, impact=ticket.impact, affected_system=ticket.affected_system
            ),
            warnings=prediction.warnings,
            classifier_holdout_macro_f1=self._classifier.holdout_macro_f1,
        )

    def analyze(self, ticket: TicketRequest) -> AnalysisResponse:
        started = time.perf_counter()
        classification = self.classify(ticket)
        hits = self._kb.search(
            compose_text(ticket.title, ticket.description),
            top_k=self._settings.retrieval_top_k,
            min_score=self._settings.retrieval_min_score,
        )
        resolution = self._generator.generate(ticket, hits)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        # Log identifiers and outcomes only; never ticket text.
        logger.info(
            "analyzed ticket_id=%s category=%s priority=%s kb_hits=%d mode=%s ms=%.1f",
            classification.ticket_id,
            classification.category,
            classification.predicted_priority.level,
            len(hits),
            resolution.mode,
            elapsed_ms,
        )
        return AnalysisResponse(
            **classification.model_dump(),
            retrieved=[hit_to_schema(h) for h in hits],
            kb_match_found=bool(hits),
            retrieval_method=self._kb.method,
            resolution=resolution,
            processing_time_ms=elapsed_ms,
        )
