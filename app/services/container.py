"""Builds the long-lived services used by the API."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.retrieval.embeddings import build_embedder
from app.retrieval.knowledge_base import KnowledgeBase, load_articles
from app.services.analysis import TicketAnalysisService
from app.services.classifier import TicketClassifier
from app.services.llm import ResolutionGenerator


@dataclass
class Container:
    settings: Settings
    classifier: TicketClassifier
    knowledge_base: KnowledgeBase
    generator: ResolutionGenerator
    analysis: TicketAnalysisService


def build_container(settings: Settings) -> Container:
    classifier = TicketClassifier.load_or_train(
        settings.model_path, settings.training_data_path, settings.classifier_min_confidence
    )
    embedder, fallback_reason = build_embedder(settings.embedding_backend, settings.embedding_model)
    knowledge_base = KnowledgeBase(load_articles(settings.kb_dir), embedder, fallback_reason)
    generator = ResolutionGenerator(settings)
    analysis = TicketAnalysisService(classifier, knowledge_base, generator, settings)
    return Container(settings, classifier, knowledge_base, generator, analysis)
