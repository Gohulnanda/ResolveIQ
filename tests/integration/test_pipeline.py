"""End-to-end analysis through the service layer (no HTTP)."""

import pytest

from app.models.schemas import TicketRequest

CASES = [
    ("Cannot connect to VPN", "The VPN client shows a timeout error after I enter my credentials.", "Network", "KB-001"),
    ("Printer is offline", "The printer shows offline although it is powered on and has paper.", "Hardware", "KB-005"),
    ("Forgot my password", "It says my account is locked and I forgot my password so I cannot sign in.", "Account and Access", "KB-009"),
    ("Database connection timeout", "The application logs show connection pool exhausted errors on the SQL server.", "Database", "KB-014"),
    ("Outlook not receiving emails", "Outlook shows disconnected in the status bar and the outbox is stuck.", "Email and Collaboration", "KB-012"),
]


@pytest.mark.parametrize("title, description, category, article", CASES)
def test_analysis_classifies_retrieves_and_cites(container, title, description, category, article):
    result = container.analysis.analyze(TicketRequest(title=title, description=description))
    assert result.category == category
    assert result.retrieved[0].article_id == article
    assert result.kb_match_found
    retrieved_ids = {r.article_id for r in result.retrieved}
    assert result.resolution.mode == "demo"
    assert result.resolution.citations and set(result.resolution.citations) <= retrieved_ids
    assert all(s.source_id in retrieved_ids for s in result.resolution.steps)
    assert result.retrieval_method == "tfidf-lsa-hybrid"
    assert result.processing_time_ms >= 0


def test_unanswerable_ticket_reports_no_knowledge(container):
    ticket = TicketRequest(
        title="Cafeteria menu", description="What is on the cafeteria lunch menu this Thursday please?"
    )
    result = container.analysis.analyze(ticket)
    assert not result.kb_match_found and result.retrieved == []
    assert result.resolution.steps == [] and result.resolution.citations == []
    assert "No relevant" in result.resolution.summary


def test_user_provided_values_stay_separate_from_predictions(container):
    ticket = TicketRequest(
        title="Printer jam", description="Paper jam error on the reception printer again.",
        urgency="low", impact="individual", affected_system="PRN-02", ticket_id="MY-1",
    )
    result = container.analysis.analyze(ticket)
    assert result.ticket_id == "MY-1"
    assert result.user_provided.urgency == "low"
    assert result.user_provided.affected_system == "PRN-02"
    assert result.predicted_priority.method.startswith("rule-based")


def test_generated_ticket_id_when_missing(container, vpn_ticket):
    first = container.analysis.classify(vpn_ticket).ticket_id
    second = container.analysis.classify(vpn_ticket).ticket_id
    assert first.startswith("TKT-") and first != second


def test_failing_llm_does_not_break_analysis(container, vpn_ticket):
    import httpx

    from app.core.config import Settings
    from app.services.analysis import TicketAnalysisService
    from app.services.llm import ResolutionGenerator

    settings = Settings(_env_file=None, llm_provider="openai", llm_api_key="k", llm_model="m")
    generator = ResolutionGenerator(
        settings, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    )
    service = TicketAnalysisService(container.classifier, container.knowledge_base, generator, settings)
    result = service.analyze(vpn_ticket)
    assert result.resolution.mode == "demo_fallback"
    assert result.resolution.fallback_reason == "provider_error"
    assert result.resolution.steps
