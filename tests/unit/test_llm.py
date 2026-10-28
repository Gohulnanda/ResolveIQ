"""LLM provider tests use httpx.MockTransport; no real network calls are made."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.services.llm import ResolutionGenerator

SECRET = "sk-test-secret-value"


@pytest.fixture(scope="module")
def hits(container, vpn_ticket_text="Cannot connect to VPN from home. VPN client times out"):
    found = container.knowledge_base.search(vpn_ticket_text, top_k=3, min_score=0.35)
    assert found
    return found


def make_generator(handler, provider="anthropic", **extra) -> tuple[ResolutionGenerator, list]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    settings = Settings(
        _env_file=None, llm_provider=provider, llm_api_key=SECRET, llm_model="test-model", **extra
    )
    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return ResolutionGenerator(settings, client), seen


def anthropic_reply(payload) -> httpx.Response:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={"content": [{"type": "text", "text": text}]})


def good_payload(source: str) -> dict:
    return {
        "summary": "Check the connection and update the client.",
        "steps": [{"text": "Reconnect with the primary profile.", "source": source}],
        "escalation": None,
    }


def test_demo_mode_is_labelled_and_grounded(hits, vpn_ticket):
    generator = ResolutionGenerator(Settings(_env_file=None, llm_provider="demo"))
    result = generator.generate(vpn_ticket, hits)
    assert result.mode == "demo" and "no LLM" in result.generated_by
    assert "not LLM-generated" in result.summary
    retrieved = {h.article.article_id for h in hits}
    assert result.steps and set(result.citations) <= retrieved
    assert all(step.source_id in retrieved for step in result.steps)


def test_no_hits_gives_no_steps_and_no_citations(vpn_ticket):
    result = ResolutionGenerator(Settings(_env_file=None)).generate(vpn_ticket, [])
    assert result.steps == [] and result.citations == []
    assert "No relevant" in result.summary


def test_llm_success_with_code_fence(hits, vpn_ticket):
    source = hits[0].article.article_id
    reply = "```json\n" + json.dumps(good_payload(source)) + "\n```"
    generator, seen = make_generator(lambda r: anthropic_reply(reply))
    result = generator.generate(vpn_ticket, hits)
    assert result.mode == "llm" and result.citations == [source]
    assert result.generated_by == "LLM: anthropic/test-model"
    request = seen[0]
    assert request.headers["x-api-key"] == SECRET
    body = json.loads(request.content)
    assert "<ticket>" in body["messages"][0]["content"]
    assert SECRET not in result.model_dump_json()


def test_openai_compatible_success(hits, vpn_ticket):
    source = hits[0].article.article_id

    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(good_payload(source))}}]}
        )

    generator, _ = make_generator(handler, provider="openai")
    assert generator.generate(vpn_ticket, hits).mode == "llm"


@pytest.mark.parametrize(
    "handler, reason",
    [
        (lambda r: httpx.Response(429, json={}), "rate_limited"),
        (lambda r: httpx.Response(500, json={}), "provider_error"),
        (lambda r: httpx.Response(401, json={}), "auth_error"),
        (lambda r: httpx.Response(200, content=b"<html>not json</html>"), "invalid_response"),
        (lambda r: anthropic_reply("I cannot help with that."), "invalid_response"),
        (lambda r: anthropic_reply({"summary": "x", "steps": []}), "invalid_response"),
        (lambda r: httpx.Response(200, json={"unexpected": True}), "invalid_response"),
    ],
)
def test_provider_failures_fall_back_and_say_so(hits, vpn_ticket, handler, reason):
    generator, _ = make_generator(handler)
    result = generator.generate(vpn_ticket, hits)
    assert result.mode == "demo_fallback"
    assert result.fallback_reason == reason
    assert "no LLM was used" in result.generated_by
    assert result.steps and set(result.citations) <= {h.article.article_id for h in hits}


def test_timeout_and_network_errors_fall_back(hits, vpn_ticket):
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("down", request=request)

    assert make_generator(timeout)[0].generate(vpn_ticket, hits).fallback_reason == "timeout"
    assert make_generator(refused)[0].generate(vpn_ticket, hits).fallback_reason == "network_error"


def test_citation_to_unretrieved_article_is_rejected(hits, vpn_ticket):
    generator, _ = make_generator(lambda r: anthropic_reply(good_payload("KB-999")))
    result = generator.generate(vpn_ticket, hits)
    assert result.mode == "demo_fallback" and result.fallback_reason == "invalid_citation"
    assert "KB-999" not in result.citations


def test_provider_without_key_or_model_falls_back_without_calling_out(hits, vpn_ticket):
    def must_not_be_called(request):
        raise AssertionError("no request should be made")

    settings = Settings(_env_file=None, llm_provider="anthropic")  # no key, no model
    generator = ResolutionGenerator(
        settings, httpx.Client(transport=httpx.MockTransport(must_not_be_called))
    )
    result = generator.generate(vpn_ticket, hits)
    assert result.mode == "demo_fallback" and result.fallback_reason == "llm_not_configured"
