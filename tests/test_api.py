"""HTTP-level tests: contracts, validation, error handling and safety."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings

VALID = {
    "title": "Cannot connect to VPN from home",
    "description": "The VPN client times out after I enter my credentials. Home internet works.",
}


def test_health_and_docs(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "version" in body
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/health", "/ready", "/api/v1/knowledge-base", "/api/v1/tickets/analyze",
        "/api/v1/tickets/classify", "/api/v1/knowledge-base/search",
    ):
        assert path in paths


def test_ready_reports_dependencies_honestly(client):
    response = client.get("/ready")
    assert response.status_code == 200
    checks = response.json()["checks"]
    assert checks["knowledge_base"]["articles"] >= 15
    assert checks["knowledge_base"]["neural_embeddings_in_use"] is False
    assert checks["llm"]["mode"] == "demo" and checks["llm"]["configured"] is False


def test_knowledge_base_list_and_detail(client):
    articles = client.get("/api/v1/knowledge-base").json()
    assert len(articles) >= 15 and {"article_id", "title", "category", "summary"} <= set(articles[0])
    detail = client.get(f"/api/v1/knowledge-base/{articles[0]['article_id']}")
    assert detail.status_code == 200 and detail.json()["resolution_steps"]
    missing = client.get("/api/v1/knowledge-base/KB-000")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_search_endpoint(client):
    body = client.post("/api/v1/knowledge-base/search", json={"query": "printer offline queue", "top_k": 2}).json()
    assert body["retrieval_method"] == "tfidf-lsa-hybrid"
    assert body["results"] and body["results"][0]["article_id"] == "KB-005"
    assert len(body["results"]) <= 2

    empty = client.post("/api/v1/knowledge-base/search", json={"query": "qwerty zzzz"}).json()
    assert empty["results"] == [] and empty["message"]


@pytest.mark.parametrize(
    "payload", [{"query": "a"}, {"query": "ok query", "top_k": 0}, {"query": "ok query", "top_k": 11}, {}]
)
def test_search_validation(client, payload):
    response = client.post("/api/v1/knowledge-base/search", json=payload)
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_classify_endpoint(client):
    response = client.post("/api/v1/tickets/classify", json={**VALID, "urgency": "high", "impact": "team"})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Network" and 0 < body["confidence"] <= 1
    assert body["predicted_priority"]["level"] in {"P1", "P2", "P3", "P4"}
    assert body["user_provided"] == {"urgency": "high", "impact": "team", "affected_system": None}
    assert "retrieved" not in body


def test_analyze_endpoint_full_contract(client):
    response = client.post("/api/v1/tickets/analyze", json={**VALID, "ticket_id": "T-100"})
    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "T-100"
    assert body["kb_match_found"] and body["retrieved"][0]["article_id"] == "KB-001"
    resolution = body["resolution"]
    assert resolution["mode"] == "demo" and resolution["steps"]
    assert set(resolution["citations"]) <= {r["article_id"] for r in body["retrieved"]}
    assert body["processing_time_ms"] >= 0
    assert "X-Process-Time-Ms" in response.headers


def test_low_confidence_ticket_is_flagged(client):
    body = client.post(
        "/api/v1/tickets/analyze",
        json={"title": "asdf qwer zxcv", "description": "lorem ipsum dolor sit amet"},
    ).json()
    assert body["low_confidence"] and body["category"] == "Other"
    assert body["warnings"] and not body["kb_match_found"]


@pytest.mark.parametrize(
    "payload, field",
    [
        ({**VALID, "title": ""}, "title"),
        ({**VALID, "title": "   "}, "title"),
        ({**VALID, "description": "too short"}, "description"),
        ({**VALID, "description": "x" * 5001}, "description"),
        ({**VALID, "urgency": "asap"}, "urgency"),
        ({**VALID, "ticket_id": "../etc/passwd"}, "ticket_id"),
        ({**VALID, "extra": "nope"}, "extra"),
        ({"description": VALID["description"]}, "title"),
    ],
)
def test_ticket_validation_errors_are_clear(client, payload, field):
    for endpoint in ("analyze", "classify"):
        response = client.post(f"/api/v1/tickets/{endpoint}", json=payload)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert any(d["field"] == field for d in error["details"])
        assert '"input"' not in response.text  # submitted values are not echoed back


def test_malformed_json_and_wrong_types(client):
    bad = client.post("/api/v1/tickets/analyze", content=b"{not json", headers={"content-type": "application/json"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
    wrong = client.post("/api/v1/tickets/analyze", json=["a", "list"])
    assert wrong.status_code == 422


def test_oversized_body_is_rejected(client):
    huge = {**VALID, "description": "x" * 40_000}
    response = client.post("/api/v1/tickets/analyze", json=huge)
    assert response.status_code == 413 and response.json()["error"]["code"] == "payload_too_large"


def test_missing_content_length_is_rejected(client):
    def body():
        yield b'{"title": "abc"}'

    response = client.post("/api/v1/tickets/analyze", content=body(), headers={"content-type": "application/json"})
    assert response.status_code == 411


def test_unknown_route_and_method_use_error_envelope(client):
    assert client.get("/nope").json()["error"]["code"] == "not_found"
    assert client.get("/api/v1/tickets/analyze").status_code == 405


def test_unicode_and_markup_inputs_are_handled(client):
    response = client.post(
        "/api/v1/tickets/analyze",
        json={"title": "Imprimante hors ligne <script>alert(1)</script>",
              "description": "L'imprimante est hors ligne depuis ce matin 😀 \u0000 ' OR 1=1 --"},
    )
    assert response.status_code == 200


def test_model_info_is_real_metrics(client):
    body = client.get("/api/v1/model-info").json()
    assert body["holdout"]["n_test"] > 0 and "synthetic" in body["note"].lower()


def test_unhandled_errors_do_not_leak_details(settings):
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        def boom(_ticket):
            raise RuntimeError("secret-internal-detail /srv/app/db.py")

        app.state.container.analysis.analyze = boom  # type: ignore[method-assign]
        response = test_client.post("/api/v1/tickets/analyze", json=VALID)
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "error", "message": "An internal error occurred."}}
    assert "secret-internal-detail" not in response.text and "Traceback" not in response.text


def test_startup_failure_is_reported_not_hidden(tmp_path_factory, tmp_path):
    broken = make_settings(tmp_path_factory, kb_dir=tmp_path)  # empty KB directory
    with TestClient(create_app(broken)) as test_client:
        assert test_client.get("/health").status_code == 200
        ready = test_client.get("/ready")
        assert ready.status_code == 503 and ready.json()["status"] == "not_ready"
        assert "/" not in ready.json()["checks"]["startup"]["detail"]  # no paths leaked
        assert test_client.post("/api/v1/tickets/analyze", json=VALID).status_code == 503


def test_api_key_never_appears_in_responses(tmp_path_factory):
    settings = make_settings(
        tmp_path_factory, llm_provider="anthropic", llm_api_key="sk-live-should-never-leak", llm_model="m"
    )
    with TestClient(create_app(settings)) as test_client:
        ready = test_client.get("/ready")
        assert ready.json()["checks"]["llm"]["configured"] is True
        assert "sk-live-should-never-leak" not in ready.text
        assert "sk-live-should-never-leak" not in test_client.get("/openapi.json").text
