import pytest

from app.retrieval.embeddings import HybridTfidfLsaEmbedder, build_embedder
from app.retrieval.knowledge_base import load_articles, parse_article

VALID = """---
id: KB-900
title: Test article
category: Network
tags: a, b
---
## Summary
Short summary.

## Resolution steps
1. First step
2. Second step

## Escalate when
Things are on fire.
"""


@pytest.fixture(scope="module")
def kb(container):
    return container.knowledge_base


def test_parse_article_extracts_fields():
    article = parse_article(VALID, "x.md")
    assert article.article_id == "KB-900"
    assert article.tags == ("a", "b")
    assert article.steps == ["First step", "Second step"]
    assert article.escalation == "Things are on fire."


@pytest.mark.parametrize(
    "raw, message",
    [
        ("no front matter", "front matter"),
        (VALID.replace("id: KB-900\n", ""), "missing 'id'"),
        (VALID.replace("## Resolution steps", "## Other"), "Resolution steps"),
    ],
)
def test_parse_article_rejects_bad_files(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_article(raw, "x.md")


def test_load_articles_detects_duplicates_and_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_articles(tmp_path)
    (tmp_path / "a.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "b.md").write_text(VALID, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_articles(tmp_path)


def test_real_knowledge_base_loads(kb):
    assert len(kb.articles) >= 15
    assert all(a.steps for a in kb.articles)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("VPN keeps timing out when I work from home", "KB-001"),
        ("printer shows offline and jobs are stuck", "KB-005"),
        ("I forgot my password and my account is locked", "KB-009"),
        ("nightly backup failed because the disk is full", "KB-015"),
    ],
)
def test_search_ranks_expected_article_first(kb, query, expected):
    hits = kb.search(query, top_k=3, min_score=0.35)
    assert hits and hits[0].article.article_id == expected
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_search_abstains_on_unrelated_query_and_respects_top_k(kb):
    assert kb.search("qwerty zzzz xxxx", top_k=3, min_score=0.35) == []
    assert len(kb.search("network problem", top_k=2, min_score=0.0)) == 2


def test_method_name_is_honest(kb):
    assert kb.method == "tfidf-lsa-hybrid"
    assert "not neural" in kb.embedder.description.lower()


def test_embedder_requires_fit():
    with pytest.raises(RuntimeError):
        HybridTfidfLsaEmbedder().encode(["x"])


def test_auto_backend_falls_back_with_a_reason_when_neural_unavailable():
    embedder, reason = build_embedder("auto", "sentence-transformers/does-not-exist")
    assert embedder.name == "tfidf-lsa-hybrid"
    assert reason  # fallback is never silent
