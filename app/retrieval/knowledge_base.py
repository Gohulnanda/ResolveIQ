"""Markdown knowledge base: parsing, section-level chunking, and vector search."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
_STEP = re.compile(r"^\s*\d+\.\s+(.+?)\s*$", re.M)
STEPS_SECTION = "Resolution steps"
ESCALATION_SECTION = "Escalate when"
MAX_PASSAGE_CHARS = 700
MIN_SHARED_TERMS = 2
_WORD = re.compile(r"[a-z0-9]{2,}")


def content_terms(text: str) -> set[str]:
    """Lower-cased words of 2+ characters, excluding English stop words."""
    return {w for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS}


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    category: str
    tags: tuple[str, ...]
    sections: dict[str, str]

    @property
    def summary(self) -> str:
        return self.sections.get("Summary", "")

    @property
    def steps(self) -> list[str]:
        return _STEP.findall(self.sections.get(STEPS_SECTION, ""))

    @property
    def escalation(self) -> str | None:
        return self.sections.get(ESCALATION_SECTION) or None


@dataclass(frozen=True)
class SearchHit:
    article: Article
    score: float
    matched_section: str
    matched_passage: str


@dataclass(frozen=True)
class _Chunk:
    article_index: int
    section: str
    text: str


def parse_article(raw: str, source: str) -> Article:
    match = _FRONT_MATTER.match(raw.replace("\r\n", "\n"))
    if not match:
        raise ValueError(f"{source}: missing '---' front matter block")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    for required in ("id", "title", "category"):
        if not meta.get(required):
            raise ValueError(f"{source}: front matter is missing '{required}'")

    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in match.group(2).splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current, buffer = line[3:].strip(), []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    if STEPS_SECTION not in sections:
        raise ValueError(f"{source}: missing '## {STEPS_SECTION}' section")

    tags = tuple(t.strip() for t in meta.get("tags", "").split(",") if t.strip())
    return Article(meta["id"], meta["title"], meta["category"], tags, sections)


def load_articles(kb_dir: Path) -> list[Article]:
    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No knowledge-base articles found in {kb_dir}")
    articles = [parse_article(p.read_text(encoding="utf-8"), p.name) for p in paths]
    ids = [a.article_id for a in articles]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate knowledge-base article IDs: {duplicates}")
    return articles


class KnowledgeBase:
    """Indexes each article section as a passage and ranks articles by their best passage."""

    def __init__(self, articles: list[Article], embedder: Embedder, fallback_reason: str | None = None):
        self.articles = articles
        self.embedder = embedder
        self.fallback_reason = fallback_reason
        self._by_id = {a.article_id: a for a in articles}

        self._chunks: list[_Chunk] = []
        for index, article in enumerate(articles):
            for section, body in article.sections.items():
                if body:
                    text = f"{article.title}. {' '.join(article.tags)}. {section}. {body}"
                    self._chunks.append(_Chunk(index, section, text))

        self._article_terms = [
            content_terms(" ".join([a.title, *a.tags, *a.sections.values()])) for a in articles
        ]
        embedder.fit([c.text for c in self._chunks])
        self._matrix = embedder.encode([c.text for c in self._chunks])

    @property
    def method(self) -> str:
        return self.embedder.name

    def get(self, article_id: str) -> Article | None:
        return self._by_id.get(article_id)

    def search(self, query: str, top_k: int, min_score: float) -> list[SearchHit]:
        """Brute-force cosine search (exact; the corpus is tiny). Returns at most ``top_k``
        articles whose best passage scores at least ``min_score``, best first.

        With a lexical backend a match must also share at least ``MIN_SHARED_TERMS`` distinct
        content words with the article (fewer if the query has fewer), because a single rare
        shared word can otherwise produce a deceptively high cosine score."""
        query_terms = content_terms(query)
        needed = min(MIN_SHARED_TERMS, len(query_terms))
        scores = self._matrix @ self.embedder.encode([query])[0]
        best: dict[int, tuple[float, _Chunk]] = {}
        for chunk, score in zip(self._chunks, scores):
            current = best.get(chunk.article_index)
            if current is None or score > current[0]:
                best[chunk.article_index] = (float(score), chunk)

        hits = [
            SearchHit(
                article=self.articles[idx],
                score=round(float(np.clip(score, 0.0, 1.0)), 4),
                matched_section=chunk.section,
                matched_passage=self.articles[idx].sections[chunk.section][:MAX_PASSAGE_CHARS],
            )
            for idx, (score, chunk) in best.items()
            if score >= min_score
            and (
                not self.embedder.requires_term_overlap
                or len(query_terms & self._article_terms[idx]) >= needed
            )
        ]
        hits.sort(key=lambda h: (-h.score, h.article.article_id))
        return hits[:top_k]
