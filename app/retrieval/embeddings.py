"""Text-to-vector backends for knowledge-base retrieval.

Two backends exist and each reports an honest ``name``:

* ``HybridTfidfLsaEmbedder`` (default, fully offline). Every vector is the concatenation
  of a TF-IDF vector and an LSA vector (TF-IDF reduced with truncated SVD), each
  L2-normalised and weighted so that the dot product of two vectors equals
  ``w * cosine_tfidf + (1 - w) * cosine_lsa``. LSA captures co-occurrence structure, but
  this is NOT a neural embedding model and should not be described as one.
* ``SentenceTransformerEmbedder`` (optional). Real neural embeddings; requires the
  ``sentence-transformers`` package and a model that is downloadable or cached locally.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.core.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    description: str
    requires_term_overlap: bool  # lexical backends need shared words as evidence of a match

    def fit(self, texts: Sequence[str]) -> None: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HybridTfidfLsaEmbedder:
    name = "tfidf-lsa-hybrid"
    description = (
        "Offline hybrid of TF-IDF cosine and latent semantic analysis (truncated SVD) "
        "cosine. Deterministic; not neural embeddings."
    )
    requires_term_overlap = True

    def __init__(self, n_components: int = 64, lexical_weight: float = 0.5):
        self.n_components = n_components
        self.lexical_weight = lexical_weight
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> None:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english", strip_accents="unicode"
        )
        matrix = tfidf.fit_transform(texts)
        k = max(2, min(self.n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
        svd.fit(matrix)
        self._tfidf, self._svd = tfidf, svd

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._tfidf is None or self._svd is None:
            raise RuntimeError("Embedder must be fitted before encoding.")
        sparse = self._tfidf.transform(texts)
        lexical = normalize(sparse).toarray()
        latent = normalize(self._svd.transform(sparse))
        w = self.lexical_weight
        return np.hstack([np.sqrt(w) * lexical, np.sqrt(1.0 - w) * latent])


class SentenceTransformerEmbedder:
    description = "Neural sentence embeddings via the sentence-transformers library."
    requires_term_overlap = False

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def fit(self, texts: Sequence[str]) -> None:  # pretrained; nothing to fit
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True))


def build_embedder(backend: str, model_name: str) -> tuple[Embedder, str | None]:
    """Return ``(embedder, fallback_reason)``. The reason is set when a neural backend was
    requested (explicitly or via ``auto``) but could not be used."""
    if backend == "hybrid":
        return HybridTfidfLsaEmbedder(), None
    try:
        return SentenceTransformerEmbedder(model_name), None
    except ImportError:
        reason = "sentence-transformers is not installed"
    except Exception as exc:  # noqa: BLE001 - model download/load can fail in many ways
        logger.warning("Could not load embedding model %s: %s", model_name, type(exc).__name__)
        reason = f"embedding model could not be loaded ({type(exc).__name__})"
    if backend == "sentence-transformers":
        logger.warning("sentence-transformers requested but unavailable: %s. Using hybrid.", reason)
    return HybridTfidfLsaEmbedder(), reason
