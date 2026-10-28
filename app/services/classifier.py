"""Ticket category classification: TF-IDF features + multinomial logistic regression."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import RANDOM_SEED

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = (
    "Network",
    "Hardware",
    "Software",
    "Account and Access",
    "Database",
    "Email and Collaboration",
    "Other",
)
FALLBACK_CATEGORY = "Other"
SHORT_TEXT_WORDS = 6
MIN_KNOWN_TERM_RATIO = 0.30  # below this share of recognised words the text is treated as unintelligible
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
DATA_ORIGIN_NOTE = (
    "Trained on synthetic, template-generated sample tickets included in this repository. "
    "Not real enterprise data; scores do not predict production performance."
)


def compose_text(title: str, description: str) -> str:
    return f"{title.strip()}. {description.strip()}"


def build_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", lowercase=True
                ),
            ),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=seed)),
        ]
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a ticket CSV with columns title, description, category."""
    df = pd.read_csv(path)
    missing = {"title", "description", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    df = df.dropna(subset=["title", "description", "category"]).copy()
    unknown = sorted(set(df["category"]) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"{path}: unknown categories {unknown}")
    if df.empty:
        raise ValueError(f"{path}: no usable rows")
    df["text"] = [compose_text(t, d) for t, d in zip(df["title"], df["description"])]
    return df.reset_index(drop=True)


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Accuracy, macro and per-class precision/recall/F1 and a confusion matrix."""
    labels = list(CATEGORIES)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "n_samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f), 4),
        "per_class": {
            label: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_and_evaluate(
    df: pd.DataFrame, seed: int = RANDOM_SEED, test_size: float = 0.2
) -> tuple[Pipeline, dict[str, Any]]:
    """Evaluate on a stratified hold-out split, then refit on all rows for serving."""
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=test_size, stratify=df["category"], random_state=seed
    )
    eval_model = build_pipeline(seed).fit(x_train, y_train)
    holdout = evaluate_predictions(list(y_test), list(eval_model.predict(x_test)))
    holdout.update(test_size=test_size, n_train=len(x_train), n_test=len(x_test))

    serving_model = build_pipeline(seed).fit(df["text"], df["category"])
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "sklearn_version": sklearn.__version__,
        "n_training_rows": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["category"].value_counts().items()},
        "holdout": holdout,
        "note": DATA_ORIGIN_NOTE,
        "serving_model": "Refit on 100% of rows after hold-out evaluation.",
    }
    return serving_model, metrics


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    low_confidence: bool
    alternatives: list[tuple[str, float]]
    warnings: list[str] = field(default_factory=list)


class TicketClassifier:
    def __init__(self, pipeline: Pipeline, metrics: dict[str, Any], min_confidence: float):
        self._pipeline = pipeline
        self.metrics = metrics
        self.min_confidence = min_confidence

    # --- persistence -------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: Path, min_confidence: float) -> "TicketClassifier":
        artifact = joblib.load(path)
        return cls(artifact["pipeline"], artifact["metrics"], min_confidence)

    @classmethod
    def train(cls, data_path: Path, min_confidence: float) -> "TicketClassifier":
        pipeline, metrics = train_and_evaluate(load_dataset(data_path))
        return cls(pipeline, metrics, min_confidence)

    @classmethod
    def load_or_train(
        cls, model_path: Path, data_path: Path, min_confidence: float
    ) -> "TicketClassifier":
        """Load a saved model; train (deterministically) and save one if it is missing,
        unreadable, or was built with a different scikit-learn version."""
        if model_path.exists():
            try:
                loaded = cls.load(model_path, min_confidence)
                if loaded.metrics.get("sklearn_version") == sklearn.__version__:
                    return loaded
                logger.warning("Saved model built with another scikit-learn version; retraining.")
            except Exception:  # noqa: BLE001 - any unpickling problem means retrain
                logger.exception("Could not load saved model; retraining.")
        classifier = cls.train(data_path, min_confidence)
        try:
            classifier.save(model_path)
        except OSError:
            logger.exception("Could not save trained model to %s; continuing in memory.", model_path)
        return classifier

    # --- inference ---------------------------------------------------------------
    def predict(self, title: str, description: str) -> Prediction:
        text = compose_text(title, description)
        probabilities = self._pipeline.predict_proba([text])[0]
        order = np.argsort(probabilities)[::-1]
        classes = self._pipeline.classes_
        ranked = [(str(classes[i]), float(probabilities[i])) for i in order]
        top_label, top_prob = ranked[0]

        warnings: list[str] = []
        if len(text.split()) < SHORT_TEXT_WORDS:
            warnings.append("Very short ticket text; the prediction may be unreliable.")

        known_ratio = self._known_term_ratio(text)
        unintelligible = known_ratio < MIN_KNOWN_TERM_RATIO
        low = top_prob < self.min_confidence or unintelligible
        category = top_label
        if low:
            category = FALLBACK_CATEGORY
            reason = (
                f"only {known_ratio:.0%} of the words are known to the model"
                if unintelligible
                else f"top model confidence {top_prob:.2f} is below the threshold {self.min_confidence:.2f}"
            )
            warnings.append(
                f"Low confidence ({reason}); routed to '{FALLBACK_CATEGORY}' for human triage. "
                f"The model's best guess was '{top_label}'."
            )
        return Prediction(
            category=category,
            confidence=round(top_prob, 4),
            low_confidence=low,
            alternatives=[(label, round(prob, 4)) for label, prob in ranked[:3]],
            warnings=warnings,
        )

    def _known_term_ratio(self, text: str) -> float:
        vocabulary = self._pipeline.named_steps["tfidf"].vocabulary_
        tokens = _TOKEN.findall(text.lower())
        return sum(t in vocabulary for t in tokens) / len(tokens) if tokens else 0.0

    @property
    def holdout_macro_f1(self) -> float | None:
        return self.metrics.get("holdout", {}).get("macro_f1")
