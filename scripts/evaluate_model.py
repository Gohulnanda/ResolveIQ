"""Reproducible evaluation of classification and retrieval.

Reports:
  1. Classification, 5-fold stratified cross-validation on the synthetic training set.
  2. Classification on data/challenge_tickets.csv (hand-written, never used for training).
  3. Confidence behaviour: how often predictions fall below the low-confidence threshold.
  4. Retrieval: hit@1, hit@3, MRR over data/retrieval_eval.json, plus abstention on
     queries that have no relevant article.

Writes docs/evaluation_results.md. Usage:  python scripts/evaluate_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import RANDOM_SEED, Settings  # noqa: E402
from app.retrieval.embeddings import build_embedder  # noqa: E402
from app.retrieval.knowledge_base import KnowledgeBase, load_articles  # noqa: E402
from app.services.classifier import (  # noqa: E402
    DATA_ORIGIN_NOTE,
    TicketClassifier,
    build_pipeline,
    compose_text,
    evaluate_predictions,
    load_dataset,
)

CHALLENGE_PATH = ROOT / "data" / "challenge_tickets.csv"
RETRIEVAL_PATH = ROOT / "data" / "retrieval_eval.json"
REPORT_PATH = ROOT / "docs" / "evaluation_results.md"


def confusion_table(result: dict) -> str:
    labels = result["labels"]
    short = [label[:9] for label in labels]
    lines = ["| actual \\ predicted | " + " | ".join(short) + " |", "|---|" + "---|" * len(labels)]
    for label, row in zip(labels, result["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def per_class_table(result: dict) -> str:
    lines = ["| Category | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for label, m in result["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    return "\n".join(lines)


def classification_section(settings: Settings) -> tuple[str, dict]:
    train_df = load_dataset(settings.training_data_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_pred = cross_val_predict(build_pipeline(), train_df["text"], train_df["category"], cv=cv)
    cv_result = evaluate_predictions(list(train_df["category"]), list(cv_pred))

    classifier = TicketClassifier.train(settings.training_data_path, settings.classifier_min_confidence)
    challenge = load_dataset(CHALLENGE_PATH)
    predictions = [classifier.predict(t, d) for t, d in zip(challenge["title"], challenge["description"])]
    raw_top = [p.alternatives[0][0] for p in predictions]  # model's own top label, before thresholding
    ch_result = evaluate_predictions(list(challenge["category"]), raw_top)
    low = sum(p.low_confidence for p in predictions)

    # Confidence behaviour on nonsense input
    nonsense = ["asdf qwer zxcv", "lorem ipsum dolor sit amet", "12345 67890 13579"]
    nonsense_preds = [classifier.predict(t, "Nothing meaningful here at all") for t in nonsense]

    confidences = np.array([p.confidence for p in predictions])
    correct = np.array([a == b for a, b in zip(raw_top, challenge["category"])])
    text = f"""## 1. Classification

{DATA_ORIGIN_NOTE}

### 1a. 5-fold stratified cross-validation on the synthetic training set ({cv_result['n_samples']} tickets)

Accuracy **{cv_result['accuracy']:.3f}**, macro precision {cv_result['macro_precision']:.3f}, macro recall {cv_result['macro_recall']:.3f}, macro F1 **{cv_result['macro_f1']:.3f}**.

{per_class_table(cv_result)}

Confusion matrix (rows = actual, columns = predicted):

{confusion_table(cv_result)}

These numbers are high because the synthetic tickets are assembled from a small shared phrase pool, so a
held-out ticket often contains the same phrases as training tickets. Treat them as a sanity check of the
pipeline, not as evidence of real-world accuracy.

### 1b. Hand-written challenge set ({ch_result['n_samples']} tickets, never used for training)

Accuracy **{ch_result['accuracy']:.3f}**, macro precision {ch_result['macro_precision']:.3f}, macro recall {ch_result['macro_recall']:.3f}, macro F1 **{ch_result['macro_f1']:.3f}** (model's top label, before applying the confidence threshold).

{per_class_table(ch_result)}

Confusion matrix (rows = actual, columns = predicted):

{confusion_table(ch_result)}

With only 5 tickets per class these figures have wide uncertainty (one ticket moves a class's recall by 20 points).

### 1c. Confidence behaviour

- Low-confidence threshold: {settings.classifier_min_confidence:.2f} (predictions below it are routed to "Other" for human triage).
- Challenge set: {low} of {len(predictions)} tickets fell below the threshold. Mean top-class probability {confidences.mean():.2f} overall, {confidences[correct].mean() if correct.any() else float('nan'):.2f} on correctly classified tickets, {confidences[~correct].mean() if (~correct).any() else float('nan'):.2f} on misclassified ones.
- Nonsense inputs ({len(nonsense)}): {sum(p.low_confidence for p in nonsense_preds)} of {len(nonsense_preds)} flagged low-confidence (top probabilities: {', '.join(f'{p.confidence:.2f}' for p in nonsense_preds)}).
"""
    return text, {"cv": cv_result, "challenge": ch_result, "low": low}


def retrieval_section(settings: Settings) -> tuple[str, dict]:
    spec = json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))
    embedder, reason = build_embedder("hybrid", settings.embedding_model)
    kb = KnowledgeBase(load_articles(settings.kb_dir), embedder, reason)

    relevant = [q for q in spec["queries"] if q["expected"]]
    negative = [q for q in spec["queries"] if not q["expected"]]
    hit1 = hit3 = 0
    reciprocal: list[float] = []
    misses: list[str] = []
    latencies: list[float] = []
    for q in relevant:
        start = time.perf_counter()
        hits = kb.search(q["query"], top_k=3, min_score=settings.retrieval_min_score)
        latencies.append((time.perf_counter() - start) * 1000)
        ids = [h.article.article_id for h in hits]
        rank = next((i for i, a in enumerate(ids, 1) if a in q["expected"]), None)
        hit1 += rank == 1
        hit3 += rank is not None
        reciprocal.append(1.0 / rank if rank else 0.0)
        if rank != 1:
            misses.append(f"- `{q['query']}` expected {q['expected']}, got {ids}")

    abstained = 0
    false_pos: list[str] = []
    for q in negative:
        hits = kb.search(q["query"], top_k=3, min_score=settings.retrieval_min_score)
        if not hits:
            abstained += 1
        else:
            false_pos.append(f"- `{q['query']}` returned {[h.article.article_id for h in hits]} (top score {hits[0].score})")

    n = len(relevant)
    text = f"""## 2. Retrieval

Method: **{kb.method}** ({kb.embedder.description}). Minimum score {settings.retrieval_min_score:.2f}.
{n} queries with at least one relevant article, plus {len(negative)} queries with no relevant article.

| Metric | Value |
|---|---|
| Hit@1 | {hit1}/{n} = {hit1 / n:.3f} |
| Hit@3 | {hit3}/{n} = {hit3 / n:.3f} |
| MRR | {np.mean(reciprocal):.3f} |
| Correct abstention (no-answer queries) | {abstained}/{len(negative)} |
| Median search latency | {np.median(latencies):.1f} ms |

Queries where the first result was not an expected article:
{chr(10).join(misses) if misses else '- none'}

No-answer queries that still returned results:
{chr(10).join(false_pos) if false_pos else '- none'}

The queries were written by the same author as the knowledge base, so these results are optimistic.
"""
    return text, {"hit1": hit1 / n, "hit3": hit3 / n, "mrr": float(np.mean(reciprocal)), "abstained": abstained, "negatives": len(negative)}


def main() -> None:
    settings = Settings()
    cls_text, cls_data = classification_section(settings)
    ret_text, ret_data = retrieval_section(settings)
    report = f"""# Evaluation results

Generated by `python scripts/evaluate_model.py` (seed {RANDOM_SEED}).

**Limitations.** The training data is synthetic and template-based, the knowledge base has only 15
articles, and the evaluation sets are small and written by the project author. Offline benchmark numbers on
data like this do not demonstrate production performance; real tickets are messier, longer, multilingual
and drawn from a different distribution. Re-evaluate on real, labelled tickets before relying on any figure here.

{cls_text}
{ret_text}"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
