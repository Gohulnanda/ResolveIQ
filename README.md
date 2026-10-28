# ResolveIQ: AI-Powered IT Support Intelligence Platform

ResolveIQ takes an IT support ticket and returns a predicted category, a predicted priority with its
reasoning, the most relevant troubleshooting articles from a local knowledge base, and a suggested
resolution that is grounded in (and cites) those articles. It has a FastAPI backend, a Streamlit
dashboard, a scikit-learn classifier, a retrieval layer, an optional LLM provider, and a test suite.

It is a portfolio project. It runs fully offline in **demo mode** with no API key.

> **Honesty notes (please read)**
> - The classifier is trained on **synthetic, template-generated tickets** included in this repo, not real enterprise data.
> - Retrieval uses an **offline TF-IDF + LSA hybrid. It is not neural embeddings** and is labelled as such everywhere (API, UI). A neural backend is optional.
> - In **demo mode the resolution is a deterministic template** that copies steps from retrieved articles. It is labelled "demo" and is never presented as LLM output.
> - The real-LLM code path is covered by mocked-transport unit tests only. It has **not been run against a live provider** in this build.
> - Priority is a **rule-based heuristic**, not a learned model.

## Quick start

Requires Python 3.13 (the pinned versions were tested on 3.13).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Terminal 1: API (trains and saves the classifier on first start if needed)
uvicorn app.main:create_app --factory --port 8000

# Terminal 2: dashboard
streamlit run frontend/streamlit_app.py
```

Open the dashboard at http://localhost:8501 and the API docs at http://localhost:8000/docs.

**Quick demo:** on the "Analyze ticket" page pick a sample ticket from the dropdown, press *Analyze ticket*,
then try the "Cafeteria question" sample to see the system report that no knowledge is available.

### Docker

```bash
docker compose up --build     # API on :8000, UI on :8501
```

The compose file and Dockerfile were syntax-checked, but **the image was not built in the authoring
environment (no Docker daemon was available)**. The same install, training and startup steps were run
manually in a clean virtual environment.

## Using a real LLM (optional)

```bash
cp .env.example .env     # then edit .env
```

```
LLM_PROVIDER=anthropic          # or: openai (any OpenAI-compatible endpoint; set LLM_BASE_URL)
LLM_API_KEY=your-key-here
LLM_MODEL=your-model-id
```

- The key is read only from the environment or `.env` (git-ignored) and is never logged or returned by the API.
- The LLM sees only the retrieved articles and the ticket. It must return JSON in which every step cites a
  retrieved article ID; output that fails validation or cites anything else is rejected.
- On timeout, rate limit (429), auth error, provider error, network error or malformed output, the API
  returns a deterministic fallback with `mode: "demo_fallback"` and a `fallback_reason`.
- Response `mode` is one of `demo`, `llm`, `demo_fallback`. The UI shows a badge for each.

## Optional neural embeddings

```bash
pip install -r requirements-embeddings.txt     # large: pulls in PyTorch
# .env: EMBEDDING_BACKEND=sentence-transformers   (or leave "auto")
```

If the package or model is unavailable the app falls back to the offline hybrid and reports the reason in
`/ready` (`embedding_fallback_reason`). Neural cosine scores sit on a different scale, so retune
`RETRIEVAL_MIN_SCORE` if you switch. This path could not be exercised in the authoring environment (the model
host was unreachable), so it is untested beyond the fallback behaviour.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness: classifier, knowledge base, retrieval method, LLM mode (503 if startup failed) |
| GET | `/api/v1/knowledge-base` | List articles |
| GET | `/api/v1/knowledge-base/{id}` | One full article |
| POST | `/api/v1/knowledge-base/search` | Search articles |
| POST | `/api/v1/tickets/classify` | Category and priority only |
| POST | `/api/v1/tickets/analyze` | Classification + retrieval + grounded resolution |
| GET | `/api/v1/model-info` | Training metadata and hold-out metrics |
| GET | `/docs` | Swagger UI |

```bash
curl -s -X POST http://localhost:8000/api/v1/tickets/analyze \
  -H 'content-type: application/json' \
  -d '{
    "title": "Cannot connect to VPN from home",
    "description": "The VPN client times out after I enter my credentials. Home internet works fine.",
    "affected_system": "LAPTOP-0142",
    "urgency": "high",
    "impact": "individual"
  }'
```

Actual output from the command above (abridged: only the first of 5 steps is shown, and the first
retrieved article's passage and step list are omitted; latency varies by machine):

```json
{
  "ticket_id": "DEMO-001",
  "category": "Network",
  "confidence": 0.7792,
  "low_confidence": false,
  "predicted_priority": {
    "level": "P3",
    "label": "Medium",
    "score": 2,
    "reasoning": [
      "Requester-stated urgency 'high' (+2)",
      "Requester-stated impact 'individual' (+0)"
    ],
    "method": "rule-based heuristic (not learned from data)"
  },
  "user_provided": {
    "urgency": "high",
    "impact": "individual",
    "affected_system": "LAPTOP-0142"
  },
  "retrieved": [
    {
      "article_id": "KB-001",
      "title": "VPN connection fails or keeps disconnecting",
      "score": 0.5461,
      "matched_section": "Resolution steps"
    }
  ],
  "kb_match_found": true,
  "retrieval_method": "tfidf-lsa-hybrid",
  "resolution": {
    "mode": "demo",
    "summary": "Demo response (deterministic template, not LLM-generated). Steps are copied from the retrieved article(s): KB-001 VPN connection fails or keeps disconnecting.",
    "steps": [
      {
        "text": "Confirm the home internet works by opening a public website before connecting to the VPN.",
        "source_id": "KB-001"
      },
      "... (5 steps)"
    ],
    "citations": [
      "KB-001"
    ],
    "generated_by": "Deterministic template over retrieved articles; no LLM was used."
  },
  "processing_time_ms": 11.9
}
```

Errors use one envelope: `{"error": {"code": "...", "message": "...", "details": [{"field": "...", "message": "..."}]}}`
with 422 for validation, 413 for bodies over `MAX_REQUEST_BYTES`, 411 when `Content-Length` is missing,
503 when the service is not ready, and a generic 500 that never exposes stack traces or configuration.
Submitted values are not echoed back in error responses.

## How it works

```
ticket -> validate -> classifier (TF-IDF + logistic regression) -> category, confidence
                   -> priority heuristic (text signals + stated urgency/impact)
                   -> retrieval (section-level index, best passage per article)
                   -> resolution (demo template | LLM, validated against retrieved IDs)
```

See [docs/architecture.md](docs/architecture.md) for the diagram and design decisions.

**Classification.** TF-IDF (1-2 grams) with multinomial logistic regression, fixed seed. Predictions below
`CLASSIFIER_MIN_CONFIDENCE`, or text where under 30% of the words are known to the model, are routed to
"Other" and flagged `low_confidence`, with the model's best guess in a warning. Very short tickets get a warning.

**Priority.** Points from text signals (outage, wide scope, business-critical system, degraded service,
workaround available) plus the requester's urgency and impact; each signal counts once. Totals map to
P1 (>=7), P2 (>=4), P3 (>=2), P4. Stated urgency alone cannot produce P1. The requester's values are returned
separately in `user_provided`; they are inputs to the heuristic, not replaced by it.

**Retrieval.** Each article section is a passage; articles are ranked by their best passage using exact
cosine search over normalised vectors (the corpus is tiny, so no ANN index is warranted). The offline
backend also requires at least two shared content words, because a single rare shared word otherwise
yields a deceptively high cosine score (a bug found during testing: nonsense text containing the word "sit"
matched a printer article).

## Evaluation

Reproduce everything with:

```bash
python scripts/generate_sample_data.py   # regenerates data/sample_tickets.csv (seeded)
python scripts/train_model.py            # trains, prints hold-out metrics, saves models/classifier.joblib
python scripts/evaluate_model.py         # writes docs/evaluation_results.md
```

| Check | Result |
|---|---|
| Synthetic data, 5-fold CV (385 tickets) | accuracy 1.000, macro F1 1.000 (**inflated**: shared phrase pool) |
| Hand-written challenge set (35 tickets, never trained on) | accuracy 0.886, macro F1 0.883 |
| Challenge tickets flagged low-confidence | 11 of 35, covering all 4 misclassified tickets |
| Retrieval (29 queries) | Hit@1 29/29, Hit@3 29/29, MRR 1.000 |
| Correct "no answer" abstention | 2 of 3 (misses "holiday schedule for the service desk") |

Full per-class precision/recall/F1 and confusion matrices: [docs/evaluation_results.md](docs/evaluation_results.md).

**Why these numbers do not prove production performance.** The training data is synthetic and drawn from a
small phrase pool; the challenge set has 5 tickets per class (one ticket moves a class's recall by 20
points); the retrieval queries were written by the same person who wrote the 15-article knowledge base; and
both confidence thresholds (0.30 classifier, 0.35 retrieval) were chosen after looking at these same small
sets, so the figures above are optimistic. Real tickets are longer, messier and distributed differently.
Re-evaluate on real labelled tickets before trusting any of this.

## Testing

```bash
python -m pytest
```

86 tests, none needing network access or an API key: unit tests (priority, schemas, classifier, retrieval,
LLM provider with mocked HTTP), integration tests (full pipeline, LLM failure fallback) and API tests
(contracts, validation, 413/411/404/405, malformed JSON, unicode and injection-style input, safe 500s,
startup failure reporting, secret non-disclosure).

## Security and reliability

- Config via environment/`.env` only; `.env.example` holds placeholders; `.env` and model artifacts are git-ignored.
- Input limits: title 200 chars, description 5,000, request body 32 KiB (configurable); unknown fields rejected.
- Ticket text is passed to an LLM inside a delimited block with an instruction to treat it as untrusted data;
  this reduces but does not eliminate prompt-injection risk. Citation validation limits what a manipulated model can claim.
- Logs record ticket ID, category, priority, hit count, mode and latency, never ticket text. Nothing is persisted per request.
- LLM calls have a timeout; every failure path degrades to the labelled fallback.
- There is no authentication or rate limiting; put the API behind a gateway before exposing it.

## Limitations

- Sample data only; 7 broad categories; English only.
- 15 knowledge-base articles written for this project; coverage is narrow.
- The offline retriever is lexical + LSA, so it can miss paraphrases with no shared vocabulary.
- The model file is rebuilt automatically if missing or built with another scikit-learn version.
- No screenshots are included; none could be captured in the authoring environment.

## Repository layout

```
app/            FastAPI app (api/, core/config, models/schemas, services/, retrieval/)
frontend/       Streamlit dashboard
data/           sample_tickets.csv (synthetic), challenge_tickets.csv, retrieval_eval.json, knowledge_base/*.md
scripts/        generate_sample_data.py, train_model.py, evaluate_model.py
tests/          unit/, integration/, test_api.py
docs/           architecture.md, evaluation_results.md
```

## License

MIT. See [LICENSE](LICENSE).
