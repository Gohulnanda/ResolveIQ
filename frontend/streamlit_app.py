"""ResolveIQ dashboard. A thin client: all intelligence lives in the REST API.

Run:  streamlit run frontend/streamlit_app.py      (API_URL defaults to http://localhost:8000)
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60  # seconds; generous so an LLM-backed analysis can finish

SAMPLE_TICKETS = {
    "VPN will not connect (Network)": {
        "title": "Cannot connect to VPN from home",
        "description": "The VPN client times out after I enter my credentials. Home internet works fine for everything else.",
        "affected_system": "LAPTOP-0142", "urgency": "high", "impact": "individual",
    },
    "Production database unreachable (Database)": {
        "title": "Production database connection refused",
        "description": "Our order service gets connection refused from the SQL server on port 1433. All users are affected and checkout is down.",
        "affected_system": "sql-prod-01", "urgency": "critical", "impact": "organization",
    },
    "Locked out after phone change (Account)": {
        "title": "Authenticator codes rejected on new phone",
        "description": "I changed phones and the six digit authenticator codes are no longer accepted, so I cannot sign in.",
        "affected_system": "", "urgency": "medium", "impact": "individual",
    },
    "Cafeteria question (no knowledge available)": {
        "title": "Cafeteria menu question",
        "description": "What is on the cafeteria lunch menu this Thursday, and is it open late?",
        "affected_system": "", "urgency": "low", "impact": "individual",
    },
}
NONE = "(not specified)"
URGENCIES = [NONE, "low", "medium", "high", "critical"]
IMPACTS = [NONE, "individual", "team", "department", "organization"]


# ----------------------------------------------------------------------------- API helpers
class ApiError(Exception):
    pass


def call_api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Is it running?") from exc
    except requests.Timeout as exc:
        raise ApiError("The API did not respond in time.") from exc
    if response.status_code >= 400:
        try:
            error = response.json()["error"]
            lines = [error["message"]] + [
                f"- **{d.get('field') or 'request'}**: {d['message']}" for d in error.get("details") or []
            ]
        except (ValueError, KeyError):
            lines = [f"The API returned HTTP {response.status_code}."]
        raise ApiError("\n".join(lines))
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_status() -> dict:
    try:
        return {"ok": True, "data": call_api("GET", "/ready")}
    except ApiError as exc:
        return {"ok": False, "error": str(exc)}


# ----------------------------------------------------------------------------- UI pieces
def mode_badge(mode: str) -> str:
    return {
        "llm": ":green-badge[LLM-generated]",
        "demo": ":orange-badge[Demo mode: deterministic template, no LLM]",
        "demo_fallback": ":red-badge[LLM unavailable: deterministic fallback used]",
    }.get(mode, mode)


def sidebar_status() -> None:
    status = fetch_status()
    st.sidebar.subheader("System status")
    if not status["ok"]:
        st.sidebar.error(status["error"])
        return
    checks = status["data"]["checks"]
    kb, llm = checks["knowledge_base"], checks["llm"]
    st.sidebar.success("API ready")
    if llm["mode"] == "llm":
        st.sidebar.markdown(f"**Mode:** :green-badge[LLM: {llm['provider']}]")
    else:
        st.sidebar.markdown("**Mode:** :orange-badge[Demo (no LLM)]")
    st.sidebar.caption(f"Retrieval: `{kb['retrieval_method']}` over {kb['articles']} articles")
    if not kb["neural_embeddings_in_use"]:
        st.sidebar.caption("Not neural embeddings: lexical + LSA similarity.")
    if kb.get("embedding_fallback_reason"):
        st.sidebar.caption(f"Neural backend unavailable: {kb['embedding_fallback_reason']}.")


def render_analysis(result: dict) -> None:
    resolution = result["resolution"]
    st.subheader(f"Result for {result['ticket_id']}")
    st.markdown(mode_badge(resolution["mode"]))
    if resolution.get("fallback_reason"):
        st.warning(f"Provider issue (`{resolution['fallback_reason']}`). The steps below come from the retrieved articles, not an LLM.")

    left, mid, right = st.columns(3)
    left.metric("Predicted category", result["category"])
    mid.metric("Predicted priority", f"{result['predicted_priority']['level']} {result['predicted_priority']['label']}")
    right.metric("Classifier confidence", f"{result['confidence']:.0%}")
    st.progress(min(max(result["confidence"], 0.0), 1.0))

    for warning in result["warnings"]:
        st.warning(warning)
    if result["low_confidence"]:
        st.info("Low confidence: treat the category as a suggestion and route to a human for triage.")

    with st.expander("Why this priority? (rule-based heuristic)"):
        for line in result["predicted_priority"]["reasoning"]:
            st.write(f"- {line}")
        user = result["user_provided"]
        st.caption(
            f"Requester stated urgency: {user['urgency'] or 'none'}, impact: {user['impact'] or 'none'}. "
            "These are inputs to the heuristic, kept separate from the prediction."
        )
    with st.expander("Other category candidates"):
        st.dataframe(
            pd.DataFrame(result["alternatives"]).rename(columns={"category": "Category", "probability": "Probability"}),
            hide_index=True, width="stretch",
        )

    st.markdown("### Suggested resolution")
    st.write(resolution["summary"])
    if resolution["steps"]:
        for number, step in enumerate(resolution["steps"], 1):
            source = f" `[{step['source_id']}]`" if step["source_id"] else ""
            st.markdown(f"{number}. {step['text']}{source}")
    if resolution.get("escalation"):
        st.markdown(f"**Escalate when:** {resolution['escalation']}")
    if resolution["citations"]:
        st.markdown("**Sources cited:** " + ", ".join(f"`{c}`" for c in resolution["citations"]))
    st.caption(resolution["generated_by"])

    st.markdown("### Retrieved knowledge-base articles")
    if not result["kb_match_found"]:
        st.info("No knowledge-base article was relevant enough to this ticket.")
    for article in result["retrieved"]:
        with st.expander(f"{article['article_id']} · {article['title']} · score {article['score']:.2f}"):
            st.caption(f"Category: {article['category']} · matched section: {article['matched_section']}")
            st.write(article["matched_passage"])
    st.caption(f"Retrieval method: {result['retrieval_method']} · processed in {result['processing_time_ms']} ms")


def page_analyze() -> None:
    st.title("Analyze a support ticket")
    st.write("Submit a ticket to get a predicted category and priority, relevant knowledge-base articles, and a grounded resolution.")

    choice = st.selectbox("Load a sample ticket", [NONE, *SAMPLE_TICKETS], key="sample_choice")
    sample = SAMPLE_TICKETS.get(choice, {})
    suffix = choice  # changing the sample resets the form fields

    with st.form("ticket_form"):
        title = st.text_input("Title *", value=sample.get("title", ""), max_chars=200, key=f"t{suffix}")
        description = st.text_area(
            "Description *", value=sample.get("description", ""), max_chars=5000, height=140, key=f"d{suffix}"
        )
        c1, c2, c3, c4 = st.columns(4)
        ticket_id = c1.text_input("Ticket ID", placeholder="auto-generated", key=f"i{suffix}")
        system = c2.text_input("Affected system", value=sample.get("affected_system", ""), key=f"s{suffix}")
        urgency = c3.selectbox("Urgency", URGENCIES, index=URGENCIES.index(sample.get("urgency", NONE)), key=f"u{suffix}")
        impact = c4.selectbox("Impact", IMPACTS, index=IMPACTS.index(sample.get("impact", NONE)), key=f"m{suffix}")
        submitted = st.form_submit_button("Analyze ticket", type="primary")

    if submitted:
        payload = {"title": title, "description": description}
        if ticket_id.strip():
            payload["ticket_id"] = ticket_id.strip()
        if system.strip():
            payload["affected_system"] = system.strip()
        if urgency != NONE:
            payload["urgency"] = urgency
        if impact != NONE:
            payload["impact"] = impact
        try:
            with st.spinner("Classifying, retrieving knowledge and drafting a resolution..."):
                st.session_state["result"] = call_api("POST", "/api/v1/tickets/analyze", json=payload)
        except ApiError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

    if "result" in st.session_state:
        render_analysis(st.session_state["result"])


def page_knowledge_base() -> None:
    st.title("Knowledge base")
    query = st.text_input("Search articles", placeholder="e.g. printer offline, password reset, slow queries")
    try:
        if query.strip():
            with st.spinner("Searching..."):
                found = call_api("POST", "/api/v1/knowledge-base/search", json={"query": query.strip(), "top_k": 5})
            st.caption(f"Method: {found['retrieval_method']} · minimum score {found['min_score']}")
            if not found["results"]:
                st.info(found.get("message") or "No results.")
            for hit in found["results"]:
                with st.expander(f"{hit['article_id']} · {hit['title']} · score {hit['score']:.2f}", expanded=True):
                    st.caption(f"Matched section: {hit['matched_section']}")
                    for number, step in enumerate(hit["resolution_steps"], 1):
                        st.markdown(f"{number}. {step}")
        else:
            articles = call_api("GET", "/api/v1/knowledge-base")
            st.caption(f"{len(articles)} articles. Enter a query above to search them.")
            for article in articles:
                with st.expander(f"{article['article_id']} · {article['title']}"):
                    st.caption(f"{article['category']} · tags: {', '.join(article['tags'])}")
                    st.write(article["summary"])
                    if st.button("Show full article", key=f"open-{article['article_id']}"):
                        detail = call_api("GET", f"/api/v1/knowledge-base/{article['article_id']}")
                        for name, body in detail["sections"].items():
                            st.markdown(f"**{name}**\n\n{body}")
    except ApiError as exc:
        st.error(str(exc))


def page_about() -> None:
    st.title("About ResolveIQ")
    st.write(
        "ResolveIQ classifies IT support tickets, retrieves relevant troubleshooting articles and "
        "drafts a resolution grounded in them. It is a portfolio project, not a production system."
    )
    try:
        info = call_api("GET", "/api/v1/model-info")
    except ApiError as exc:
        st.error(str(exc))
        return
    holdout = info["holdout"]
    st.subheader("Classifier evaluation")
    st.warning(info["note"])
    a, b, c = st.columns(3)
    a.metric("Hold-out accuracy", f"{holdout['accuracy']:.3f}")
    b.metric("Hold-out macro F1", f"{holdout['macro_f1']:.3f}")
    c.metric("Hold-out size", holdout["n_test"])
    st.caption(
        "Hold-out scores on synthetic data are optimistic. See docs/evaluation_results.md in the "
        "repository for the hand-written challenge set and retrieval results."
    )
    labels = holdout["labels"]
    st.markdown("**Hold-out confusion matrix** (rows: actual, columns: predicted)")
    st.dataframe(pd.DataFrame(holdout["confusion_matrix"], index=labels, columns=labels), width="stretch")
    st.markdown("**Per-class precision / recall / F1**")
    per_class = pd.DataFrame(holdout["per_class"]).T
    st.dataframe(per_class, width="stretch")
    st.caption(f"Trained {info['trained_at']} on {info['n_training_rows']} rows (seed {info['random_seed']}).")


def main() -> None:
    st.set_page_config(page_title="ResolveIQ", page_icon="🛠️", layout="wide")
    st.sidebar.title("ResolveIQ")
    st.sidebar.caption("AI-powered IT support intelligence")
    page = st.sidebar.radio("Navigate", ["Analyze ticket", "Knowledge base", "About & evaluation"])
    sidebar_status()
    {"Analyze ticket": page_analyze, "Knowledge base": page_knowledge_base, "About & evaluation": page_about}[page]()


if __name__ == "__main__":
    main()
