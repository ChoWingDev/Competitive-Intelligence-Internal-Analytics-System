"""
ui/app.py
---------
Person B — Week 4
Streamlit dashboard for the PM Data Assistant.

Run: streamlit run ui/app.py
"""

import asyncio
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.router.router import RouterAgent
from src.memory.session import (
    add_turn,
    clear_session,
    get_recent_context,
    needs_clarification,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PM Data Assistant",
    page_icon="📊",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "router" not in st.session_state:
    st.session_state.router = RouterAgent()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("📁 Upload Industry Reports")
    uploaded_files = st.file_uploader(
        "Add PDF reports for RAG",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        reports_dir = Path("data/raw/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            dest = reports_dir / f.name
            dest.write_bytes(f.read())
        st.success(f"Uploaded {len(uploaded_files)} report(s). Rebuild RAG index to apply.")
        if st.button("🔄 Rebuild RAG Index"):
            st.session_state.router.rag_pipeline.build()
            st.success("RAG index rebuilt!")

    st.divider()
    st.subheader("💬 Session")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    if st.button("🗑️ New Chat"):
        clear_session(st.session_state.session_id)
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("📌 Example Questions")
    example_questions = [
        "What was the ROI for each campaign last month?",
        "What is our churn rate vs industry benchmark?",
        "Which traffic source has the best conversion rate?",
        "What are best practices for reducing churn in fashion ecommerce?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.prefill_question = q


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("📊 PM Data Assistant")
st.caption("Ask questions about your campaign performance — compared against industry benchmarks.")

# Render chat history
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_report(turn["report"])


def render_report(report_data):
    """Render a structured PM report into Streamlit components."""
    if isinstance(report_data, str):
        st.write(report_data)
        return

    # Summary
    if report_data.get("summary"):
        st.info(f"💡 **Summary:** {report_data['summary']}")

    # Comparison table
    if report_data.get("comparison_table"):
        st.subheader("📋 Performance vs Industry")
        df = pd.DataFrame(report_data["comparison_table"])

        def color_status(val):
            if "above" in val.lower():
                return "background-color: #ffd6d6"  # red tint = worse
            elif "below" in val.lower():
                return "background-color: #d6f5d6"  # green tint = better
            return ""

        if "status" in df.columns:
            st.dataframe(df.style.applymap(color_status, subset=["status"]), use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    # Action items
    if report_data.get("action_items"):
        st.subheader("🎯 Action Items")
        for i, item in enumerate(report_data["action_items"], 1):
            st.write(f"{i}. {item}")

    # Data sources
    if report_data.get("data_sources"):
        st.caption(f"Sources: {', '.join(report_data['data_sources'])}")


# Chat input
prefill = st.session_state.pop("prefill_question", "")
question = st.chat_input("Ask a PM question, e.g. 'What was our email campaign ROI last month?'") or prefill

if question:
    # Check for vague questions before sending to router
    clarification = needs_clarification(question)
    if clarification:
        with st.chat_message("assistant"):
            st.warning(clarification)
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Querying database and industry reports..."):
                try:
                    # Inject session memory into question context
                    context = get_recent_context(st.session_state.session_id)
                    full_question = f"{context}\n\nCurrent question: {question}" if context else question

                    # Run the router (async in sync Streamlit context)
                    report = asyncio.run(st.session_state.router.run(full_question))
                    report_dict = report.model_dump()

                    render_report(report_dict)

                    # Save to session memory and chat history
                    add_turn(st.session_state.session_id, question, report.summary)
                    st.session_state.chat_history.append({
                        "question": question,
                        "report": report_dict,
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.caption("Check your API key and database setup.")
