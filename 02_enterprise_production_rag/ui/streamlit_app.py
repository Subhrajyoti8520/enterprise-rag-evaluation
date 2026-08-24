"""
Streamlit Web User Interface for Enterprise SEC 10-K RAG

This script implements an interactive web dashboard using Streamlit to serve as the 
front-end client for the FastAPI RAG backend:
1. Dynamic UI & Session Tracking: Maintains conversation state across chat turns with 
   persistent session IDs generated via `uuid`.
2. Hyperparameter & Filter Controls: Provides sidebar sliders and inputs allowing users 
   to manually filter by stock ticker and adjust Stage 1 (Recall) and Stage 2 (Precision) depths.
3. Observability & Source Attribution: Expands each assistant response to show telemetry 
   metrics (latency, grounding status, rewritten query, citations, and ranked context passages with scores).
4. Session Flushing: Communicates with backend REST endpoints to clear conversational history 
   and reset UI states on demand.
"""
#===========================================================================================================
import os
import uuid
import requests
import streamlit as st

# -----------------Page Configuration--------------------
st.set_page_config(
    page_title="Enterprise SEC 10-K RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/chat")
SESSION_CLEAR_URL = os.getenv("SESSION_CLEAR_URL", "http://localhost:8000/api/v1/sessions")

# Initialize Session States
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------Sidebar Controls----------------------
with st.sidebar:
    st.title("⚙️ Engine Controls")
    st.markdown("Configure retrieval hyperparameters for the two-stage pipeline.")
    
    target_ticker = st.text_input("🏢 Filter Ticker (Optional)", placeholder="e.g. AAPL, NVDA, TSLA").strip().upper()
    top_k_retrieve = st.slider("Stage 1 Bi-Encoder Recall", min_value=5, max_value=30, value=15, step=1)
    top_k_return = st.slider("Stage 2 Cross-Encoder Precision", min_value=1, max_value=8, value=4, step=1)

    st.markdown("---")
    st.caption(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

    if st.button("🗑️ Reset Chat History", use_container_width=True):
        try:
            requests.delete(f"{SESSION_CLEAR_URL}/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
# ----------------Main Chat Area-----------------------
st.title("📊 Enterprise SEC 10-K Financial Assistant")
st.markdown("Query audited corporate SEC filings with **Two-Stage Neural Retrieval** and **Strict Context Grounding**.")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            with st.expander("🔍 Trace & Source Attribution"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{meta.get('latency', 0.0):.2f}s")
                c2.metric("Inferred Ticker", meta.get("detected_ticker") or "None")
                c3.markdown("**Status:** " + ("🟢 `GROUNDED`" if meta.get("is_grounded") else "🟡 `PARTIAL`"))
                
                st.markdown(f"**Rewritten Query:** `{meta.get('rewritten_query')}`")
                
                if meta.get("citations"):
                    st.markdown(f"**Citations:** {', '.join(meta['citations'])}")

                st.markdown("---")
                st.markdown("**Retrieved & Reranked Context Passages:**")
                for i, ctx in enumerate(meta.get("context", []), start=1):
                    st.markdown(f"**[{i}] [{ctx['ticker']}] Document ID: `{ctx['source_doc_id']}`** — *Score: `{ctx['rerank_score']:.4f}`*")
                    st.caption(ctx["content"][:300] + "...")

# Chat Input & Orchestration
if prompt := st.chat_input("E.g., What drove Apple's gross margin expansion in 2023?"):
    # Append & Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing filings with Two-Stage Retrieval..."):
            payload = {
                "session_id": st.session_state.session_id,
                "message": prompt,
                "target_ticker": target_ticker if target_ticker else None,
                "top_k_retrieve": top_k_retrieve,
                "top_k_return": top_k_return
            }

            try:
                res = requests.post(API_URL, json=payload, timeout=45)
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    
                    st.markdown(answer)
                    
                    meta_payload = {
                        "rewritten_query": data["rewritten_query"],
                        "detected_ticker": data.get("detected_ticker"),
                        "latency": data["latency_seconds"],
                        "is_grounded": data["is_grounded"],
                        "citations": data["citations"],
                        "context": data["context"]
                    }

                    with st.expander("🔍 Trace & Source Attribution"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Latency", f"{data['latency_seconds']:.2f}s")
                        c2.metric("Inferred Ticker", data.get("detected_ticker") or "None")
                        c3.markdown("**Status:** " + ("🟢 `GROUNDED`" if data["is_grounded"] else "🟡 `PARTIAL`"))
                        
                        st.markdown(f"**Rewritten Query:** `{data['rewritten_query']}`")
                        
                        if data["citations"]:
                            st.markdown(f"**Citations:** {', '.join(data['citations'])}")

                        st.markdown("---")
                        for i, ctx in enumerate(data["context"], start=1):
                            st.markdown(f"**[{i}] [{ctx['ticker']}] Document ID: `{ctx['source_doc_id']}`** — *Score: `{ctx['rerank_score']:.4f}`*")
                            st.caption(ctx["content"][:300] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": meta_payload
                    })
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to FastAPI backend at `{API_URL}`: {e}")