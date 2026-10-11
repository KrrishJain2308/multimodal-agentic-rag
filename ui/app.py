import sys
import os
import time
import json
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import get_agent
from src.ingest import ingest_document, clear_database, initialize_vector_store
from src.config import MAX_FILE_SIZE_MB

# ─── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* --- Global --- */
    .stApp {
        background-color: #0e1117;
    }

    /* --- Header Banner --- */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .header-banner h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }

    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13151a 0%, #1a1d24 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #a78bfa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 1rem;
    }

    /* --- Status Badge --- */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-online {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* --- Metric Card --- */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* --- Welcome Card --- */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 1rem;
        text-align: center;
    }
    .welcome-container .emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .welcome-container h2 {
        color: white;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .welcome-container p {
        color: rgba(255,255,255,0.55);
        font-size: 0.95rem;
        max-width: 500px;
        line-height: 1.6;
    }

    /* --- Feature Cards --- */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 2rem;
        max-width: 700px;
    }
    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .feature-card:hover {
        border-color: rgba(167, 139, 250, 0.4);
    }
    .feature-card .icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-card .title {
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .feature-card .desc {
        color: rgba(255,255,255,0.45);
        font-size: 0.73rem;
        margin-top: 0.3rem;
    }

    /* --- Source Expander --- */
    .source-chip {
        display: inline-block;
        background: rgba(167, 139, 250, 0.12);
        color: #c4b5fd;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
        border: 1px solid rgba(167, 139, 250, 0.2);
    }

    /* --- Chat message tweaks --- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* --- Hide Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🧠 Multimodal Agentic RAG</h1>
    <p>Upload documents → Build knowledge base → Chat with your AI agent</p>
</div>
""", unsafe_allow_html=True)

# ─── HELPER: Get KB Stats ─────────────────────────────────────────
def get_kb_stats():
    """Get knowledge base statistics from ChromaDB."""
    try:
        vs = initialize_vector_store()
        collection = vs._collection
        count = collection.count()
        return {"chunks": count, "status": "online"}
    except Exception:
        return {"chunks": 0, "status": "offline"}


# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    # --- Status ---
    st.markdown("## System Status")
    kb_stats = get_kb_stats()

    if kb_stats["status"] == "online":
        st.markdown('<span class="status-badge status-online">● Database Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">● Database Offline</span>', unsafe_allow_html=True)

    # --- Knowledge Base Metrics ---
    st.markdown("## Knowledge Base")
    doc_estimate = max(1, kb_stats['chunks'] // 15) if kb_stats['chunks'] > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{doc_estimate}</div>
        <div class="label">Documents Ingested</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # --- Document Upload ---
    st.markdown("## Ingest Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "csv", "docx", "pptx", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"📎 {len(uploaded_files)} file(s) selected")

    if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one file first.")
        else:
            progress_bar = st.progress(0, text="Starting ingestion...")
            success_count = 0
            fail_count = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                # --- File Size Validation ---
                file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.warning(f"⚠️ **{uploaded_file.name}** is too large ({file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit). Skipped.")
                    fail_count += 1
                    continue

                temp_path = os.path.join("data", uploaded_file.name)
                os.makedirs("data", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                progress_text = f"Processing {uploaded_file.name} ({idx+1}/{len(uploaded_files)})..."
                progress_bar.progress((idx) / len(uploaded_files), text=progress_text)

                try:
                    ingest_document(temp_path)
                    st.success(f"✅ {uploaded_file.name}")
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Failed to ingest {uploaded_file.name}: {str(e)}")
                    fail_count += 1

            progress_bar.progress(1.0, text="Done!")
            time.sleep(0.5)
            progress_bar.empty()

            if success_count > 0:
                st.toast(f"🎉 Successfully ingested {success_count} document(s)!", icon="✅")
            if fail_count > 0:
                st.toast(f"⚠️ {fail_count} document(s) failed.", icon="⚠️")

            st.rerun()

    st.markdown("---")

    # --- Database Management ---
    st.markdown("## Manage")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🚨 Reset KB", use_container_width=True):
            clear_database()
            st.toast("Knowledge base cleared!", icon="🗑️")
            st.rerun()

    st.markdown("---")

    # --- Export Chat ---
    st.markdown("## Export")
    if st.session_state.get("messages"):
        chat_export = json.dumps(st.session_state.messages, indent=2, default=str)
        st.download_button(
            label="📥 Download Chat History",
            data=chat_export,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("No chat history to export yet.")


# ─── INITIALIZATION ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_agent():
    return get_agent()

if "agent" not in st.session_state:
    try:
        with st.spinner("Initializing AI agent..."):
            st.session_state.agent = load_agent()
    except Exception as e:
        st.session_state.agent = None
        st.error(f"⚠️ Failed to initialize Agent: `{str(e)}`")
        st.info("💡 **Next steps:** Make sure you have a valid `GOOGLE_API_KEY` in your `.env` file and restart the app.")


# ─── WELCOME SCREEN ───────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <div class="emoji">🚀</div>
        <h2>Welcome to Multimodal Agentic RAG</h2>
        <p>Upload your documents using the sidebar, then ask questions about them.
        The AI agent will search your knowledge base and provide cited answers.</p>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">📄</div>
                <div class="title">Multi-Format</div>
                <div class="desc">PDF, DOCX, PPTX, XLSX, CSV</div>
            </div>
            <div class="feature-card">
                <div class="icon">🔍</div>
                <div class="title">Smart Search</div>
                <div class="desc">Semantic vector retrieval</div>
            </div>
            <div class="feature-card">
                <div class="icon">🤖</div>
                <div class="title">AI Agent</div>
                <div class="desc">Intelligent reasoning & answers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── CHAT DISPLAY ─────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if available
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    source_name = src.get("source", "Unknown")
                    page = src.get("page", "")
                    page_info = f" — Page {page}" if page else ""
                    st.markdown(f'<span class="source-chip">📄 {source_name}{page_info}</span>', unsafe_allow_html=True)
                    preview = src.get("preview", "")
                    if preview:
                        st.caption(preview[:200] + "..." if len(preview) > 200 else preview)


# ─── CHAT INPUT ────────────────────────────────────────────────────
if prompt := st.chat_input("💬 Ask a question about your documents..."):
    if st.session_state.agent is None:
        st.error("The agent is not initialized. Please check your API key and refresh the page.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()

            with st.spinner("🔍 Searching knowledge base & reasoning..."):
                try:
                    # Format history for LangGraph
                    formatted_messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ]

                    # Invoke the agent
                    response = st.session_state.agent.invoke({"messages": formatted_messages})

                    # Extract the final answer
                    raw_content = response["messages"][-1].content
                    if isinstance(raw_content, list):
                        full_response = "".join(
                            part.get("text", "")
                            for part in raw_content
                            if isinstance(part, dict) and "text" in part
                        )
                    else:
                        full_response = str(raw_content)

                    # Extract source documents from intermediate messages
                    sources = []
                    for msg in response["messages"]:
                        if hasattr(msg, "type") and msg.type == "tool":
                            # Tool messages contain retrieved documents
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if content and len(content) > 20:
                                sources.append({
                                    "source": getattr(msg, "name", "document_search"),
                                    "preview": content[:300],
                                })

                    # Simulated streaming for visual polish
                    displayed_response = ""
                    words = full_response.split(" ")
                    for i, chunk in enumerate(words):
                        displayed_response += chunk + " "
                        if i % 3 == 0:  # Update every 3 words for smoother streaming
                            message_placeholder.markdown(displayed_response + "▌")
                            time.sleep(0.02)

                    message_placeholder.markdown(full_response)

                    # Show sources
                    if sources:
                        with sources_placeholder.expander("📚 Sources", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f'<span class="source-chip">📄 {src["source"]}</span>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    src["preview"][:200] + "..."
                                    if len(src["preview"]) > 200
                                    else src["preview"]
                                )

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"❌ Error: `{str(e)}`")
                    st.info("💡 Check your terminal for detailed logs. Common fixes: verify API key, ensure documents are ingested.")
