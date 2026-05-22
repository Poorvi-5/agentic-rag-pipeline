# app/streamlit_app.py
"""
Agentic RAG Pipeline — Streamlit Chat UI
Full integration of all 5 steps into one chat interface.

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import time
from datetime import datetime

# ── Page config — must be first Streamlit call ────────────────
st.set_page_config(
    page_title  = "Agentic RAG Pipeline",
    page_icon   = "🤖",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Chat bubbles */
    .user-bubble {
        background: #1e3a5f;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .bot-bubble {
        background: #1a1a2e;
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #333;
        word-wrap: break-word;
    }
    /* Score badges */
    .score-high   { color: #00c853; font-weight: bold; }
    .score-medium { color: #ffd600; font-weight: bold; }
    .score-low    { color: #ff5252; font-weight: bold; }
    /* Tool badge */
    .tool-badge {
        background: #0d3b66;
        color: #4fc3f7;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78em;
        font-family: monospace;
    }
    /* Sidebar metric */
    .metric-box {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
        margin: 6px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────
def init_session():
    if "messages"       not in st.session_state:
        st.session_state.messages       = []
    if "total_queries"  not in st.session_state:
        st.session_state.total_queries  = 0
    if "total_passed"   not in st.session_state:
        st.session_state.total_passed   = 0
    if "tools_used"     not in st.session_state:
        st.session_state.tools_used     = {}
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False

init_session()


# ── Load pipeline (cached so it loads only once) ──────────────
@st.cache_resource(show_spinner="Loading Agentic RAG Pipeline...")
def load_pipeline():
    """Load all pipeline components once and cache them."""
    try:
        from src.config import validate_config
        validate_config()

        # Pre-load the embedding model
        from src.tools.vector_search import _load_resources
        _load_resources()

        # Pre-create sample DB if not exists
        from src.tools.sql_tool import create_sample_database
        from src.config import DATA_DIR
        db_path = DATA_DIR / "sample_database.db"
        if not db_path.exists():
            create_sample_database()

        return True, "Pipeline loaded successfully"
    except Exception as e:
        return False, str(e)


# ── Helper: get score colour ──────────────────────────────────
def score_color(score: float) -> str:
    if score >= 0.75:
        return "score-high"
    elif score >= 0.55:
        return "score-medium"
    return "score-low"


def score_emoji(score: float) -> str:
    if score >= 0.75:
        return "✅"
    elif score >= 0.55:
        return "⚠️"
    return "❌"


# ── Helper: extract tool names from steps ─────────────────────
def extract_tools(steps: list) -> list:
    tools = []
    for step in steps:
        if isinstance(step, tuple) and len(step) >= 1:
            action = step[0]
            if hasattr(action, "tool"):
                tools.append(action.tool)
    return list(dict.fromkeys(tools))  # deduplicated


# ── Process a query through the full pipeline ─────────────────
def process_query(query: str) -> dict:
    """Run query through reflection pipeline and return result dict."""
    from src.reflection.evaluator import run_with_reflection

    start_time = time.time()
    result     = run_with_reflection(query)
    elapsed    = time.time() - start_time

    # Extract tools used from all evaluation attempts
    tools = []
    for evaluation in result.evaluations:
        pass  # evaluations don't have steps

    return {
        "query":    query,
        "answer":   result.final_answer,
        "score":    result.final_score,
        "passed":   result.passed,
        "attempts": result.attempts,
        "elapsed":  elapsed,
        "evals":    result.evaluations,
    }


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Agentic RAG Pipeline")
        st.markdown("*Built with LangChain + Groq + FAISS*")
        st.divider()

        # Pipeline status
        st.markdown("### ⚙️ Pipeline Status")
        ready, msg = load_pipeline()

        if ready:
            st.session_state.pipeline_ready = True
            st.success("Pipeline Ready ✅")
        else:
            st.error(f"Pipeline Error ❌\n{msg}")
            st.stop()

        st.divider()

        # Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.total_queries)
        with col2:
            passed = st.session_state.total_passed
            total  = st.session_state.total_queries
            rate   = f"{int(passed/total*100)}%" if total > 0 else "—"
            st.metric("Pass Rate", rate)

        st.divider()

        # Available tools
        st.markdown("### 🛠️ Available Tools")
        tools_info = [
            ("🔍", "vector_search",    "Knowledge base"),
            ("🗄️", "sql_database",     "Structured data"),
            ("🌐", "web_search",       "Real-time web"),
            ("🌤️", "external_api",     "Live weather"),
        ]
        for icon, name, desc in tools_info:
            st.markdown(
                f"{icon} **{name}**  \n"
                f"<small style='color:gray'>{desc}</small>",
                unsafe_allow_html=True
            )

        st.divider()

        # Sample queries
        st.markdown("### 💡 Try These Queries")
        sample_queries = [
            "What is Retrieval Augmented Generation?",
            "Show all products under $100",
            "What is the weather in Delhi?",
            "Latest news in AI today",
            "What are embeddings in machine learning?",
            "List employees in Engineering department",
            "What is the temperature in Mumbai?",
            "How does LangChain work?",
        ]

        for sq in sample_queries:
            if st.button(
                sq[:42] + "..." if len(sq) > 42 else sq,
                use_container_width=True,
                key=f"sample_{sq[:20]}"
            ):
                st.session_state["pending_query"] = sq

        st.divider()

        # Clear chat
        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.messages      = []
            st.session_state.total_queries = 0
            st.session_state.total_passed  = 0
            st.rerun()


# ── Main chat area ────────────────────────────────────────────
def render_chat():
    st.markdown("## 💬 Chat with Your Knowledge Base")
    st.markdown(
        "Ask anything — I'll search your documents, database, "
        "web, or live APIs automatically."
    )
    st.divider()

    # Render existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            # Bot message
            meta    = msg.get("meta", {})
            score   = meta.get("score",    0.0)
            passed  = meta.get("passed",   False)
            elapsed = meta.get("elapsed",  0.0)
            attempts= meta.get("attempts", 1)
            evals   = meta.get("evals",    [])

            # Answer bubble
            st.markdown(
                f'<div class="bot-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True
            )

            # Metadata row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                css = score_color(score)
                st.markdown(
                    f'<span class="{css}">'
                    f'{score_emoji(score)} Score: {score:.2f}'
                    f'</span>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"⏱️ <small>{elapsed:.1f}s</small>",
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"🔄 <small>{attempts} attempt(s)</small>",
                    unsafe_allow_html=True
                )
            with col4:
                status = "✅ Passed" if passed else "⚠️ Best effort"
                st.markdown(
                    f"<small>{status}</small>",
                    unsafe_allow_html=True
                )

            # Evaluation details expander
            if evals:
                with st.expander("📋 View Evaluation Details"):
                    for i, ev in enumerate(evals, 1):
                        st.markdown(f"**Attempt {i}**")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Relevance",    f"{ev.relevance:.2f}")
                        c2.metric("Grounding",    f"{ev.grounding:.2f}")
                        c3.metric("Completeness", f"{ev.completeness:.2f}")
                        c4.metric("Final Score",  f"{ev.score:.2f}")
                        st.caption(f"💬 Feedback: {ev.feedback}")
                        if i < len(evals):
                            st.divider()

            st.markdown("")  # spacing


# ── Handle user input ─────────────────────────────────────────
def handle_input():
    # Check for sidebar sample query click
    pending = st.session_state.pop("pending_query", None)

    # Chat input box at bottom
    user_input = st.chat_input(
        "Ask me anything — documents, database, web, or weather..."
    )

    # Use whichever came in
    query = pending or user_input

    if not query:
        return

    if not st.session_state.pipeline_ready:
        st.error("Pipeline not ready yet. Please wait.")
        return

    # Add user message to history
    st.session_state.messages.append({
        "role":    "user",
        "content": query
    })
    st.session_state.total_queries += 1

    # Show thinking spinner while processing
    with st.spinner("🧠 Agent is thinking... selecting tools... reflecting..."):
        result = process_query(query)

    # Add bot response to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "meta":    result
    })

    if result["passed"]:
        st.session_state.total_passed += 1

    st.rerun()


# ── Entry point ───────────────────────────────────────────────
def main():
    render_sidebar()
    render_chat()
    handle_input()


if __name__ == "__main__":
    main()