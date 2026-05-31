"""
Restaurant Review RAG — Streamlit Web UI
Pipeline  : ChromaDB (local) → Google Search → Llama3.2 (Ollama)
Memory    : SQLite persistent storage
"""

import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS
from vector import retriever
from memory import save_conversation, get_history, clear_history

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL & PROMPT
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    return OllamaLLM(model="llama3.2")

model = load_model()

TEMPLATE = """
You are an expert in answering questions about a restaurant based on customer reviews.

Here are relevant reviews from our database:
{reviews}

Here is additional context from web search:
{web_context}

Answer the following question using both sources. 
If reviews mention it, prioritise that. If not found in reviews, use web context.
Be helpful, concise and friendly.

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)
chain  = prompt | model


# ═══════════════════════════════════════════════════════════════════════════
#  GOOGLE / WEB SEARCH  (via DuckDuckGo — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return combined snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " restaurant review", max_results=max_results))
        if not results:
            return "No web results found."
        snippets = [f"- {r['title']}: {r['body']}" for r in results]
        return "\n".join(snippets)
    except Exception as e:
        return f"Web search unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_query(question: str) -> tuple[str, str, str]:
    """
    Run full RAG pipeline:
    1. Retrieve from ChromaDB
    2. Search web
    3. Generate answer with Llama3.2
    Returns (answer, rag_context, web_context)
    """
    # Step 1 — RAG retrieval
    rag_docs    = retriever.invoke(question)
    rag_context = "\n\n".join([d.page_content for d in rag_docs]) if rag_docs else "No relevant reviews found."

    # Step 2 — Web search
    web_context = web_search(question)

    # Step 3 — LLM answer
    answer = chain.invoke({
        "reviews":     rag_context,
        "web_context": web_context,
        "question":    question,
    })

    # Determine source
    source = "both"
    if "No relevant reviews" in rag_context:
        source = "google"
    elif "No web results" in web_context:
        source = "rag"

    return answer, rag_context, web_context, source


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Restaurant Review Assistant",
        page_icon="🍕",
        layout="wide",
    )

    st.title("🍕 Restaurant Review Assistant")
    st.caption("Powered by ChromaDB + Web Search + Llama3.2 (local)")

    # ── Sidebar — History ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("💬 Conversation History")

        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.success("History cleared!")
            st.rerun()

        history = get_history(limit=30)
        if not history:
            st.caption("No conversations yet.")
        else:
            for item in reversed(history):
                with st.expander(f"Q: {item['question'][:40]}…", expanded=False):
                    st.markdown(f"**Source:** `{item['source']}`")
                    st.markdown(f"**Time:** {item['timestamp'][:19]}")
                    st.markdown(f"**Answer:** {item['answer'][:300]}…")

    # ── Main — Query input ────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Ask about the restaurant:",
            placeholder="e.g. How is the pizza? Is the service good? What do people say about the pasta?",
        )
    with col2:
        submit = st.button("🔍 Ask", use_container_width=True)

    # ── Suggested questions ───────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    q_cols = st.columns(4)
    suggestions = [
        "How is the pizza?",
        "Is the service good?",
        "What about the ambience?",
        "Best dishes to order?",
    ]
    for i, suggestion in enumerate(suggestions):
        if q_cols[i].button(suggestion, use_container_width=True):
            question = suggestion
            submit   = True

    # ── Run pipeline ──────────────────────────────────────────────────────
    if submit and question:
        with st.spinner("Searching reviews and web…"):
            answer, rag_context, web_context, source = run_query(question)
            save_conversation(question, answer, source)

        # Answer
        st.markdown("---")
        st.subheader("💡 Answer")
        st.markdown(answer)

        # Source badge
        source_labels = {
            "rag":    "📚 From restaurant reviews",
            "google": "🌐 From web search",
            "both":   "📚🌐 From reviews + web search",
        }
        st.caption(source_labels.get(source, source))

        # Expandable context
        with st.expander("📚 Retrieved Reviews (RAG context)"):
            st.text(rag_context)

        with st.expander("🌐 Web Search Results"):
            st.text(web_context)

    # ── Recent conversations ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🕐 Recent Conversations")
    history = get_history(limit=5)
    if not history:
        st.caption("No conversations yet. Ask your first question above!")
    else:
        for item in reversed(history):
            with st.expander(f"Q: {item['question']}", expanded=False):
                st.markdown(f"**A:** {item['answer']}")
                st.caption(f"Source: `{item['source']}` | {item['timestamp'][:19]}")


if __name__ == "__main__":
    main()
