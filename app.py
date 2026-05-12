"""
RAG Bug Intelligence Dashboard
================================
Project 2 — Retrieval-Augmented Generation over bug report data
Author: Gustavo Varejao

Architecture:
  CSV data → TF-IDF vectorization → cosine similarity retrieval
           → prompt augmentation → Claude API → Streamlit UI

Key learning areas:
  - How RAG retrieval works (TF-IDF + cosine similarity)
  - How to augment an LLM prompt with retrieved context
  - How mode-specific system prompts shape LLM behavior
  - How caching reduces latency and API cost
"""

import os
import math
import time
import re
from collections import Counter

import anthropic
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Bug Intelligence",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# 2. CUSTOM STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* Import monospace font for terminal feel */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* Global overrides */
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0e0e0f; border-right: 1px solid #1e1e21; }
[data-testid="stSidebar"] * { color: #e8e6e0 !important; }

/* Main area */
.stApp { background-color: #161618; }
h1, h2, h3 { color: #EF9F27 !important; font-family: 'JetBrains Mono', monospace; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1e1e21; border: 1px solid #2a2a2e;
    border-radius: 8px; padding: 12px;
}
[data-testid="stMetricLabel"] > div { color: #7a7870 !important; font-size: 11px; }
[data-testid="stMetricValue"] > div { color: #EF9F27 !important; font-size: 22px; }

/* Retrieved bug cards */
.bug-card {
    background: #1e1e21; border: 1px solid #2a2a2e;
    border-radius: 8px; padding: 12px; margin-bottom: 8px;
    font-size: 12px;
}
.bug-card:hover { border-color: rgba(239,159,39,0.4); }
.bug-id   { color: #5a5855; font-size: 10px; }
.bug-title { color: #e8e6e0; font-weight: 500; margin: 4px 0; }
.score-high { color: #97C459; }
.score-med  { color: #EF9F27; }
.score-low  { color: #F09595; }

/* Pipeline step badges */
.step { display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 10px; margin: 2px; border: 1px solid #2a2a2e; color: #7a7870; }
.step-active { border-color: rgba(239,159,39,0.5); color: #EF9F27;
               background: rgba(239,159,39,0.08); }
.step-done   { border-color: rgba(99,153,34,0.4); color: #97C459;
               background: rgba(99,153,34,0.08); }

/* Chat messages */
.user-msg {
    background: rgba(239,159,39,0.1); border: 1px solid rgba(239,159,39,0.25);
    border-radius: 8px; padding: 10px 14px; margin: 8px 0;
    color: #e8e6e0; font-size: 13px; text-align: right;
}
.assistant-msg {
    background: #1e1e21; border: 1px solid #2a2a2e;
    border-radius: 8px; padding: 10px 14px; margin: 8px 0;
    color: #e8e6e0; font-size: 13px;
}
.msg-label { font-size: 10px; color: #5a5855; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. DATA LOADING
#    We cache this so it only runs once per session.
#    st.cache_data is Streamlit's built-in caching decorator.
# ─────────────────────────────────────────────

@st.cache_data
def load_bugs(path: str = "bugs_extended.csv") -> pd.DataFrame:
    """Load bug data from CSV. Returns a DataFrame."""
    df = pd.read_csv(path)
    return df


# ─────────────────────────────────────────────
# 4. TF-IDF VECTORIZER
#
#    TF-IDF (Term Frequency-Inverse Document Frequency) converts
#    text into numeric vectors that capture word importance:
#
#    TF  = how often a word appears in THIS document
#    IDF = log(total docs / docs containing this word)
#          → rare words across corpus get higher weight
#
#    TF-IDF(word, doc) = TF * IDF
#    Result: each bug becomes a sparse dictionary {word: score}
# ─────────────────────────────────────────────

STOPWORDS = {
    "the","and","are","for","not","but","was","had","has","this",
    "that","with","from","they","have","been","were","will","can",
    "its","app","user","users","when","does","into","after","also",
    "more","than","one","two","three","each","via","per","show",
    "shows","still","even","just","only","then","than","being","very"
}

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]
    return tokens

def build_tfidf_index(df: pd.DataFrame) -> list[dict]:
    """
    Build a TF-IDF vector for each bug.

    We concatenate all text fields so the retrieval is multi-field:
    title + description + component + severity + customer_impact

    Returns a list of dicts, one per bug: {term: tfidf_score}
    """
    # Step 1: Tokenize each bug's full text
    doc_tokens = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row['description']} {row['component']} {row['severity']} {row['customer_impact']}"
        doc_tokens.append(tokenize(text))

    N = len(doc_tokens)  # total number of documents

    # Step 2: Compute document frequency (df) — how many docs contain each term
    doc_freq: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):               # set() deduplicates within a doc
            doc_freq[term] = doc_freq.get(term, 0) + 1

    # Step 3: Compute TF-IDF vector for each document
    tfidf_index = []
    for tokens in doc_tokens:
        tf = Counter(tokens)                   # raw term frequency
        vec = {}
        for term, count in tf.items():
            tf_score  = count / len(tokens)    # normalize by doc length
            idf_score = math.log((N + 1) / (doc_freq.get(term, 0) + 1))  # +1 smoothing
            vec[term] = tf_score * idf_score
        tfidf_index.append(vec)

    return tfidf_index


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Cosine similarity between two TF-IDF vectors.

    cos(θ) = (A · B) / (|A| * |B|)

    Range: 0 (no overlap) to 1 (identical).
    We only compute over the union of keys for efficiency.
    """
    all_terms = set(vec_a) | set(vec_b)
    dot_product = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in all_terms)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def vectorize_query(query: str) -> dict:
    """
    Convert a query string into a TF vector (no IDF — query is one doc).
    We skip IDF here; the cosine similarity with the corpus vectors
    still works well because the corpus vectors encode IDF weighting.
    """
    tokens = tokenize(query)
    if not tokens:
        return {}
    tf = Counter(tokens)
    return {term: count / len(tokens) for term, count in tf.items()}


# ─────────────────────────────────────────────
# 5. RETRIEVAL FUNCTION
#
#    Given a query, return the top-k most similar bugs.
#    This is the core of the RAG "R" (Retrieve) step.
# ─────────────────────────────────────────────

def retrieve(
    query: str,
    df: pd.DataFrame,
    tfidf_index: list[dict],
    top_k: int = 6
) -> list[dict]:
    """
    Retrieve the top_k bugs most similar to the query.

    Returns a list of dicts with bug data + similarity score,
    sorted by score descending.
    """
    query_vec = vectorize_query(query)
    if not query_vec:
        return []

    scored = []
    for i, bug_vec in enumerate(tfidf_index):
        score = cosine_similarity(query_vec, bug_vec)
        scored.append({
            "bug": df.iloc[i].to_dict(),
            "score": score,
        })

    # Sort by similarity, descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────
# 6. PROMPT AUGMENTATION
#
#    This is the "A" in RAG — Augment the LLM prompt
#    with the retrieved context so the model answers
#    from real data, not hallucinated knowledge.
#
#    Each mode has a different system prompt that shapes
#    how the model reasons over the retrieved bugs.
# ─────────────────────────────────────────────

def build_context_block(results: list[dict]) -> str:
    """
    Serialize retrieved bugs into a readable context string
    that gets injected into the system prompt.
    """
    lines = []
    for r in results:
        bug = r["bug"]
        score_pct = r["score"] * 100
        lines.append(
            f"[BUG-{bug['id']} | {bug['component']} | {bug['severity']} | sim={score_pct:.1f}%]\n"
            f"Title: {bug['title']}\n"
            f"Description: {bug['description']}\n"
            f"Customer Impact: {bug['customer_impact']}\n"
            f"Created: {bug['created_at']}"
        )
    return "\n\n".join(lines)


MODE_PROMPTS = {
    "Q&A": lambda ctx, q: f"""You are a senior engineering analyst. Answer the user's question about bug reports using ONLY the retrieved context below. Be specific, cite bug IDs (e.g. BUG-1), and be concise (3-5 sentences max). Do not invent bugs not in the context.

RETRIEVED BUGS:
{ctx}

User question: {q}""",

    "Similarity": lambda ctx, q: f"""You are a bug triage specialist detecting duplicate and related issues. Based on the retrieved bugs below, identify clusters of semantically similar issues, explain what makes them related, and flag likely duplicates. Reference bug IDs explicitly.

RETRIEVED BUGS:
{ctx}

Similarity query: {q}""",

    "Exec Summary": lambda ctx, q: f"""You are a Senior TPM preparing a VP-level executive briefing. Based on the retrieved bugs, write a concise executive summary with: (1) top risk areas, (2) customer impact, (3) three prioritized action items. Be direct and crisp — no filler language.

RETRIEVED BUGS:
{ctx}

Briefing focus: {q}""",

    "Fix Recs": lambda ctx, q: f"""You are a principal engineer doing a technical review. For each relevant bug in the retrieved context, provide: root cause hypothesis, recommended fix approach, and effort estimate (S/M/L). Reference bug IDs. Be opinionated and specific — not generic advice.

RETRIEVED BUGS:
{ctx}

Scope: {q}""",
}

MODE_SUGGESTED = {
    "Q&A": [
        "What are the highest priority Auth bugs?",
        "Which bugs pose legal or compliance risk?",
        "What's causing the most revenue loss?",
    ],
    "Similarity": [
        "Find bugs similar to payment failures",
        "Find duplicates related to login crashes",
        "Find bugs similar to cart data loss",
    ],
    "Exec Summary": [
        "Summarize the highest-risk areas for exec review",
        "What should the VP of Engineering prioritize this sprint?",
        "Summarize compliance and legal exposure",
    ],
    "Fix Recs": [
        "Suggest fixes for Auth session management bugs",
        "How should we fix the duplicate charge issue?",
        "What's the fix strategy for search performance bugs?",
    ],
}


# ─────────────────────────────────────────────
# 7. CLAUDE API CALL
#
#    After retrieval + augmentation, we call Claude.
#    The full augmented prompt goes in as the system message.
#    We use streaming so the response appears token-by-token.
# ─────────────────────────────────────────────

def call_claude_streaming(system_prompt: str, user_query: str, api_key: str):
    """
    Call Claude with a system prompt (contains retrieved context)
    and the user query. Streams the response back.

    Yields text chunks as they arrive.
    """
    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_query}],
    ) as stream:
        for text_chunk in stream.text_stream:
            yield text_chunk


# ─────────────────────────────────────────────
# 8. HELPER: SCORE COLOR
# ─────────────────────────────────────────────

def score_color(score: float) -> str:
    if score > 0.15:
        return "score-high"
    elif score > 0.07:
        return "score-med"
    return "score-low"

def severity_color(sev: str) -> str:
    return {"High": "#F09595", "Medium": "#EF9F27", "Low": "#97C459"}.get(sev, "#7a7870")


# ─────────────────────────────────────────────
# 9. SESSION STATE INITIALIZATION
#    Streamlit reruns top-to-bottom on every interaction.
#    st.session_state persists values across reruns.
# ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []         # list of {role, content, retrieved}

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = None


# ─────────────────────────────────────────────
# 10. SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## RAG//BugIntel")
    st.markdown("*Retrieval-Augmented Bug Analysis*")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get yours at console.anthropic.com",
    )

    st.divider()

    mode = st.radio(
        "Analysis Mode",
        options=["Q&A", "Similarity", "Exec Summary", "Fix Recs"],
        help=(
            "Q&A: Factual questions about bugs\n"
            "Similarity: Find related/duplicate bugs\n"
            "Exec Summary: VP-level briefing\n"
            "Fix Recs: Engineering recommendations"
        ),
    )

    top_k = st.slider(
        "Bugs to retrieve (k)",
        min_value=2, max_value=10, value=6,
        help="How many bugs to pull from the index per query. More = richer context, higher token cost.",
    )

    st.divider()
    st.markdown("**Pipeline**")

    pipeline_steps = ["encode query", "cosine retrieval", "rank & filter", "augment prompt", "llm generate"]
    current_step = st.session_state.pipeline_step

    for i, step in enumerate(pipeline_steps):
        if current_step is None:
            css = "step"
        elif step == current_step:
            css = "step step-active"
        elif pipeline_steps.index(step) < pipeline_steps.index(current_step):
            css = "step step-done"
        else:
            css = "step"
        st.markdown(f'<span class="{css}">● {step}</span>', unsafe_allow_html=True)

    st.divider()

    # Corpus stats
    try:
        df_stats = load_bugs()
        st.markdown("**Corpus stats**")
        col1, col2 = st.columns(2)
        col1.metric("Bugs", len(df_stats))
        col2.metric("Components", df_stats["component"].nunique())
        col1.metric("High sev", int((df_stats["severity"] == "High").sum()))
        col2.metric("Open", int((df_stats["status"] == "Open").sum()))
    except FileNotFoundError:
        st.warning("bugs_extended.csv not found")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.last_retrieved = []
        st.session_state.pipeline_step = None
        st.rerun()


# ─────────────────────────────────────────────
# 11. MAIN LAYOUT
# ─────────────────────────────────────────────

col_chat, col_context = st.columns([2, 1], gap="medium")

# ── Left: Chat panel ──────────────────────────
with col_chat:
    st.markdown("### Chat")

    # Suggested queries for current mode
    st.markdown("**Try a query:**")
    sq_cols = st.columns(len(MODE_SUGGESTED[mode]))
    suggested_query = None
    for i, sq in enumerate(MODE_SUGGESTED[mode]):
        if sq_cols[i].button(sq, key=f"sq_{i}_{mode}", use_container_width=True):
            suggested_query = sq

    st.divider()

    # Render chat history
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="user-msg"><div class="msg-label">you</div>{turn["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg"><div class="msg-label">rag//claude</div>{turn["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Query input
    with st.form("query_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your query",
            placeholder="Ask a question about the bug data...",
            height=80,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Run RAG ↗", use_container_width=True)

    # Use suggested query if a button was clicked
    if suggested_query:
        user_input = suggested_query
        submitted = True


# ── Right: Retrieved context panel ───────────
with col_context:
    st.markdown("### Retrieval Context")
    st.caption("Bugs retrieved for the last query, ranked by cosine similarity")

    if not st.session_state.last_retrieved:
        st.info("Retrieved bugs will appear here after each query.")
    else:
        for r in st.session_state.last_retrieved:
            bug = r["bug"]
            score = r["score"]
            score_pct = f"{score * 100:.1f}%"
            sev_color = severity_color(bug["severity"])
            score_css = score_color(score)

            st.markdown(f"""
<div class="bug-card">
  <div class="bug-id">BUG-{bug['id']} &nbsp;
    <span class="{score_css}" style="float:right; font-weight:500">{score_pct}</span>
  </div>
  <div class="bug-title">{bug['title']}</div>
  <span style="font-size:10px; color:{sev_color}; border:1px solid {sev_color}33;
    border-radius:3px; padding:1px 5px;">{bug['severity']}</span>
  <span style="font-size:10px; color:#7a7870; border:1px solid #2a2a2e;
    border-radius:3px; padding:1px 5px; margin-left:4px;">{bug['component']}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 12. RAG PIPELINE EXECUTION
#     Triggered when user submits a query.
#     Runs through all 5 pipeline steps in order.
# ─────────────────────────────────────────────

if submitted and user_input.strip():

    if not api_key:
        st.error("Add your Anthropic API key in the sidebar to continue.")
        st.stop()

    # Load data and build index (cached after first run)
    try:
        df = load_bugs()
    except FileNotFoundError:
        st.error("bugs_extended.csv not found. Make sure it's in the same directory as app.py.")
        st.stop()

    tfidf_index = build_tfidf_index(df)   # cached implicitly via @st.cache_data on load_bugs

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ── Step 1: Encode query ───────────────────
    st.session_state.pipeline_step = "encode query"
    st.rerun()

# Pipeline continuation after rerun — check if we're mid-pipeline
# We track this via a pending query in session_state
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if submitted and user_input.strip() and api_key:
    st.session_state.pending_query = {
        "query": user_input,
        "mode": mode,
        "top_k": top_k,
    }

if st.session_state.pending_query and st.session_state.pipeline_step == "encode query":
    pending = st.session_state.pending_query
    query   = pending["query"]
    k       = pending["top_k"]
    m       = pending["mode"]

    try:
        df         = load_bugs()
        tfidf_idx  = build_tfidf_index(df)

        # ── Step 2: Retrieve ───────────────────
        st.session_state.pipeline_step = "cosine retrieval"
        results = retrieve(query, df, tfidf_idx, top_k=k)

        # ── Step 3: Rank & filter ──────────────
        st.session_state.pipeline_step = "rank & filter"
        st.session_state.last_retrieved = results

        # ── Step 4: Augment prompt ─────────────
        st.session_state.pipeline_step = "augment prompt"
        context_block = build_context_block(results)
        system_prompt = MODE_PROMPTS[m](context_block, query)

        # ── Step 5: Generate ───────────────────
        st.session_state.pipeline_step = "llm generate"

        response_text = ""
        # Stream directly into the chat panel
        with col_chat:
            with st.spinner("Generating..."):
                response_placeholder = st.empty()
                for chunk in call_claude_streaming(system_prompt, query, api_key):
                    response_text += chunk
                    response_placeholder.markdown(
                        f'<div class="assistant-msg"><div class="msg-label">rag//claude</div>{response_text}▌</div>',
                        unsafe_allow_html=True,
                    )

        # Finalize: remove cursor, store in history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.session_state.pending_query = None
        st.session_state.pipeline_step = "llm generate"  # leave as done
        st.rerun()

    except anthropic.AuthenticationError:
        st.error("Invalid API key. Check your Anthropic API key in the sidebar.")
        st.session_state.pending_query = None
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.pending_query = None
