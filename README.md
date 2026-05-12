# RAG Bug Intelligence Dashboard

A conversational AI app that answers questions about bug reports using
Retrieval-Augmented Generation (RAG). Built on the Claude API and Streamlit.

---

## Overview

This is Project 2 in a series of AI portfolio projects.

**Project 1** used OpenAI LLM for classification and summarization, and the OpenAI embeddings API for duplicate detection.  
**Project 2** (this project) implements RAG, including TF-IDF vectorization and cosine similarity from scratch (no vector DB or embeddings API), to show how retrieval works at the algorithm level, then hands off to Claude for generation.

---

## Architecture

```
bugs_extended.csv
      ↓
  Pandas DataFrame
      ↓
  TF-IDF Vectorization          ← tokenize + compute tf * idf per bug
      ↓
  Cosine Similarity Retrieval   ← rank bugs by similarity to user query
      ↓
  Prompt Augmentation           ← inject retrieved bugs into system prompt
      ↓
  Claude API (claude-sonnet-4)  ← generate grounded answer
      ↓
  Streamlit UI                  ← stream response, show retrieval panel
```

## Key Design Decisions

- **TF-IDF over embeddings API** — Project 1 used OpenAI embeddings (API call, cost, latency).
  This project implements TF-IDF + cosine similarity in pure Python to make the retrieval
  mechanism transparent and learnable. Trade-off: less semantic generalization, but zero cost
  and fully inspectable.

- **No vector database** — At 100 bugs, brute-force cosine similarity is fast enough. In
  production you'd use FAISS, Pinecone, or pgvector. The algorithm is identical.

- **Mode-specific system prompts** — Each analysis mode (Q&A, Similarity, Exec Summary,
  Fix Recs) uses a different system prompt that shapes how the model reasons. Same retrieval,
  different generation behavior. This is a key RAG design pattern.

- **Workflow, not agents** — Same philosophy as Project 1: deterministic, sequential pipeline.
  Predictable and debuggable. Every step is visible in the UI.

- **Streaming** — Claude's response streams token-by-token using `client.messages.stream()`.
  Reduces perceived latency significantly.

---

## Running Locally

```bash
# 1. Clone or download this folder
cd rag_bug_intelligence

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure bugs_extended.csv is in this directory

# 5. Run
streamlit run app.py
```

Enter your Anthropic API key in the sidebar when the app opens.  
Get a key at: https://console.anthropic.com

---

## Project Structure

```
rag_bug_intelligence/
├── app.py                  ← full application (single file, heavily commented)
├── bugs_extended.csv       ← 100 bug reports across 8 components
├── requirements.txt
└── README.md
```

---

## Learning Areas

The code in `app.py` is heavily commented to be a learning resource:

| Section | What you'll learn |
|---------|------------------|
| `tokenize()` | Text preprocessing for IR |
| `build_tfidf_index()` | How TF-IDF scoring works mathematically |
| `cosine_similarity()` | Vector similarity as a retrieval signal |
| `retrieve()` | The RAG "R" — brute-force k-NN retrieval |
| `build_context_block()` | The RAG "A" — how context is formatted for injection |
| `MODE_PROMPTS` | How system prompt design shapes LLM behavior |
| `call_claude_streaming()` | Streaming with the Anthropic Python SDK |
| `st.session_state` | Streamlit state management across reruns |

---

## Future Improvements

- Replace TF-IDF with real embeddings (Voyage AI, OpenAI, or local sentence-transformers)
- Add a FAISS index for sub-millisecond retrieval at scale
- Persist chat history across sessions
- Add confidence scoring and hallucination detection
- Support real-time ingestion from Jira or GitHub Issues API

---

## Author

Gustavo Varejao  
Senior Technical Program Manager
