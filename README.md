# RAG Q&A API

A small RAG service: it answers questions from a text document, cites the chunks it used, and abstains when the document doesn't cover the question. Built with FastAPI, the Gemini API and NumPy.

## How it works

```
Ingest:  document -> chunk (500/100) -> embed -> index/chunks.json + chunks.npy
Ask:     question -> embed -> cosine top-3 -> below threshold? fallback : LLM -> answer + sources
```

- **Chunking:** `RecursiveCharacterTextSplitter` (paragraph-first, so no mid-word cuts).
- **Embeddings:** `gemini-embedding-001` with `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types.
- **Abstention:** similarity threshold of 0.65 (calibrated on 6 questions: answerable 0.76-0.80, unrelated 0.51-0.57), plus a grounded system prompt that cites `[n]` and states what's missing.
- **Reliability:** retry with backoff on 429/5xx; the API returns 503 when retries run out and a generic 500 for other errors.

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set GEMINI_API_KEY and GEMINI_MODEL
python ingest.py            # build the index
python -m uvicorn api:app --reload
```

Docs: http://127.0.0.1:8000/docs

## API

`POST /ask` with `{"question": "Why can small chunks hurt retrieval?"}` returns:

```json
{
  "answer": "... [1]",
  "sources": [{"source": "ai_engineering.txt", "chunk_id": 13, "score": 0.77, "text": "..."}]
}
```

An unrelated question returns `200` with a fallback message and `sources: []`. An empty or too-long question (1-500 characters) returns `422`.