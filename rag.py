import json
import numpy as np
from llm import client, with_retry
from google.genai import types
import os
from pathlib import Path

SIMILARITY_THRESHOLD = 0.65
BASE_DIR = Path(__file__).parent
CHUNKS_JSON = BASE_DIR / "index" / "chunks.json"
CHUNKS_NPY = BASE_DIR / "index" / "chunks.npy"

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunks_embeddings = np.load(CHUNKS_NPY)

chunks_norm = chunks_embeddings / np.linalg.norm(chunks_embeddings, keepdims=True, axis=1)

def search_chunks(query, k=3):
    response = with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    ))

    query_embedding = response.embeddings[0].values
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    scores = chunks_norm @ query_norm
    topK_scores = np.argsort(scores)[::-1][:k]

    return [(scores[i], chunks[i]) for i in topK_scores]

def build_prompt(result, query):
    context=""
    for i, (score, chunk) in enumerate(result, start=1):
        context += f"[{i}] {chunk["text"]}\n"
    return f"Context:\n{context}\n Question: {query}"

SYSTEM = """You are a helpful assistant that answers questions using ONLY the numbered context passages provided.

Rules:
1. Use only the information in the context. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "I couldn't find this in the provided documents."
3. Cite the passages you used, like [1] or [2].
4. Answer in the same language as the question and keep it concise.
5. If the context only partly answers the question, say what the context does cover and state clearly what it does not cover."""

def answer(query):
    result = search_chunks(query)
    if not result or result[0][0] < SIMILARITY_THRESHOLD:
        fallback = "Answer could not be retrieved. We don't have enough information to provide the answer to your query."
        return {"answer": fallback, "sources": []}

    prompt = build_prompt(result, query)

    response = with_retry(lambda: client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"),
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2)
    ))

    sources = [{
        "source": chunk["source"],
        "chunk_id": chunk["id"],
        "score": float(score),
        "text": chunk["text"]
    }
        for score, chunk in result
    ]

    return {"answer": response.text, "sources": sources}

if __name__ == "__main__":
    print(answer("What happens when a model memorizes its training examples instead of learning general patterns?"))