import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from google.genai import types
from llm import client, with_retry
from pathlib import Path

BASE_DIR = Path(__file__).parent
AI_ENGINEERING_TXT = BASE_DIR / "data" / "ai_engineering.txt"
CHUNKS_JSON = BASE_DIR / "index" / "chunks.json"
CHUNKS_NPY = BASE_DIR / "index" / "chunks.npy"

with open(AI_ENGINEERING_TXT, "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

if os.path.exists(CHUNKS_JSON):
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
else:
    texts = splitter.split_text(text)
    chunks = [{"id": i, "text": text, "source": "ai_engineering.txt"} for i, text in enumerate(texts)]
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

if os.path.exists(CHUNKS_NPY):
    chunks_embeddings = np.load(CHUNKS_NPY)
else:
    chunks_embeddings = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = with_retry(lambda: client.models.embed_content(
            model="gemini-embedding-001",
            contents=[chunk["text"] for chunk in batch],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        ))
        for embedding in response.embeddings:
            chunks_embeddings.append(embedding.values)
    chunks_embeddings = np.array(chunks_embeddings)
    assert len(chunks) == chunks_embeddings.shape[0]
    np.save(CHUNKS_NPY, chunks_embeddings)

print(chunks_embeddings.shape)