from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from rag import answer
from llm import RetryExhaustedError
import logging

logging.basicConfig(level=logging.INFO)

class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

class SourceResponse(BaseModel):
    source: str
    chunk_id: int
    score: float
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]

app = FastAPI()

@app.post("/ask", response_model=AskResponse)
def ask(request: SearchRequest):
    try:
        return answer(request.question)
    except RetryExhaustedError:
        raise HTTPException(503, "LLM Service Unavailable")
    except Exception:
        logging.exception("An unknown error occurred")
        raise HTTPException(500, "Internal Server Error")
