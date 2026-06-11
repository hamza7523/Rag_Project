import os
os.environ["HF_HOME"] = r"E:/models/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"E:/models/sentence_transformers"
from semantic_cache import SemanticCache

from contextlib import asynccontextmanager
from typing import Optional
import time
# Add to imports at the top of api.py
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_metrics import (
    Request_Counter,
    Request_Latency,
    Active_Requests,
    CACHE_HIT_COUNTER,
)
from fastapi import FastAPI, HTTPException, Request  # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from pydantic import BaseModel, Field # type: ignore[import]

from retrieval import load_retriever, retrieve
from generation import answer   # your existing answer() function


# ── Pydantic models (API contracts) ──────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Defines the shape of every incoming query.
    FastAPI validates this automatically — malformed requests
    are rejected before they touch your RAG logic.
    """
    question: str = Field(..., min_length=3, description="The natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")

class SourceChunk(BaseModel):
    """Represents a single retrieved context chunk returned to the client."""
    rank: int
    source: str
    page: str | int
    content_preview: str          # first 200 chars — enough for the client to show provenance

class QueryResponse(BaseModel):
    """
    The full response returned to the client.
    Structured responses matter in production — the client (UI, downstream service)
    shouldn't have to parse free text to extract the answer.
    """
    question: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float             # how long the full pipeline took


# ── Application state ─────────────────────────────────────────────────────────
# These objects are loaded ONCE at startup and reused across every request.
# Loading ChromaDB + BM25 on every request would add 5-10s per query.

class AppState:
    vectorstore = None
    bm25_index = None
    documents = None
    metadatas = None


# ── Lifespan: startup and shutdown logic ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    """
    Runs ONCE when the server starts (before accepting any requests).
    Loads all heavy objects into AppState so requests can reuse them.

    Interview note: This is the standard FastAPI pattern for managing
    expensive shared resources (DB connections, ML models, indexes).
    The alternative — loading on every request — is a common anti-pattern
    that tanks performance under any real load.
    """
    print("🚀 Starting up — loading retriever components...")
    (
        AppState.vectorstore,
        AppState.bm25_index,
        AppState.documents,
        AppState.metadatas,
    ) = load_retriever()
    app.state.cache = SemanticCache()

    print("✅ Retriever loaded. Ready to serve requests.")

    yield   # server is now live and accepting requests
    
    app.state.cache.close()  # clean up Redis connection on shutdown

    # anything after yield runs on shutdown
    print("🛑 Shutting down — releasing resources.")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Beverages RAG Chatbot",
    description="Production RAG API — BM25 + semantic retrieval over internal sales data.",
    version="1.0.0",
    lifespan=lifespan,
)
# Add this AFTER your app = FastAPI(...) block
# This auto-exposes a /metrics endpoint that Prometheus will scrape.
# instrument(app) hooks into FastAPI's middleware to track HTTP-level metrics.
# expose(app) registers the /metrics route.
Instrumentator().instrument(app).expose(app)

# CORS middleware: allows the Streamlit UI (or any browser client) to call this API.
# Without this, browsers block cross-origin requests by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to specific origins in real production
    allow_methods=["*"],
    allow_headers=["*"],
)
#metrics endpoint for the prometheus to scrape later

    

# ── Health check endpoint ─────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Lightweight endpoint for load balancers and monitoring systems
    to verify the service is alive without triggering any RAG logic.

    Interview note: Every production service needs a /health endpoint.
    Kubernetes liveness/readiness probes hit this. If it returns non-200,
    the orchestrator knows to restart the pod or stop routing traffic to it.
    """
    retriever_loaded = AppState.vectorstore is not None
    return {
        "status": "healthy" if retriever_loaded else "degraded",
        "retriever_loaded": retriever_loaded,
    }


# ── Main query endpoint ───────────────────────────────────────────────────────

# Replace your entire /query endpoint with this instrumented version:

@app.post("/query", response_model=QueryResponse)
async def query_rag(http_request: Request, request: QueryRequest):

    # Increment active requests gauge — we're now processing one more request
    Active_Requests.inc()

    try:
        # ── Cache check ───────────────────────────────────────────────────────
        cached = http_request.app.state.cache.get(request.question)
        if cached:
            # Record the hit before returning — counter must fire even on early return
            CACHE_HIT_COUNTER.inc()
            Request_Counter.labels(status="success").inc()
            # Latency for cache hits is effectively 0 — record it as such
            Request_Latency.observe(0.0)
            return QueryResponse(
                question=request.question,
                answer=cached,
                sources=[],
                latency_ms=0.0,
            )

        # ── Guard ─────────────────────────────────────────────────────────────
        if AppState.vectorstore is None:
            Request_Counter.labels(status="error").inc()
            raise HTTPException(
                status_code=503,
                detail="Retriever not initialized. Service is starting up or encountered an error."
            )

        # ── Full RAG pipeline ─────────────────────────────────────────────────
        start_time = time.perf_counter()
        source_chunks = []

        try:
            answer_text, source_chunks = answer(
                query=request.question,
                vectorstore=AppState.vectorstore,
                bm25_index=AppState.bm25_index,
                documents=AppState.documents,
                metadatas=AppState.metadatas,
            )
        except Exception as e:
            Request_Counter.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

        latency_seconds = time.perf_counter() - start_time

        # Record latency in seconds — Prometheus convention is always seconds
        # The histogram will place this observation in the right bucket automatically
        Request_Latency.observe(latency_seconds)
        Request_Counter.labels(status="success").inc()

        # Store in cache for future requests
        http_request.app.state.cache.set(request.question, answer_text)

        sources = [
            SourceChunk(
                rank=i + 1,
                source=chunk["metadata"].get("source", "unknown"),
                page=chunk["metadata"].get("page", "?"),
                content_preview=chunk["content"][:200],
            )
            for i, chunk in enumerate(source_chunks)
        ]

        return QueryResponse(
            question=request.question,
            answer=answer_text,
            sources=sources,
            latency_ms=round(latency_seconds * 1000, 2),
        )

    finally:
        # finally block ALWAYS runs — whether success, error, or early return
        # This guarantees the gauge is decremented even if an exception escapes
        # Without this, a crash would leave the gauge permanently inflated — silent corruption
        Active_Requests.dec()
        
        
    
# ── Run directly for development ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn  # type: ignore[import]
    # host="0.0.0.0" makes the server reachable from other machines on the network,
    # not just localhost — important when Streamlit UI runs in a separate process.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)