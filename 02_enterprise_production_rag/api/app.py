# """
# Enterprise Financial RAG FastAPI Application

# This module provides the production REST API layer orchestrating the full end-to-end 
# financial RAG pipeline:
# 1. Lifespan Warm-Up: Loads ML models (Bi-Encoder, Cross-Encoder, LLM client) and ChromaDB 
#    connections into `app.state` on server boot to avoid request-time initialization lag.
# 2. End-to-End Orchestration: Coordinates query reformulation, entity/ticker extraction, 
#    two-stage retrieval (vector search + reranking), grounded synthesis, and memory tracking.
# 3. Structured Request/Response Contracts: Uses Pydantic models to validate input payloads 
#    and guarantee typed, structured responses with full provenance (citations, scores, latency).
# 4. Session Management & Health Checks: Provides `/health` endpoints for container 
#    orchestration (Kubernetes/Docker) and `DELETE` routes for clearing conversational sessions.
# """
#==================================================================================================
import time
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# --- FIX: Ensure 02_enterprise_production_rag is in Python's search path ---
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.retrieve import EnterpriseRetriever, RetrievedChunk
from api.chat_manager import ChatSessionManager, ReformulatedQuery
from api.generate import RAGGenerator, GeneratedResponse

# ----------------Logging Configuration-----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("enterprise_rag_api")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma_sec_db"

# -------------Lifespan Management (Warm-up on boot, clean shutdown)----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronously initializes ML models and ChromaDB handles onto app.state
    to ensure non-blocking dependency injection across workers.
    """
    logger.info("⚙️ Booting Enterprise Financial RAG Service Layer...")
    if not DB_PATH.exists():
        logger.warning(f"⚠️ WARNING: Database directory {DB_PATH} not found. Run ingest_data.py first.")

    # Initialize singletons on app.state
    try:
        app.state.retriever = EnterpriseRetriever(db_path=DB_PATH)
        app.state.chat_manager = ChatSessionManager()
        app.state.generator = RAGGenerator()
        logger.info("✅ Two-Stage Retriever, Session Manager & Generator successfully loaded into memory.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system components: {e}")
        raise e

    yield

    # Clean teardown
    logger.info("🛑 Unloading model weights and flushing active sessions...")
    if hasattr(app.state, "chat_manager"):
        app.state.chat_manager.sessions.clear()
    logger.info("✓ Shutdown complete.")

# ----------------------FastAPI App Initialization------------------------------
app = FastAPI(
    title="Enterprise SEC 10-K RAG Engine",
    description="High-throughput, asynchronous Two-Stage Retrieval Augmented Generation API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific UI domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------Request & Response Schemas-------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(default="default_session", description="Unique conversation session token")
    message: str = Field(..., min_length=1, description="User query or financial follow-up question")
    target_ticker: Optional[str] = Field(default=None, description="Optional explicit ticker override (e.g., AAPL)")
    top_k_retrieve: Optional[int] = Field(default=15, ge=1, le=50, description="Stage 1 Bi-Encoder retrieval depth")
    top_k_return: Optional[int] = Field(default=4, ge=1, le=10, description="Stage 2 Cross-Encoder rerank output count")

class ContextSnippet(BaseModel):
    content: str
    ticker: str
    source_doc_id: str
    rerank_score: float

class ChatResponse(BaseModel):
    session_id: str
    original_query: str
    rewritten_query: str
    detected_ticker: Optional[str]
    answer: str
    is_grounded: bool
    citations: List[str]
    context: List[ContextSnippet]
    latency_seconds: float

# ----------------------API Endpoints--------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check(request: Request):
    """Liveness/readiness probe for container orchestration."""
    is_ready = hasattr(request.app.state, "retriever") and request.app.state.retriever is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "vector_db_connected": is_ready
    }

@app.post("/api/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(payload: ChatRequest, request: Request):
    """
    Fully non-blocking RAG execution pipeline:
    1. Asynchronously rewrites queries & resolves coreferences.
    2. Executes Two-Stage retrieval (Bi-Encoder recall + Cross-Encoder rerank).
    3. Synthesizes grounded answers with strict Pydantic output parsing.
    4. Records conversation history with bounded memory sliding windows.
    """
    t_start = time.perf_counter()

    retriever: EnterpriseRetriever = request.app.state.retriever
    chat_mgr: ChatSessionManager = request.app.state.chat_manager
    generator: RAGGenerator = request.app.state.generator

    # Coreference Query Reformulation & Entity Extraction
    reformulated: ReformulatedQuery = await chat_mgr.a_rewrite_query(
        session_id=payload.session_id,
        latest_query=payload.message
    )
    
    # Priority: Explicit request parameter > LLM inferred ticker > None
    effective_ticker = payload.target_ticker or reformulated.detected_ticker

    # Asynchronous Two-Stage Retrieval
    try:
        retrieved_chunks: List[RetrievedChunk] = await retriever.a_retrieve_and_rerank(
            query=reformulated.standalone_query,
            target_ticker=effective_ticker,
            top_k_retrieve=payload.top_k_retrieve,
            top_k_return=payload.top_k_return
        )
    except Exception as e:
        logger.error(f"❌ Retrieval failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval subsystem failed: {str(e)}"
        )

    # Asynchronous Grounded Answer Synthesis
    try:
        gen_result: GeneratedResponse
        gen_result, _ = await generator.a_generate_grounded_answer(
            query=reformulated.standalone_query,
            retrieved_context=retrieved_chunks
        )
    except Exception as e:
        logger.error(f"❌ Generation failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation subsystem failed: {str(e)}"
        )

    # Update Conversation Memory (Rolling window enforced automatically)
    chat_mgr.add_message(payload.session_id, "user", payload.message)
    chat_mgr.add_message(payload.session_id, "assistant", gen_result.answer)

    total_latency = time.perf_counter() - t_start

    return ChatResponse(
        session_id=payload.session_id,
        original_query=payload.message,
        rewritten_query=reformulated.standalone_query,
        detected_ticker=effective_ticker,
        answer=gen_result.answer,
        is_grounded=gen_result.is_grounded,
        citations=gen_result.citations,
        context=[
            ContextSnippet(
                content=chunk.content,
                ticker=chunk.ticker,
                source_doc_id=chunk.source_doc_id,
                rerank_score=chunk.rerank_score
            )
            for chunk in retrieved_chunks
        ],
        latency_seconds=round(total_latency, 3)
    )

@app.delete("/api/v1/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def clear_session(session_id: str, request: Request):
    """Flushes conversational memory for a specific user session."""
    chat_mgr: ChatSessionManager = request.app.state.chat_manager
    chat_mgr.clear_session(session_id)
    return {"status": "success", "message": f"Session '{session_id}' memory cleared."}