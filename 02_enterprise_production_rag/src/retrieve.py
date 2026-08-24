# """
# Enterprise Two-Stage RAG Retriever Pipeline

# This script defines an asynchronous, production-ready retrieval module for querying 
# financial SEC 10-K data. It utilizes a two-stage "Recall & Precision" architecture:
# 1. Stage 1 (Recall): Uses a fast Bi-Encoder to search a ChromaDB vector database 
#    for the top-K most similar chunks, supporting exact-match ticker filtering.
# 2. Stage 2 (Precision): Uses a heavier, highly accurate Cross-Encoder to score and 
#    rerank the retrieved chunks based on deep semantic relevance to the query.
# 3. Async Design: CPU/GPU intensive tasks are wrapped in `asyncio.to_thread` to ensure 
#    the pipeline is non-blocking and safe for highly concurrent web frameworks (e.g., FastAPI).
# 4. Structured Output: Returns results strictly typed via a Pydantic schema.
# """
# ==============================================================================================
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------Structured Output Schema----------------------
class RetrievedChunk(BaseModel):
    content: str = Field(description="The actual text content of the SEC chunk")
    ticker: str = Field(description="The financial ticker(e.g. AAPL)")
    source_doc_id: str = Field(description="Original document ID from dataset")
    rerank_score: float = Field(description="Cross-Encoder relevance score")

# ------------------Logging Setup-------------------
logger = logging.getLogger(__name__)

class EnterpriseRetriever:
    def __init__(
        self, 
        db_path: Path = Path(__file__).resolve().parent.parent / "data" / "chroma_sec_db",
        collection_name: str = "sec_10k_reports",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base"
    ):
        logger.info("🔌 Connecting to ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_collection(collection_name)

        # Detect optimal hardware to prevent CPU bottlenecking
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        logger.info(f"🧠 Loading Bi-Encoder [{embedding_model_name}] on {self.device.upper()}...")
        self.encoder = SentenceTransformer(embedding_model_name, device=self.device)
        
        logger.info(f"🎯 Loading Cross-Encoder [{reranker_model_name}] on {self.device.upper()}...")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

    async def a_retrieve_and_rerank(
        self,
        query: str,
        target_ticker: Optional[str] = None,
        top_k_retrieve: int = 15,
        top_k_return: int = 4
    ) -> List[RetrievedChunk]:
        """
        Asynchronous, non-blocking two-stage retrieval pipeline.
        Safe to call directly from FastAPI route handlers.
        """
        # STAGE 1: Bi-Encoder Vector Search (Recall)
        # Offload encoding to a separate thread so asyncio enent loop isn't blocked
        query_vector = await asyncio.to_thread(
            self.encoder.encode,
            [query],
            normalize_embeddings=True
        )

        # Apply strict metadata pre-filtering if a specific company is requested
        where_filter = {"ticker": target_ticker.upper()} if target_ticker else None

        # Chroma query is I/O bound, can be execurted directly or thread-wrapped
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=query_vector.tolist(),
            n_results=top_k_retrieve,
            where=where_filter,
            include=["documents", "metadatas"]
        )

        if not results["documents"] or not results["documents"][0]:
            logger.warning(f"No chunks found for query: '{query}'")
            return []

        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]

        # STAGE 2: Cross-Encoder Reranking (Precision)
        cross_inputs = [[query, doc] for doc in retrieved_docs]

        # Offload intensive reranker neural network to thread pool
        rerank_scores = await asyncio.to_thread(self.reranker.predict, cross_inputs)

        # Sort descending by cross-encoder score using vectorized Numpy
        ranked_indices = np.argsort(rerank_scores)[::-1]

        final_chunks: List[RetrievedChunk] = []
        for idx in ranked_indices[:top_k_return]:
            meta = retrieved_metas[idx]
            final_chunks.append(
                RetrievedChunk(
                    content=retrieved_docs[idx],
                    ticker=meta.get("ticker", "UNKNOWN"),
                    source_doc_id=meta.get("source_doc_id", "UNKNOWN"),
                    rerank_score=float(rerank_scores[idx])
                )
            ) 

        return final_chunks