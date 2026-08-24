# """
# RAG System Evaluation & Benchmarking Pipeline

# This script implements an automated evaluation suite using the Ragas framework
# to assess the end-to-end performance of the SEC 10-K RAG pipeline:
# 1. Embedding Adapter: Bridges the local SentenceTransformer Bi-Encoder with Ragas 
#    interfaces via a custom `RagasBGEAdapter` wrapper.
# 2. Core Semantic Metrics: Evaluates output quality using three key RAG metrics:
#    - Faithfulness: Measures whether generated claims are grounded in context.
#    - Answer Relevancy: Checks how directly the response addresses the prompt.
#    - Context Precision: Measures if relevant documents are ranked at the top.
# 3. Rate Limit Management: Enforces sequential execution (`max_workers=1`) and 
#    controlled sleep intervals between evaluation passes to prevent API rate-limit spikes.
# 4. Latency & Observability Logging: Measures retrieval latency, generation latency, 
#    and semantic scores, exporting the consolidated benchmark dataset to `rag_eval_report.csv`.
# """
# ================================================================================================
import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from dotenv import load_dotenv

# 1. Resolve path to 02_enterprise_production_rag and add to sys.path FIRST
PACKAGE_ROOT = Path(__file__).resolve().parent.parent 
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# 2. Now import from src and api safely (Removed LocalBGEEmbeddingFunction)
from src.retrieve import EnterpriseRetriever
from api.generate import RAGGenerator

# RAGAS Imports
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.run_config import RunConfig
from openai import AsyncOpenAI

load_dotenv(override=True)
logger = logging.getLogger("eval_pipeline")

# Monkeypatch for RAGAS custom embedding compatibility(avoids rigid embedding-type checks within the Ragas library)
AnswerRelevancy._validate_embeddings = lambda self: None

class RagasBGEAdapter(BaseRagasEmbedding):
    """Adapts local Bi-Encoder (SentenceTransformer) for RAGAS compatibility."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def embed_text(self, text: str, **kwargs) -> list[float]:
        # .tolist() is critical here to convert numpy arrays to native python floats
        return self.encoder.encode([text])[0].tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)

    def embed_query(self, text: str) -> list[float]:
        return self.encoder.encode([text])[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

class ProductionRAGEvaluator:
    def __init__(self):
        db_path = Path(__file__).resolve().parent.parent / "data" / "chroma_sec_db"
        
        # 1. Initialize core system
        self.retriever = EnterpriseRetriever(db_path=db_path)
        self.generator = RAGGenerator(model_name="gemini-3.5-flash")
        
        # 2. Initialize RAGAS Judges
        gemini_key = os.getenv("GEMINI_API_KEY")
        eval_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_key)

        self.ragas_judge = llm_factory(model="gemini-3.5-flash-lite", client=eval_client)
        
        # 3. Connect Local Embeddings directly from the Retriever to RAGAS
        self.ragas_embeddings = RagasBGEAdapter(self.retriever.encoder)
        
        # 4. Metrics Setup
        self.faithfulness = Faithfulness(llm=self.ragas_judge)
        self.relevancy = AnswerRelevancy(llm=self.ragas_judge, embeddings=self.ragas_embeddings)
        self.precision = ContextPrecision(llm=self.ragas_judge)
        self.metrics = [self.faithfulness, self.relevancy, self.precision]

        # Ragas RunConfig controls how aggressively it hits the API
        self.run_config = RunConfig(
            timeout=60, 
            max_retries=10, 
            max_wait=30,     # Max wait time between retries
            max_workers=1    # Forces sequential execution to prevent spiking the RPM limit
        )

    async def evaluate_query(self, item: Dict[str, Any]) -> Dict[str, Any]:
        query = item["query"]
        ground_truth = item["expected"]
        
        t0 = time.perf_counter()
        
        # System Retrieval
        t_ret_start = time.perf_counter()
        retrieved_chunks = await self.retriever.a_retrieve_and_rerank(query=query)
        retrieval_latency = time.perf_counter() - t_ret_start
        context_texts = [c.content for c in retrieved_chunks]

        # System Generation
        t_gen_start = time.perf_counter()
        gen_result, _ = await self.generator.a_generate_grounded_answer(query, retrieved_chunks)
        generation_latency = time.perf_counter() - t_gen_start

        # manual 5-second buffer before RAGAS starts its evaluations
        await asyncio.sleep(5)

        # Compute Semantic RAGAS Scores
        scores = {}
        for metric in self.metrics:
            try:
                if metric.name == "faithfulness":
                    res = await metric.ascore(user_input=query, response=gen_result.answer, retrieved_contexts=context_texts)
                elif metric.name == "answer_relevancy":
                    res = await metric.ascore(user_input=query, response=gen_result.answer)
                elif metric.name == "context_precision":
                    res = await metric.ascore(user_input=query, retrieved_contexts=context_texts, reference=ground_truth)
                else:
                    res = await metric.ascore(user_input=query, response=gen_result.answer, retrieved_contexts=context_texts)
                
                score_val = res.value if hasattr(res, "value") else res
                scores[metric.name] = round(float(score_val), 4) if score_val is not None else None

                # 10-second breather between each metric calculation
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Metric {metric.name} failed: {e}")
                scores[metric.name] = None

        total_latency = time.perf_counter() - t0

        # Combine System Observability + RAGAS Semantic metrics
        return {
            "query": query,
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision"),
            "is_grounded_flag": gen_result.is_grounded,
            "retrieval_latency_sec": round(retrieval_latency, 3),
            "generation_latency_sec": round(generation_latency, 3),
            "total_latency_sec": round(total_latency, 3)
        }

async def run_benchmark():
    golden_test_set = [
        {"query": "What factors drove NVIDIA's data center revenue growth?", "expected": "Data Center growth was driven by demand for GPU computing platforms and Cloud Service Provider agreements."},
        {"query": "What are the primary operational risk factors mentioned in Apple's filings?", "expected": "Supply chain dependencies, global market competition, and international trade regulations."}
    ]

    evaluator = ProductionRAGEvaluator()
    results = []
    
    for item in golden_test_set:
        res = await evaluator.evaluate_query(item)
        results.append(res)
        print(f"✓ Evaluated: '{item['query'][:30]}...' -> Faithfulness: {res['faithfulness']}")

    df = pd.DataFrame(results)
    df.to_csv("rag_eval_report.csv", index=False)
    print("\n✅ Evaluation complete. Report saved to 'rag_eval_report.csv'.")
    print(df.mean(numeric_only=True).to_string())

if __name__ == "__main__":
    asyncio.run(run_benchmark())