import os
import time
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# ............Retrieval & Ranking Metrics...............

def calculate_precision_at_k(relevance_scores: List[int], k: int = 5) -> float:
    """
    Calculates Precision@K.
    Args:
        relevance_scores: Binary relevance list (1 for relevant, 0 for irrelevant).
        k: Cutoff rank.
    """
    if k <= 0 or not relevance_scores:
        return 0.0
    top_k = relevance_scores[:k]
    return float(sum(1 for score in top_k if score > 0) / k)

def calculate_recall_at_k(relevance_scores: List[int], total_relevant: int, k: int = 5) -> float:
    """
    Calculates Recall@K.
    Args:
        relevance_scores: Binary relevance list (1 for relevant, 0 for irrelevant).
        total_relevant: Total known ground-truth relevant documents for this query.
        k: Cutoff rank.
    """
    if total_relevant <= 0 or not relevance_scores:
        return 0.0
    top_k = relevance_scores[:k]
    relevant_retrieved = sum(1 for score in top_k if score > 0)
    return float(relevant_retrieved / total_relevant)

def calculate_mrr(rankings: List[int]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR).
    Args:
        rankings: List of 1-indexed ranks for the first relevant document.
                  If not found, use 0.
    """
    if not rankings:
        return 0.0
    reciprocal_ranks = [1.0 / r if r > 0 else 0.0 for r in rankings]
    return float(np.mean(reciprocal_ranks))

def calculate_ndcg(relevance_scores: List[float], k: int = 5) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG@k).
    Args:
        relevance_scores: Relevance values in retrieval order.
        k: Depth to evaluate.
    """
    scores = relevance_scores[:k]
    if not scores:
        return 0.0
    
    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores))
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))

    return float(dcg / idcg) if idcg > 0 else 0.0 

# ................ Generation Lexical Metric .....................

def calculate_keyword_coverage(generated_text: str, expected_keywords: List[str]) -> float:
    """
    Calculates the percentage of expected ground-truth keywords in the response.
    """
    if not expected_keywords:
        return 0.0
    text_lower = generated_text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return float(found / len(expected_keywords))

# ............... LLM-as-a-Judge Evaluation .......................

class RAGJudgeVerdict(BaseModel):
    faithfulness_score: int = Field(
        ge=1, le=5,
        description="1-5 rating: Is the answer strictly grounded in the provided context without hallucinations?"
    )
    relevancy_score: int = Field(
        ge=1, le=5,
        description="1-5 rating: Does the answer directly and completely answer the query?"
    )
    faithfulness_critique: str = Field(
        description="Brief explanation highlighting any ungrounded assertions or confirming strict faithfulness."
    )
    relevancy_critique: str = Field(
        description="Brief explanation of how well the answer satisfies the user's query."
    )
    passed: bool = Field(
        description="True if faithfulness >= 4 and relevancy >= 4."
    )

@retry(
    wait=wait_random_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError))
)
def evaluate_rag_triad_with_llm(
    client: OpenAI,
    model: str,
    query: str,
    retrieved_context: str,
    generated_answer: str,
    ground_truth: Optional[str] = None,
    timeout: float = 30.0,
    rate_limit_delay: float = 1.0
) -> RAGJudgeVerdict:
    """
    Evaluates response quality using LLM-as-a-Judge with structured output.
    Includes timeouts and automatic retry handling for Gemini free-tier rate limits.
    """
    system_prompt = (
        "You are an impartial, strict RAG evaluation judge. "
        "Evaluate the quality of the generated answer based ONLY on the provided query and retrieved context.\n\n"
        "Scoring rubric (1 to 5):\n"
        "1: Completely hallucinates or fails to address query.\n"
        "3: Partially grounded or partially answers the query.\n"
        "5: Fully grounded in the context with zero hallucinations, perfectly answers query."
    )

    user_content = f"""
[USER QUERY]: {query}

[RETRIEVED CONTEXT]:
{retrieved_context}

[GENERATED ANSWER]:
{generated_answer}
"""
    if ground_truth:
        user_content += f"\n[GROUND TRUTH REFERENCE]:\n{ground_truth}"

    # Rate limiting buffer before request
    if rate_limit_delay > 0:
        time.sleep(rate_limit_delay)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        response_format=RAGJudgeVerdict,
        timeout=timeout
    )
    
    return response.choices[0].message.parsed