# """
# Grounded Financial Response Generation Engine

# This module serves as the final synthesis stage of an enterprise RAG pipeline, generating
# strictly grounded answers from retrieved SEC 10-K document chunks. Key capabilities:
# 1. Context Formatting: Compiles and structures retrieved chunk objects into a clean, 
#    labeled context string containing document IDs, ticker symbols, and excerpts.
# 2. Strict Grounding Prompts: Guides the LLM to act as a financial research assistant, 
#    forbidding extrapolation, speculation, or hallucination of figures not in the context.
# 3. Structured Output & Attribution: Uses Pydantic schemas via OpenAI-compatible structured 
#    outputs (`beta.chat.completions.parse`) to return typed answers, grounding flags (`is_grounded`), 
#    and source citation tags.
# 4. Non-Blocking Async Execution: Built on `AsyncOpenAI` for non-blocking I/O in concurrent 
#    production servers (e.g., FastAPI), using temperature 0.0 for deterministic financial outputs.
# """
# ========================================================================================================
import os
import logging
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

#-------------Structured Output Schema-------------------
class GeneratedResponse(BaseModel):
    answer: str = Field(
        description="The final answered grounded strictly in the provided SEC 10-K snippets."
    )
    is_grounded: bool = Field(
        description="True if the answer was completely derived from the text. False if the context lacked sufficient information."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="A list of specific source identifiers (e.g., 'Document 1 | AAPL') used to generate the answer."
    )

class RAGGenerator:
    """
    Asynchronous, production-grade response generation engine.
    Utilizes Gemini's structured outputs and async APIs to prevent server blocking.
    """
    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
        request_timeout: float = 25.0
    ):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            logger.warning("⚠️ GEMINI_API_KEY missing from environment setup.")
        
        # Fully asynchronous client
        self.client = AsyncOpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=3,
            timeout=request_timeout,
        )
        self.model = model_name

    async def a_generate_grounded_answer(
        self,
        query: str,
        retrieved_context: List[Any] # Accepts the RetrievedChunk objects from retrieve.py
    ) -> Tuple[GeneratedResponse, List[Any]]:
        """
        Asynchronously generates an answer strictly grounded in the retrieved chunks.
        Returns a structured Pydantic object and the original context array.
        """

        # Format the retrieved objects into a readable context block
        context_blocks = []
        for i, chunk in enumerate(retrieved_context, start=1):
            # Access attributes of the RetrievedChunk object
            ticker = getattr(chunk, 'ticker', 'UNKNOWN')
            content = getattr(chunk, 'content', '')
            source_id = getattr(chunk, 'source_doc_id', 'UNKNOWN')

            context_blocks.append(f"[Document {i} | Ticker: {ticker} | DocID: {source_id}]\n{content}")

        formatted_context = "\n\n---\n\n".join(context_blocks)

        if not formatted_context.strip():
            return GeneratedResponse(
                answer="No relevant financial context could be retrieved to securely answer this query.",
                is_grounded=False,
                citations=[]
            ), retrieved_context

        # Construct the strict enterprise prompt
        system_prompt = (
            "You are an elite enterprise financial research assistant analyzing SEC 10-K reports.\n"
            "Your objective is to answer the user's query strictly and accurately based ONLY on the provided excerpts.\n"
            "STRICT RULES:\n"
            "1. Ground all claims in the context. If the context does not contain enough information to answer definitively, set is_grounded to false and state what is missing.\n"
            "2. Never extrapolate financial trends or make up numbers outside the provided text.\n"
            "3. Maintain a concise, objective financial reporting tone."
        )
        
        user_content = f"Context Excerpts:\n{formatted_context}\n\nUser Question: {query}"

        # Asynchronous Inference with Structured Output
        try:
            logger.info(f"Generating grounded response for query: '{query}'...")
            # We use beta.chat.completions.parse for structured output
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=GeneratedResponse,
                temperature=0.0 # Strict determinism for financial data
            )

            parsed_response: GeneratedResponse = response.choices[0].message.parsed

            return parsed_response, retrieved_context

        except Exception as e:
            logger.error(f"❌ Generation pipeline execution failure: {e}")
            # Graceful fallback object
            return GeneratedResponse(
                answer="An error occurred while synthesizing the final response. Please try again.",
                is_grounded=False,
                citations=[]
            ), retrieved_context