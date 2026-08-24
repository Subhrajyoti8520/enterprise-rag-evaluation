# """
# Conversational Session Manager & Query Reformulation Module

# This script manages multi-turn conversation states and optimizes incoming user 
# queries for a financial RAG system. It performs the following key functions:
# 1. Session Memory Management: Stores conversational turns per session using an in-memory 
#    sliding window mechanism to bound memory growth and prevent context bloat.
# 2. Async OpenAI/Gemini Integration: Uses OpenAI's asynchronous client configured with 
#    Google Gemini's OpenAI-compatible endpoint for non-blocking I/O operations.
# 3. Coreference Resolution & Entity Extraction: Rewrites context-dependent user queries 
#    (e.g., "What was its revenue?") into self-contained search queries (e.g., "What was 
#    Apple's 2023 revenue?") while extracting the relevant stock ticker.
# 4. Structured Outputs: Leverages Pydantic with the Structured Outputs API (`client.beta.chat.completions.parse`) 
#    to guarantee strictly typed extraction of the rewritten query and detected ticker.
# """
# ===============================================================================================================
import os
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------- Structured Output Schema for Query Reformulation ---------------
class ReformulatedQuery(BaseModel):
    standalone_query: str = Field(
        description="The fully contextualized, standalone search query resolving all pronouns and implicit references."
    )
    detected_ticker: Optional[str] = Field(
        default=None,
        description="Uppercase stock ticker (e.g., AAPL, NVDA, MSFT) if mentioned or inferred from history, else None."
    )

# ---------- Chat Session Manager -------------
class ChatSessionManager:
    """
    Production-grade conversational session store with:
    - Non-blocking asynchronous LLM orchestration (AsyncOpenAI)
    - Structured Pydantic query rewriting & ticker extraction
    - Bounded memory sliding windows (prevents memory leaks)
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        max_history_turns: int = 10,
        request_timeout: float = 30.0,
    ):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment variables.")
        
        self.client = AsyncOpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=3,
            timeout=request_timeout,
        )
        self.model = model_name
        self.max_history_turns = max_history_turns
        self.request_timeout = request_timeout
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves conversation history for a given session."""
        return self.sessions.setdefault(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        """
        Appends a message to the session history.
        Enforces a sliding window to keep memory footprint bounded.
        """
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": role, "content": content})

        # Keep only the most recent N turns (2 * max_history_turns messages)
        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            self.sessions[session_id] = history[-max_messages:]

    def clear_session(self, session_id: str):
        """Cleans up session history when conversations end."""
        self.sessions.pop(session_id, None)

    async def a_rewrite_query(self, session_id: str, latest_query: str) -> ReformulatedQuery:
        """
        Asynchronously resolves coreferences and extracts the target ticker
        using Gemini structured outputs.
        """
        history = self.get_history(session_id)
        if not history:
            return ReformulatedQuery(standalone_query=latest_query, detected_ticker=None)
        
        # Use the last 4 messages(2 turns) for focused coreference context
        recent_context = history[-4:]
        history_formatted = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_context]
        )

        system_prompt = (
            "You are an expert financial search query optimizer.\n"
            "Analyze the conversation history and the latest user query:\n"
            "1. Rewrite the query into a standalone, search-ready financial question resolving all pronouns (it, they, its, their).\n"
            "2. Identify any company ticker mentioned or implied from conversation history (e.g., AAPL, MSFT, TSLA, NVDA).\n"
            "If no entity exists, set detected_ticker to null."
        )

        user_content = (
            f"Conversation History:\n{history_formatted}\n\n"
            f"Latest Query: {latest_query}"
        )

        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=ReformulatedQuery,
                temperature=0.0,
                timeout=self.request_timeout
            )
            parsed: ReformulatedQuery = response.choices[0].message.parsed
            logger.info(f"🔄 Rewrote: '{latest_query}' -> '{parsed.standalone_query}' (Ticker: {parsed.detected_ticker})")
            return parsed

        except Exception as e:
            logger.error(f"❌ Coreference resolution failed: {e}. Falling back to original query.")
            return ReformulatedQuery(standalone_query=latest_query, detected_ticker=None)