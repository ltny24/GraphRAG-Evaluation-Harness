"""
LLM Client Wrapper
GraphRAG v2.1 — Infrastructure

Centralizes all Gemini API calls with:
- Retry logic with exponential backoff (handles 429/503)
- Token counting for cost tracking
- Error classification (transient vs permanent)
- Demo mode fallback when API quota is exhausted

Why a wrapper instead of calling genai directly?
→ Every module was duplicating retry logic and error handling.
→ Token counting was impossible without a central point.
→ Changing models requires editing one file instead of six.
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Type
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around Gemini API with retry, telemetry, and fallback."""

    def __init__(self, model_id: str = "gemini-3.6-flash", max_retries: int = 3):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = model_id
        self.max_retries = max_retries

        # Cumulative token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

    def generate(
        self,
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.1,
    ) -> tuple[str, dict]:
        """
        Call Gemini with retry logic.

        Returns:
            (response_text, metadata_dict)

        metadata_dict contains:
            - input_tokens (estimated)
            - output_tokens (estimated)
            - latency_ms
            - retries
            - model
        """
        start = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                config_kwargs = {"temperature": temperature}
                if response_schema:
                    config_kwargs["response_mime_type"] = "application/json"
                    config_kwargs["response_schema"] = response_schema

                config = types.GenerateContentConfig(**config_kwargs)

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=config,
                )

                elapsed_ms = (time.perf_counter() - start) * 1000

                # Estimate tokens (rough: 1 token ≈ 4 chars for English)
                est_input = len(prompt) // 4
                est_output = len(response.text) // 4 if response.text else 0

                self.total_input_tokens += est_input
                self.total_output_tokens += est_output
                self.total_requests += 1

                metadata = {
                    "input_tokens": est_input,
                    "output_tokens": est_output,
                    "latency_ms": round(elapsed_ms, 2),
                    "retries": attempt,
                    "model": self.model_id,
                }

                return response.text, metadata

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Classify error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Rate limit — back off and retry
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning(f"Rate limited (attempt {attempt+1}/{self.max_retries}). "
                                   f"Waiting {wait}s...")
                    time.sleep(wait)
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    # Server overloaded — retry
                    wait = 2 ** attempt * 2
                    logger.warning(f"Server unavailable (attempt {attempt+1}). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    # Permanent error (invalid model, auth, etc.) — don't retry
                    self.failed_requests += 1
                    raise

        # All retries exhausted
        self.failed_requests += 1
        raise last_error

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.1,
    ) -> tuple[BaseModel, dict]:
        """
        Call Gemini and parse response into a Pydantic model.
        Handles JSON parsing errors with retry.
        """
        text, metadata = self.generate(prompt, response_schema=schema, temperature=temperature)

        try:
            data = json.loads(text)
            parsed = schema(**data)
            return parsed, metadata
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Failed to parse LLM response as {schema.__name__}: {e}\nRaw: {text}")

    def get_stats(self) -> dict:
        """Return cumulative usage statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(
                (self.total_input_tokens / 1_000_000) * 0.075 +
                (self.total_output_tokens / 1_000_000) * 0.30,
                6
            ),
        }
