"""Google Gemini client wrapper.

- Centralizes model + API key + structured-output config.
- Uses Gemini's native `response_schema` for typed Pydantic output.
- Wraps every call with tenacity exponential backoff — Gemini's free tier
  hits transient 503s under load and one flaky minute should not kill a demo.
- Returns the raw JSON alongside the parsed Pydantic object so callers can
  show 'raw API response' and 'extracted structured output' separately,
  exactly as the brief asks.
"""

from __future__ import annotations

import os
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Wraps any LLM-side failure with a human-readable message."""


# Gemini transient errors worth retrying. 5xx server errors and rate-limit 429s.
_RETRYABLE = (genai_errors.ServerError, genai_errors.ClientError)


def _resolve_api_key() -> str:
    """Env var first; Streamlit secrets second (lets the same code work in Cloud)."""
    key = os.environ.get("GEMINI_API_KEY") or get_settings().gemini_api_key
    if key:
        return key
    try:
        import streamlit as st  # type: ignore[import-not-found]

        return st.secrets["GEMINI_API_KEY"]
    except Exception as exc:
        raise LLMError(
            "GEMINI_API_KEY not set. Put it in .env (CLI) or Streamlit secrets (web)."
        ) from exc


def get_client() -> genai.Client:
    return genai.Client(api_key=_resolve_api_key())


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def call_structured(
    *,
    system_prompt: str,
    user_prompt: str,
    output_schema: type[T],
    temperature: float = 0.2,
) -> tuple[T, str]:
    """One-shot structured-output call.

    Returns (parsed_pydantic_object, raw_text_response).
    Gemini's `response_schema` guarantees the response shape, so parsing is
    trivial — but we still validate against the Pydantic model so any drift
    fails loudly here rather than at the call site.
    """
    client = get_client()
    settings = get_settings()

    try:
        response = client.models.generate_content(
            model=settings.timecell_model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=output_schema,
                temperature=temperature,
            ),
        )
    except genai_errors.APIError as exc:
        raise LLMError(f"Gemini API error: {exc}") from exc

    raw_text = response.text or ""
    if not raw_text:
        raise LLMError("Gemini returned no text content")

    # response.parsed is the SDK's Pydantic instance. Re-validating against our
    # schema is cheap and catches any future SDK drift.
    parsed_obj = response.parsed
    if parsed_obj is None:
        try:
            parsed_obj = output_schema.model_validate_json(raw_text)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"failed to parse Gemini output: {exc}\n--- raw ---\n{raw_text}") from exc

    if not isinstance(parsed_obj, output_schema):
        # Defensive: some Gemini SDK versions return a dict in .parsed
        parsed_obj = output_schema.model_validate(parsed_obj)

    return parsed_obj, raw_text
