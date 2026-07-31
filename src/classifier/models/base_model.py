"""Base class for all LLM model wrappers."""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Shared HTTP client — reused across all models for connection pooling.
_shared_client: Optional[httpx.AsyncClient] = None


async def get_shared_client() -> httpx.AsyncClient:
    """Return (and lazily create) a shared async HTTP client."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=10.0),
            limits=httpx.Limits(max_connections=60, max_keepalive_connections=30),
        )
    return _shared_client


class BaseModel(ABC):
    """Abstract base class for LLM model implementations.

    All models must implement ``classify_content`` which sends text to
    the model and returns a structured dict with at least
    ``classification``, ``confidence``, and optionally
    ``predicted_distribution``.
    """

    def __init__(self, name: str, model_id: str, api_key_env: str):
        self.name = name
        self.model_id = model_id
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} not found in environment")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(prompt: str, text: str):
        """Build a chat-completion message list (system + user)."""
        return [
            {"role": "system", "content": "You are an expert marker of undergraduate psychology essays in the UK."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": text},
                ],
            },
        ]

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Extract a JSON object from the model's raw text output."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        # If no JSON object at all, return raw content
        if "{" not in text:
            return {"content": text}

        # Extract the outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to fix common issues: trailing commas, unescaped newlines in strings
            import re
            cleaned = re.sub(r',\s*}', '}', text)  # trailing commas
            cleaned = re.sub(r',\s*]', ']', cleaned)  # trailing commas in arrays
            # Replace literal newlines inside string values
            cleaned = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace('\n', '\\n'), cleaned, flags=re.DOTALL)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"error": f"Failed to parse model response as JSON", "raw_response": raw[:200]}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def classify_content(
        self, data: Dict[str, str], prompt: str
    ) -> Dict[str, Any]:
        """Send *data* to the model with *prompt* and return structured results.

        Args:
            data: ``{"type": "text", "content": "<essay text>"}``
            prompt: The fully-rendered prompt string.

        Returns:
            Dict with ``classification``, ``confidence``,
            ``predicted_distribution`` on success, or ``error`` on failure.
        """
