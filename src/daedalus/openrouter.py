from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parents[2] / ".env")

MAX_TOKENS_GENERATE = 4096
MAX_TOKENS_VALIDATE = 512

# Validation uses a different model than generation to avoid self-grading circularity.
# DeepSeek-V4-Pro scores Llama-generated variants; neither judges its own output.
VALIDATOR_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

_PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()

# Auto-select: prefer Together if key present and no explicit override to openrouter.
if not _PROVIDER:
    _PROVIDER = "together" if os.environ.get("TOGETHER_KEY") else "openrouter"

if _PROVIDER == "together":
    MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # Together AI doesn't use OpenRouter provider-routing headers; pass empty dict so
    # callers that do extra_body=PROVIDER_ROUTING are safe.
    PROVIDER_ROUTING: dict = {}
else:
    MODEL = "anthropic/claude-sonnet-4-5"
    # Routes via direct Anthropic rather than Amazon Bedrock; Bedrock has tighter
    # per-minute token quotas that cause 402 errors under batch generation loads.
    PROVIDER_ROUTING = {"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}


def get_client() -> OpenAI:
    if _PROVIDER == "together":
        key = os.environ.get("TOGETHER_KEY")
        if not key:
            raise RuntimeError("TOGETHER_KEY not set. Add it to .env")
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=key)

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Add it to .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def get_client_for_model(model: str) -> tuple[OpenAI, dict]:
    """Return (client, provider_routing) for the given model string.

    Routing rules, in priority order:
    1. anthropic/ prefix → always OpenRouter with direct Anthropic provider routing.
    2. LLM_PROVIDER=openrouter set explicitly → OpenRouter for all other models.
    3. TOGETHER_KEY present → Together AI (handles meta-llama/, deepseek-ai/, etc.).
    4. Fallback → OpenRouter via OPENROUTER_API_KEY.
    """
    if model.startswith("anthropic/"):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set — required for anthropic/ models")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        routing = {"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}
        return client, routing

    # Respect explicit provider override before checking keys.
    if _PROVIDER == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set (LLM_PROVIDER=openrouter requires it)")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key), {}

    together_key = os.environ.get("TOGETHER_KEY")
    if together_key:
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=together_key), {}

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key), {}

    raise RuntimeError("No API key available — set TOGETHER_KEY or OPENROUTER_API_KEY in .env")
