"""LLM adapters. EchoLLM lets the whole pipeline run with zero API calls."""
from __future__ import annotations

from typing import Optional


class EchoLLM:
    """Zero-infra synthesizer: returns the assembled evidence as the 'answer'.

    Not a real LLM — it proves the retrieval/fusion pipeline end-to-end without
    any API key. Swap in any object with ``generate(prompt, system=, timeout=)``
    (OpenAI, vLLM, or any HTTP LLM) for real synthesis.
    """

    def generate(self, prompt: str, *, system: str = "", timeout: Optional[float] = None) -> str:
        # The prompt already contains the fused evidence/relations/class-seed.
        body = prompt.split("Question:", 1)[-1].strip()
        return "[EchoLLM — inject a real LLM for synthesis] fused evidence:\n" + body[:1500]


class CallableLLM:
    """Wrap any ``fn(prompt, system) -> str`` as an LLM."""

    def __init__(self, fn):
        self._fn = fn

    def generate(self, prompt: str, *, system: str = "", timeout: Optional[float] = None) -> str:
        return self._fn(prompt, system)
