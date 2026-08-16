"""
NVIDIA LLM client — OpenAI-compatible chat completions.

NVIDIA integrate.api.nvidia.com, OpenAI SDK ile uyumlu bir chat API sunar.
Bu modül yargi-mcp'nin LocalEmbedder mantığına benzer şekilde, NVIDIA LLM'leri
için sade bir sarmalayıcı (wrapper) sağlar.

Desteklenen modeller (Türkçe için önerilenler):
    - nvidia/llama-3.1-nemotron-70b-instruct  (önerilen — Türkçe güçlü)
    - meta/llama-3.1-70b-instruct              (alternatif)
    - nvidia/nemotron-4-340b-instruct           (en kaliteli, yavaş)
    - mistralai/mistral-large-2-instruct        (alternatif)

Çevre değişkenleri:
    NVIDIA_API_KEY             (zorunlu) — build.nvidia.com'dan alınır
    NVIDIA_LLM_BASE_URL        (opsiyonel) — default: https://integrate.api.nvidia.com/v1
    NVIDIA_LLM_MODEL           (opsiyonel) — default: nvidia/llama-3.1-nemotron-70b-instruct
    NVIDIA_LLM_TEMPERATURE     (opsiyonel) — default: 0.2 (hukuki: düşük yaratıcılık)
    NVIDIA_LLM_MAX_TOKENS      (opsiyonel) — default: 1500
    NVIDIA_LLM_TIMEOUT         (opsiyonel) — default: 60 saniye
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"  # NVIDIA hesabında test edilmiş, Türkçe iyi
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TIMEOUT = 90  # NVIDIA 70B ilk token bazen yavaş


# NVIDIA hesabında doğrulanmış modeller (8b/70b çalışıyor; 405b, nemotron-70b, mixtral-8x22b HESAPTA YOK/404)
# Daha iyi model bulunursa NVIDIA_LLM_MODEL env var ile override edilebilir.
VERIFIED_MODELS = [
    "meta/llama-3.1-70b-instruct",      # Önerilen — Türkçe güçlü, 70B
    "meta/llama-3.1-8b-instruct",       # Hızlı alternatif — Türkçe yeterli
]


@dataclass
class LLMResponse:
    """LLM yanıtının structured temsili."""
    text: str
    model: str
    usage: Dict[str, int]
    raw: object  # OpenAI SDK raw response


class NvidiaLLMClient:
    """
    NVIDIA integrate API için OpenAI-compatible chat client.

    Sync ve async desteği vardır. Async stream legal RAG demosu için kullanılır.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        # Config'i env var'lardan veya parametre'den al
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("LOCAL_EMBEDDING_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key gerekli. NVIDIA_API_KEY env var'ını set edin "
                "ya da constructor'a api_key parametresi verin."
            )

        self.base_url = base_url or os.environ.get("NVIDIA_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("NVIDIA_LLM_MODEL", DEFAULT_MODEL)
        self.temperature = float(temperature if temperature is not None
                                 else os.environ.get("NVIDIA_LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
        self.max_tokens = int(max_tokens if max_tokens is not None
                              else os.environ.get("NVIDIA_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        self.timeout = int(timeout if timeout is not None
                           else os.environ.get("NVIDIA_LLM_TIMEOUT", DEFAULT_TIMEOUT))

        # OpenAI SDK (LocalEmbedder ile aynı paket — zaten bağımlılık var)
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package gerekli. pip install openai ile kurun."
            ) from e

        # Sync ve async client'lar
        self._sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            f"NvidiaLLMClient hazır — model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Sync chat completion."""
        resp = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            raw=resp,
        )

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Async chat completion (non-streaming)."""
        resp = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            raw=resp,
        )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming chat completion — token token yield eder.

        CLI/REPL'de anlık yazdırma ve FastAPI SSE'de kullanılır.
        """
        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
