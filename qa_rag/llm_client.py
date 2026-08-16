"""
Multi-provider LLM client — OpenAI-compatible chat completions.

v1.3.0+: Artık tek NVIDIA değil, 4 provider destekleniyor:
    - nvidia (default): integrate.api.nvidia.com — kaliteli ama yavaş (60-240s)
    - groq:            groq.com — ÇOK HIZLI (~500 tok/s), ücretsiz tier yeterli
    - openai:          openai.com — gpt-4o-mini hızlı ve ucuz
    - ollama:          localhost — tamamen local, hiç API key yok

Tüm provider'lar OpenAI-compatible API sunduğu için aynı SDK kullanılır.
Sadece base_url + api_key + model değişir.

Çevre değişkenleri:
    LLM_PROVIDER              (opsiyonel) — nvidia|groq|openai|ollama (default: nvidia)
    LLM_API_KEY               (opsiyonel) — seçili provider'ın key'i
    LLM_BASE_URL              (opsiyonel) — provider base URL (default provider'a göre)
    LLM_MODEL                 (opsiyonel) — model adı (default provider'a göre)
    LLM_TEMPERATURE           (opsiyonel) — default: 0.2 (hukuki: düşük yaratıcılık)
    LLM_MAX_TOKENS            (opsiyonel) — default: 1500
    LLM_TIMEOUT               (opsiyonel) — default: 90s (NVIDIA için), Groq için 30s

Eski env var'lar hâlâ destekleniyor (backward compat):
    NVIDIA_API_KEY, NVIDIA_LLM_BASE_URL, NVIDIA_LLM_MODEL, NVIDIA_LLM_TEMPERATURE,
    NVIDIA_LLM_MAX_TOKENS, NVIDIA_LLM_TIMEOUT

Provider default'ları:
    nvidia: base=integrate.api.nvidia.com/v1, model=meta/llama-3.1-70b-instruct
    groq:   base=groq.com/openai/v1,         model=llama-3.3-70b-versatile
    openai: base=api.openai.com/v1,           model=gpt-4o-mini
    ollama: base=localhost:11434/v1,          model=llama3.1:8b
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-70b-instruct",
        "timeout": 90,  # NVIDIA 70B ilk token bazen çok yavaş
        "api_key_env": "NVIDIA_API_KEY",  # fallback env var
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "timeout": 30,
        "api_key_env": "GROQ_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "timeout": 30,
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b",
        "timeout": 60,
        "api_key_env": "OLLAMA_API_KEY",  # Ollama key istemez ama boş string yeterli
    },
}

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1500

# NVIDIA hesabında doğrulanmış modeller (8b/70b çalışıyor; 405b, nemotron-70b, mixtral-8x22b HESAPTA YOK/404)
# Daha iyi model bulunursa NVIDIA_LLM_MODEL env var ile override edilebilir.
VERIFIED_MODELS = [
    "meta/llama-3.1-70b-instruct",      # NVIDIA — Türkçe güçlü, 70B
    "meta/llama-3.1-8b-instruct",       # NVIDIA — Hızlı alternatif
    "llama-3.3-70b-versatile",          # Groq — HIZLI (500 tok/s)
    "llama-3.1-8b-instant",             # Groq — Çok hızlı, daha basit
    "gpt-4o-mini",                      # OpenAI — ucuz, hızlı
    "llama3.1:8b",                      # Ollama — local
]


@dataclass
class LLMResponse:
    """LLM yanıtının structured temsili."""
    text: str
    model: str
    usage: Dict[str, int]
    raw: object  # OpenAI SDK raw response
    provider: str = ""  # v1.3.0+: hangi provider üretti
    from_cache: bool = False  # v1.3.0+: cache'den mi geldi


def _resolve_provider(provider: Optional[str] = None) -> str:
    """Provider adını çöz — env veya parametre."""
    p = (provider or os.environ.get("LLM_PROVIDER") or "nvidia").lower()
    if p not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Geçersiz LLM_PROVIDER='{p}'. Geçerli değerler: {list(PROVIDER_DEFAULTS.keys())}"
        )
    return p


def _resolve_config(
    provider: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    timeout: Optional[int],
):
    """Provider'a göre config'i çöz. Verilen parametreler env'i override eder."""
    defaults = PROVIDER_DEFAULTS[provider]

    # API key: parametre > LLM_API_KEY > provider-specific env
    resolved_key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get(defaults["api_key_env"])
    )
    # Ollama key istemez — boş string yeterli
    if provider == "ollama" and not resolved_key:
        resolved_key = "ollama"

    # Base URL: parametre > LLM_BASE_URL > provider default
    resolved_base = (
        base_url
        or os.environ.get("LLM_BASE_URL")
        or defaults["base_url"]
    )

    # Model: parametre > LLM_MODEL > provider default (NVIDIA için eski env var)
    if model:
        resolved_model = model
    elif os.environ.get("LLM_MODEL"):
        resolved_model = os.environ["LLM_MODEL"]
    elif provider == "nvidia" and os.environ.get("NVIDIA_LLM_MODEL"):
        resolved_model = os.environ["NVIDIA_LLM_MODEL"]
    else:
        resolved_model = defaults["model"]

    # Timeout: parametre > LLM_TIMEOUT > provider default (NVIDIA için eski env var)
    if timeout is not None:
        resolved_timeout = timeout
    elif os.environ.get("LLM_TIMEOUT"):
        resolved_timeout = int(os.environ["LLM_TIMEOUT"])
    elif provider == "nvidia" and os.environ.get("NVIDIA_LLM_TIMEOUT"):
        resolved_timeout = int(os.environ["NVIDIA_LLM_TIMEOUT"])
    else:
        resolved_timeout = defaults["timeout"]

    return resolved_key, resolved_base, resolved_model, resolved_timeout


# ---------------------------------------------------------------------------
# Main client — multi-provider
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Multi-provider LLM client (OpenAI-compatible).

    v1.3.0+: NvidiaLLMClient'in yerini aldı. Aynı API'yi sağlar, ek olarak
    provider seçimi destekler.

    Sync + async + streaming desteği vardır.

    Usage:
        client = LLMClient()                      # LLM_PROVIDER env'den
        client = LLMClient(provider="groq")       # explicit
        resp = client.chat([{"role": "user", "content": "..."}])
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.provider = _resolve_provider(provider)
        (
            self.api_key,
            self.base_url,
            self.model,
            self.timeout,
        ) = _resolve_config(self.provider, api_key, base_url, model, timeout)

        if not self.api_key:
            raise ValueError(
                f"LLM API key gerekli. LLM_API_KEY veya "
                f"{PROVIDER_DEFAULTS[self.provider]['api_key_env']} env var'ını set edin."
            )

        # Temperature / max_tokens — eski NVIDIA_LLM_* env var'ları destekleniyor
        self.temperature = float(
            temperature if temperature is not None
            else os.environ.get("LLM_TEMPERATURE")
            or os.environ.get("NVIDIA_LLM_TEMPERATURE")
            or DEFAULT_TEMPERATURE
        )
        self.max_tokens = int(
            max_tokens if max_tokens is not None
            else os.environ.get("LLM_MAX_TOKENS")
            or os.environ.get("NVIDIA_LLM_MAX_TOKENS")
            or DEFAULT_MAX_TOKENS
        )

        # OpenAI SDK (tüm provider'lar OpenAI-compatible)
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package gerekli. pip install openai ile kurun."
            ) from e

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
            f"LLMClient hazır — provider={self.provider}, model={self.model}, "
            f"base_url={self.base_url}, timeout={self.timeout}s, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

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
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            },
            raw=resp,
            provider=self.provider,
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

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
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            },
            raw=resp,
            provider=self.provider,
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
        Groq ~500 tok/s verir, NVIDIA ~5-10 tok/s.
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


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------


class NvidiaLLMClient(LLMClient):
    """
    Backward-compat alias — v1.1.0/v1.2.0 kodunu kırmamak için.

    Yeni kod LLMClient kullanmalı. Bu sınıf sadece provider='nvidia' ile
    LLMClient çağırmakla eşdeğer.
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
        # Eski NVIDIA_API_KEY env var'ını LLM_API_KEY'e bridge et
        if api_key and not os.environ.get("LLM_API_KEY"):
            os.environ["LLM_API_KEY"] = api_key

        super().__init__(
            provider="nvidia",
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


def get_llm_client(**kwargs) -> LLMClient:
    """
    Factory — env var'lara göre uygun LLMClient oluştur.

    Default: LLM_PROVIDER env var'ı, yoksa 'nvidia'.
    """
    return LLMClient(**kwargs)
