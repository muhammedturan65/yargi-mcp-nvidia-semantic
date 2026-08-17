"""
qa_rag — Hukuki QA Chatbot (RAG pipeline)

yargi-mcp semantik arama + multi-provider LLM ile Türk hukuki sorulara
atatf referanslı cevaplar üretir.

v1.4.0: Query embedding cache (LRU + ChromaDB persistent) — NVIDIA ~1s/sorgu
        çağrısını önler, process restart'ında kalıcı.
v1.3.0: Multi-provider LLM (NVIDIA/Groq/OpenAI/Ollama) + semantik answer cache.
v1.2.0: ChromaDB kalıcı vector store + token-aware chunker + BedestenIndexer.

Kullanım:
    from qa_rag import LegalQARAG
    rag = LegalQARAG()                       # chroma backend (default)
    await rag.load_corpora()                 # ilk sefer index (5-10 dk)
    answer = rag.ask("Muvazaalı tapu satışında mirasçı hangi davayı açar?")

Hızlı LLM için (önerilen):
    export LLM_PROVIDER=groq
    export GROQ_API_KEY=gq_...
    # 500 tok/s — NVIDIA 60-240s yerine 2-5s

Cache kontrolü:
    export RAG_ANSWER_CACHE=true             # default
    export RAG_CACHE_THRESHOLD=0.92          # cosine threshold
    export RAG_QUERY_CACHE=true              # default (v1.4.0+)
    export RAG_QUERY_CACHE_LRU_SIZE=256      # in-memory LRU boyutu
"""

from .rag_engine import LegalQARAG, RAGResponse, RAGContext
from .llm_client import LLMClient, NvidiaLLMClient, get_llm_client, LLMResponse
from .answer_cache import AnswerCache, CacheHit
from .query_cache import QueryEmbeddingCache, QueryCacheHit, normalize_query, cache_key
from .prompts import SYSTEM_PROMPT_LEGAL, build_user_prompt
from .citations import format_citations, Citation
from .chunker import LegalChunker, Chunk, chunk_text
from .indexer import BedestenIndexer, IndexResult, IndexProgress

__version__ = "1.5.0"
__all__ = [
    "LegalQARAG",
    "RAGResponse",
    "RAGContext",
    "LLMClient",
    "NvidiaLLMClient",  # backward compat
    "get_llm_client",
    "LLMResponse",
    "AnswerCache",
    "CacheHit",
    "QueryEmbeddingCache",
    "QueryCacheHit",
    "normalize_query",
    "cache_key",
    "SYSTEM_PROMPT_LEGAL",
    "build_user_prompt",
    "format_citations",
    "Citation",
    "LegalChunker",
    "Chunk",
    "chunk_text",
    "BedestenIndexer",
    "IndexResult",
    "IndexProgress",
]
