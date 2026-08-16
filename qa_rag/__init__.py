"""
qa_rag — Hukuki QA Chatbot (RAG pipeline)

yargi-mcp semantik arama + NVIDIA LLM ile Türk hukuki sorulara
atatf referanslı cevaplar üretir.

v1.2.0: ChromaDB kalıcı vector store + token-aware chunker + BedestenIndexer.

Kullanım:
    from qa_rag import LegalQARAG
    rag = LegalQARAG()                       # chroma backend (default)
    await rag.load_corpora()                 # ilk sefer index (5-10 dk)
    answer = rag.ask("Muvazaalı tapu satışında mirasçı hangi davayı açar?")
"""

from .rag_engine import LegalQARAG, RAGResponse, RAGContext
from .llm_client import NvidiaLLMClient
from .prompts import SYSTEM_PROMPT_LEGAL, build_user_prompt
from .citations import format_citations, Citation
from .chunker import LegalChunker, Chunk, chunk_text
from .indexer import BedestenIndexer, IndexResult, IndexProgress

__version__ = "1.2.0"
__all__ = [
    "LegalQARAG",
    "RAGResponse",
    "RAGContext",
    "NvidiaLLMClient",
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
