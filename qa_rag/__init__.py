"""
qa_rag — Hukuki QA Chatbot (RAG pipeline)

yargi-mcp semantik arama + NVIDIA LLM ile Türk hukuki sorulara
atatf referanslı cevaplar üretir.

Kullanım:
    from qa_rag import LegalQARAG
    rag = LegalQARAG()
    answer = rag.ask("Muvazaalı tapu satışında mirasçı hangi davayı açar?")
"""

from .rag_engine import LegalQARAG, RAGResponse, RAGContext
from .llm_client import NvidiaLLMClient
from .prompts import SYSTEM_PROMPT_LEGAL, build_user_prompt
from .citations import format_citations, Citation

__version__ = "1.0.0"
__all__ = [
    "LegalQARAG",
    "RAGResponse",
    "RAGContext",
    "NvidiaLLMClient",
    "SYSTEM_PROMPT_LEGAL",
    "build_user_prompt",
    "format_citations",
    "Citation",
]
