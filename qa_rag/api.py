"""
yargi-qa FastAPI app — Hukuki QA RAG pipeline HTTP API.

Endpoint'ler:
    POST /api/ask          — Senkron RAG (full response JSON)
    POST /api/ask/stream   — Streaming RAG (Server-Sent Events)
    POST /api/load         — Yeni corpus yükle (background)
    GET  /api/info         — Corpus & model bilgisi
    GET  /health           — Sağlık kontrolü

Kullanım:
    uvicorn qa_rag.api:app --host 0.0.0.0 --port 8001 --reload

Dokümantasyon:
    http://localhost:8001/docs  (Swagger UI)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)

# Global RAG instance (lifespan içinde initialize)
_rag: Optional[Any] = None


# --- Pydantic modelleri ---

class AskRequest(BaseModel):
    """RAG soru request modeli."""
    question: str = Field(..., min_length=5, max_length=2000,
                          description="Hukuki soru (Türkçe)")
    top_k: int = Field(5, ge=1, le=15,
                       description="LLM'e kaç karar feed'lenecek")
    temperature: float = Field(0.2, ge=0.0, le=1.5,
                               description="LLM temperature (0=belirleyici, 1=yaratıcı)")


class CitationInfo(BaseModel):
    document_id: str
    title: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_url: Optional[str] = None


class AskResponse(BaseModel):
    """RAG cevap modeli."""
    question: str
    answer: str
    citations: List[CitationInfo]
    context_decision_count: int
    embedding_model: str
    llm_model: str
    llm_usage: Dict[str, int]
    total_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float


class LoadRequest(BaseModel):
    """Yeni corpus yükleme request'i."""
    initial_keyword: str = Field(..., description="Bedesten arama keyword'ü")
    semantic_query: Optional[str] = Field(None,
                                          description="Semantik arama query'si (yoksa keyword kullanılır)")
    court_types: List[str] = Field(["YARGITAYKARARI"],
                                    description="Mahkeme tipleri")


class LoadResponse(BaseModel):
    status: str
    total_documents_processed: int
    documents_in_store: int
    failed_fetches: int
    load_time_ms: float


class InfoResponse(BaseModel):
    """Corpus & model bilgisi."""
    is_corpora_loaded: bool
    documents_in_store: int
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    llm_model: Optional[str] = None
    memory_usage_mb: Optional[float] = None


# --- Lifespan: startup/shutdown ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — startup'ta RAG initialize et."""
    global _rag

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Bedesten + embedding env defaults
    os.environ.setdefault("BEDESTEN_RATE_CAPACITY", "1")
    os.environ.setdefault("BEDESTEN_RATE_REFILL_S", "4.0")
    os.environ.setdefault("BEDESTEN_RATE_MAX_WAIT_S", "60")
    os.environ.setdefault("BEDESTEN_SEMANTIC_BATCH_SIZE", "30")
    os.environ.setdefault("BEDESTEN_SEMANTIC_MAX_RETRIES", "3")
    os.environ.setdefault("EMBEDDING_PROVIDER", "local")
    os.environ.setdefault("LOCAL_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
    os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
    os.environ.setdefault("LOCAL_EMBEDDING_DIMENSION", "4096")
    os.environ.setdefault("LOCAL_EMBEDDING_INPUT_TYPE", "auto")
    os.environ.setdefault("EMBEDDING_PROMPT_STYLE", "raw")

    # NVIDIA API key kontrolü
    if not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("LOCAL_EMBEDDING_API_KEY")):
        logger.warning("NVIDIA_API_KEY set değil — RAG initialize edilemedi")
        _rag = None
    else:
        try:
            from qa_rag.rag_engine import LegalQARAG
            _rag = LegalQARAG()
            logger.info("RAG engine initialize edildi (corpora henüz yüklenmedi)")
        except Exception as e:
            logger.error(f"RAG initialize hatası: {e}")
            _rag = None

    # Otomatik corpus yükle (opsiyonel)
    if os.environ.get("QA_AUTO_LOAD_CORPORA", "").lower() in ("1", "true", "yes"):
        if _rag is not None:
            logger.info("QA_AUTO_LOAD_CORPORA=1 — otomatik corpus yükleme başlatılıyor...")
            asyncio.create_task(_auto_load_corpora())

    yield

    # Shutdown
    logger.info("yargi-qa API kapatılıyor...")


async def _auto_load_corpora():
    """Background'da varsayılan corpus yükle."""
    global _rag
    try:
        await _rag.load_corpora()
        logger.info("Otomatik corpus yükleme tamam")
    except Exception as e:
        logger.error(f"Otomatik corpus yükleme hatası: {e}")


# --- FastAPI app ---

app = FastAPI(
    title="yargi-qa — Türk Hukuki QA API",
    description=(
        "Türk hukuki sorulara emsal karar referanslı cevaplar üreten RAG pipeline.\n\n"
        "**Pipeline:** Bedesten emsal kararları → NVIDIA nv-embed-v1 (4096d) → "
        "NVIDIA Nemotron LLM (70B) → atıflı cevap."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (frontend erişimi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # üretimde domain kısıtla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper ---

def _get_rag():
    global _rag
    if _rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine hazır değil. NVIDIA_API_KEY env var'ını set edin ve yeniden başlatın."
        )
    return _rag


# --- Endpoints ---

@app.get("/health")
async def health():
    """Sağlık kontrolü."""
    return {
        "status": "ok",
        "rag_ready": _rag is not None,
        "corpora_loaded": _rag.is_corpora_loaded if _rag else False,
    }


@app.get("/api/info", response_model=InfoResponse)
async def get_info():
    """Corpus & model bilgisi."""
    rag = _get_rag()
    info = {
        "is_corpora_loaded": rag.is_corpora_loaded,
        "documents_in_store": 0,
        "embedding_model": None,
        "embedding_dimension": None,
        "llm_model": None,
        "memory_usage_mb": None,
    }
    if rag._vector_store:
        stats = rag._vector_store.get_stats()
        info["documents_in_store"] = stats["num_documents"]
        info["embedding_dimension"] = stats["dimension"]
        info["memory_usage_mb"] = round(stats["memory_usage_mb"], 2)
    if rag._embedder:
        info["embedding_model"] = rag._embedder.model
    if rag._llm_client:
        info["llm_model"] = rag._llm_client.model
    return InfoResponse(**info)


@app.post("/api/load", response_model=LoadResponse)
async def load_corpora(req: LoadRequest):
    """Yeni corpus yükle (background değil — request bitene kadar bekler)."""
    rag = _get_rag()
    import time
    t0 = time.time()

    try:
        result = await rag.load_corpora(
            initial_keyword=req.initial_keyword,
            semantic_query=req.semantic_query or req.initial_keyword,
            court_types=req.court_types,
        )
        stats = result.get("stats", {})
        return LoadResponse(
            status=result.get("status", "unknown"),
            total_documents_processed=result.get("total_documents_processed", 0),
            documents_in_store=stats.get("documents_in_store", 0),
            failed_fetches=stats.get("failed_fetches", 0),
            load_time_ms=(time.time() - t0) * 1000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corpus yükleme hatası: {e}")


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Senkron RAG — tam cevap JSON olarak döner.

    Eğer corpora yüklenmemişse 400 döner (önce /api/load çağrılmalı).
    """
    rag = _get_rag()

    if not rag.is_corpora_loaded:
        raise HTTPException(
            status_code=400,
            detail="Corpora yüklenmemiş. Önce POST /api/load çağırın."
        )

    try:
        # top_k ve temperature override
        old_top_k = rag.top_k_retrieval
        old_temp = rag.llm_temperature
        if req.top_k != rag.top_k_retrieval:
            rag.top_k_retrieval = req.top_k
        if req.temperature != rag.llm_temperature:
            if rag._llm_client:
                rag._llm_client.temperature = req.temperature
            rag.llm_temperature = req.temperature

        response = await rag.ask(req.question)

        # Restore
        rag.top_k_retrieval = old_top_k
        rag.llm_temperature = old_temp
        if rag._llm_client:
            rag._llm_client.temperature = old_temp

        return AskResponse(
            question=response.question,
            answer=response.answer,
            citations=[CitationInfo(**c) for c in response.citations],
            context_decision_count=response.context_decision_count,
            embedding_model=response.embedding_model,
            llm_model=response.llm_model,
            llm_usage=response.llm_usage,
            total_time_ms=response.total_time_ms,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
        )
    except Exception as e:
        logger.exception(f"/api/ask hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming RAG — Server-Sent Events (SSE) olarak token token akıtır.

    Frontend tarafı (EventSource veya fetch+reader):
        const resp = await fetch('/api/ask/stream', {method: 'POST', ...});
        const reader = resp.body.getReader();
        ...
    """
    rag = _get_rag()

    if not rag.is_corpora_loaded:
        raise HTTPException(
            status_code=400,
            detail="Corpora yüklenmemiş. Önce POST /api/load çağırın."
        )

    # top_k override
    old_top_k = rag.top_k_retrieval
    if req.top_k != rag.top_k_retrieval:
        rag.top_k_retrieval = req.top_k

    async def event_generator():
        try:
            async for chunk in rag.ask_stream(req.question, top_k=req.top_k):
                # SSE formatı: "data: <chunk>\n\n"
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {type(e).__name__}: {e}\n\n"
        finally:
            rag.top_k_retrieval = old_top_k

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx için
        },
    )


@app.get("/")
async def root():
    """Kök endpoint — quick info."""
    return {
        "name": "yargi-qa API",
        "version": "1.0.0",
        "description": "Türk hukuki QA RAG pipeline",
        "docs": "/docs",
        "endpoints": {
            "POST /api/ask": "Senkron RAG",
            "POST /api/ask/stream": "Streaming RAG (SSE)",
            "POST /api/load": "Corpus yükle",
            "GET /api/info": "Corpus & model bilgisi",
            "GET /health": "Sağlık kontrolü",
        },
    }


# --- CLI runner ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(
        "qa_rag.api:app",
        host=host,
        port=port,
        reload=os.environ.get("RELOAD", "").lower() in ("1", "true"),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
