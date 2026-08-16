"""
Semantic answer cache — ChromaDB-backed Q→A cache.

v1.3.0+: Aynı veya anlamsal olarak benzer sorular tekrar sorulduğunda,
LLM çağrısı yapmadan cache'den yanıt döner. Bu, NVIDIA LLM'in 60-240 saniye
latency'sini pratikte sıfıra indirir (sadece ilk soru yavaş, tekrarı ~50ms).

Çalışma prensibi:
  1. Soru geldi → NVIDIA nv-embed-v1 ile embed'le (query input_type)
  2. ChromaDB `qa_cache` collection'ında cosine search (top-1)
  3. Eğer skor >= threshold (default 0.92) → cache hit
     - Cached answer + cached citations döner
     - ~50ms (ChromaDB search) + ~1s (NVIDIA query embed)
  4. Değilse → LLM çağrısı yap, sonra Q+A+citations+embed'i cache'e yaz

Cache invalidation:
  - Cache temizleme: `cache.clear()`
  - Tek kayıt silme: `cache.invalidate(question_id)`
  - TTL: Henüz yok (hukuki soruların cevabı değişmez, karar metinleri sabit)

Kullanım:
    cache = AnswerCache(dimension=4096)
    hit = cache.lookup(question_embedding, threshold=0.92)
    if hit:
        return hit.answer, hit.citations
    # ... LLM call ...
    cache.store(question, question_embedding, answer, citations, metadata)

Env vars:
    RAG_ANSWER_CACHE         — "true"/"false" (default: true)
    RAG_CACHE_THRESHOLD      — cosine threshold (default: 0.92)
    RAG_CACHE_COLLECTION     — ChromaDB collection adı (default: qa_cache)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD = 0.92
DEFAULT_COLLECTION = "qa_cache"


@dataclass
class CacheHit:
    """Cache hit sonucu."""
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    question: str = ""
    score: float = 0.0
    cached_at: float = 0.0
    original_question: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    cache_id: str = ""


class AnswerCache:
    """
    ChromaDB-backed semantic answer cache.

    ChromaVectorStore ile aynı ChromaDB persistent client'ı paylaşır, ama
    ayrı bir collection kullanır (`qa_cache` default). Bu, decision collection
    ile cache'in birbirini etkilememesini sağlar.

    Performance:
      - lookup: ~50ms (ChromaDB cosine search, top-1)
      - store:  ~50ms (ChromaDB upsert)
      - Query embedding (NVIDIA) ~1s — bu RAG engine'in sorumluluğunda,
        cache sadece embedding'i alır.
    """

    def __init__(
        self,
        dimension: int = 4096,
        collection_name: Optional[str] = None,
        threshold: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.dimension = dimension
        self.collection_name = collection_name or os.getenv(
            "RAG_CACHE_COLLECTION", DEFAULT_COLLECTION
        )
        self.threshold = float(
            threshold if threshold is not None
            else os.getenv("RAG_CACHE_THRESHOLD", DEFAULT_THRESHOLD)
        )
        # Enabled default: true (v1.3.0+). Env var ile kapatılabilir.
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("RAG_ANSWER_CACHE", "true").lower() in (
                "true", "1", "yes", "on",
            )

        self._collection = None
        if self.enabled:
            self._init_collection()
            logger.info(
                f"AnswerCache aktif — collection='{self.collection_name}', "
                f"threshold={self.threshold}, dimension={dimension}, "
                f"mevcut kayıt={self.size()}"
            )
        else:
            logger.info("AnswerCache DEVRE DIŞI (RAG_ANSWER_CACHE=false)")

    def _init_collection(self):
        """ChromaDB collection'ı get-or-create."""
        # ChromaVectorStore ile aynı singleton client'ı paylaş
        from semantic_search.vector_store_chroma import _get_chroma_client
        from chromadb.config import Settings

        client = _get_chroma_client()

        try:
            self._collection = client.get_collection(name=self.collection_name)
            logger.info(
                f"Cache collection mevcut: {self.collection_name} "
                f"({self._collection.count()} kayıt)"
            )
        except Exception:
            self._collection = client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": str(self.dimension),
                    "type": "qa_cache",
                },
            )
            logger.info(
                f"Cache collection oluşturuldu: {self.collection_name}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        question_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[CacheHit]:
        """
        Verilen question embedding için cache'de benzer soru ara.

        Args:
            question_embedding: NVIDIA nv-embed-v1 query embedding (4096d)
            threshold: Override threshold (default: self.threshold)

        Returns:
            CacheHit veya None (cache miss / disabled / empty)
        """
        if not self.enabled or self._collection is None:
            return None

        if self._collection.count() == 0:
            return None

        thr = threshold if threshold is not None else self.threshold

        # ChromaDB query — top-1
        q = question_embedding.tolist() if isinstance(question_embedding, np.ndarray) else question_embedding
        if hasattr(q, "__len__") and len(q) > 0 and isinstance(q[0], list):
            q = q[0]  # 2D → 1D

        try:
            res = self._collection.query(
                query_embeddings=[q],
                n_results=1,
            )
        except Exception as e:
            logger.warning(f"Cache lookup hatası: {e}")
            return None

        ids_list = res.get("ids", [[]])[0]
        if not ids_list:
            return None

        docs_list = res.get("documents", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        dists_list = res.get("distances", [[]])[0]

        cid = ids_list[0]
        doc_text = docs_list[0] or ""
        meta = metas_list[0] or {}
        dist = float(dists_list[0])
        sim = 1.0 - dist  # cosine distance → similarity

        if sim < thr:
            logger.info(
                f"Cache miss — en yakın skor {sim:.4f} < threshold {thr}"
            )
            return None

        # Cache hit — meta'dan answer + citations'ı çıkar
        import json
        try:
            citations = json.loads(meta.get("citations_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            citations = []

        hit = CacheHit(
            answer=doc_text,
            citations=citations,
            question=meta.get("question", ""),
            score=sim,
            cached_at=float(meta.get("cached_at", 0)),
            original_question=meta.get("original_question", ""),
            llm_model=meta.get("llm_model", ""),
            llm_provider=meta.get("llm_provider", ""),
            cache_id=cid,
        )
        logger.info(
            f"Cache HIT — id={cid}, score={sim:.4f}, "
            f"cached {time.time() - hit.cached_at:.0f}s önce"
        )
        return hit

    def store(
        self,
        question: str,
        question_embedding: np.ndarray,
        answer: str,
        citations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Yeni Q→A çiftini cache'e yaz.

        Args:
            question: Orijinal kullanıcı sorusu
            question_embedding: Sorunun embedding'i (NVIDIA query)
            answer: LLM cevabı
            citations: Atıf listesi (RAGResponse.citations formatı)
            metadata: Ek metadata (llm_model, llm_provider, vb.)

        Returns:
            cache_id — ChromaDB'ye yazılan kayıt ID'si
        """
        if not self.enabled or self._collection is None:
            return ""

        import json

        # Cache ID — question hash'inden (aynı soru tekrar edilirse upsert olur)
        cache_id = f"qa_{abs(hash(question)) % (10**12)}"

        # Citation'lar JSON olarak metadata'da tutulur (Chroma meta primitive)
        meta = {
            "question": question,
            "original_question": question,
            "cached_at": time.time(),
            "citations_json": json.dumps(citations, ensure_ascii=False, default=str),
            "llm_model": (metadata or {}).get("llm_model", ""),
            "llm_provider": (metadata or {}).get("llm_provider", ""),
            "embedding_model": (metadata or {}).get("embedding_model", ""),
            "citation_count": len(citations),
            "answer_chars": len(answer),
        }

        # ChromaDB metadata primitive tipte olmalı
        clean_meta = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in meta.items()
            if v is not None
        }

        q = question_embedding.tolist() if isinstance(question_embedding, np.ndarray) else question_embedding
        if hasattr(q, "__len__") and len(q) > 0 and isinstance(q[0], list):
            q = q[0]

        try:
            self._collection.upsert(
                ids=[cache_id],
                embeddings=[q],
                documents=[answer],
                metadatas=[clean_meta],
            )
            logger.info(
                f"Cache STORE — id={cache_id}, answer={len(answer)} chars, "
                f"citations={len(citations)}, total={self._collection.count()}"
            )
        except Exception as e:
            logger.warning(f"Cache store hatası: {e}")
            return ""

        return cache_id

    def size(self) -> int:
        """Cache'teki kayıt sayısı."""
        if not self.enabled or self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear(self) -> int:
        """Tüm cache'i temizle. Silinen kayıt sayısını döndürür."""
        if not self.enabled or self._collection is None:
            return 0
        from semantic_search.vector_store_chroma import _get_chroma_client, _chroma_collections

        count = self._collection.count()
        client = _get_chroma_client()
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Cache temizlendi — {count} kayıt silindi")
        except Exception as e:
            logger.warning(f"Cache temizleme hatası: {e}")
            return 0

        # Cache collection'ı yeniden oluştur
        _chroma_collections.pop(self.collection_name, None)
        self._init_collection()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistikleri."""
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "collection": self.collection_name,
            "size": self.size(),
            "threshold": self.threshold,
            "dimension": self.dimension,
        }
