"""
Query embedding cache — NVIDIA API çağrısını önleyen çift katmanlı cache.

v1.4.0+: v1.3.0'daki bilinen kısıtlardan biri her sorguda ~1s NVIDIA query
embedding yapılmasıydı. Bu modül iki katmanlı cache ile bunu çözer:

  1. In-memory LRU (process içi, anında)
  2. ChromaDB persistent collection (process restart'ında kalır)

Her sorgu için akış:
  1. normalize_query(question) → "Mirasçı hangi davayı açar?" → "mirasci hangi davayi acar"
  2. cache_key = sha256(normalized)[:16]
  3. LRU'da ara → varsa dön (0 ms)
  4. ChromaDB'de ara (ID = cache_key) → varsa LRU'ya da yaz, dön (~5 ms)
  5. MISS → embedder.encode_query() çağır (~1 s)
  6. Sonucu hem LRU'ya hem ChromaDB'ye yaz

Cache key olarak normalized text kullanıyoruz (semantik embedding DEĞİL).
Bunun nedeni: aynı soruyu farklı phrasing'le sorduğunuzda farklı embedding
elde edilir (NVIDIA bunu yapar) — biz sadece "aynı soru → aynı embedding"
döngüsünü önlüyoruz. Semantik benzerlik answer_cache'in işi.

Env vars:
    RAG_QUERY_CACHE              — "true"/"false" (default: true)
    RAG_QUERY_CACHE_LRU_SIZE     — LRU boyutu (default: 256)
    RAG_QUERY_CACHE_COLLECTION   — ChromaDB collection adı (default: query_embed_cache)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_LRU_SIZE = 256
DEFAULT_COLLECTION = "query_embed_cache"


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

# Türkçe karakter → ASCII (mirasçı → mirasci, açar → acar)
_TR_ASCII_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})

# Çoklu boşluk + baş/son trim
_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """
    Sorguyu normalize et — küçük harf, Türkçe→ASCII, punct'ı temizle.

    Amaç: küçük farkları (büyük/küçük harf, baş/son boşluk, noktalama)
    eşit kabul edip cache hit oranını artırmak. Anlamsal eşleştirme
    YAPMAYIZ — bu sadece exact-match cache için.

    Order önemli: önce Türkçe karakterleri ASCII'ye çevir (yoksa lower()
    Türkçe İ'yi combining dot'lu 'i̇' yapar, translate kaçırır), sonra
    lowercase yap.

    Examples:
        "Mirasçı hangi davayı açar?"  →  "mirasci hangi davayi acar"
        "  mirasçı HANGİ davaYI açar " →  "mirasci hangi davayi acar"
        "Muvazaa ispat yükü kimde?"   →  "muvazaa ispat yuku kimde"
    """
    if not query:
        return ""
    # 1. Türkçe karakterleri ASCII'ye çevir (önce lower'dan önce, yoksa İ→i̇ combining dot)
    q = query.translate(_TR_ASCII_MAP)
    # 2. Küçük harf (artık Türkçe karakter yok, güvenli)
    q = q.lower()
    # 3. Noktalama işaretlerini boşluğa çevir (kelimeler arası tek boşluk)
    q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    # 4. Çoklu boşluğu tek boşluğa indir + trim
    q = _WS_RE.sub(" ", q).strip()
    return q


def cache_key(query: str) -> str:
    """Normalize edilmiş sorgudan 16-char hex cache key üret."""
    norm = normalize_query(query)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryCacheHit:
    """Query embedding cache hit sonucu."""
    embedding: np.ndarray
    query: str  # orijinal (normalize edilmemiş) sorgu
    cache_key: str
    source: str  # "lru" | "persistent" | "miss"
    cached_at: float = 0.0


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------


class QueryEmbeddingCache:
    """
    İki katmanlı query embedding cache: in-memory LRU + ChromaDB persistent.

    Performance (4096d NVIDIA nv-embed-v1):
      - LRU hit:         ~0.05 ms  (dict lookup)
      - Persistent hit:  ~5 ms     (ChromaDB get-by-id)
      - MISS:            ~1 s      (NVIDIA API call)

    Bir sorgu ilk kez sorulduğunda MISS olur (~1s). Aynı sorgu tekrar
    sorulduğunda LRU hit olur (~0 ms). Process restart'ından sonra
    ilk sorgu persistent hit olur (~5 ms) — yani NVIDIA'ya GİTMEZ.

    Kullanım:
        cache = QueryEmbeddingCache(dimension=4096)
        hit = cache.lookup(question)
        if hit is None:
            emb = embedder.encode_query(question)
            cache.store(question, emb)
        else:
            emb = hit.embedding
    """

    def __init__(
        self,
        dimension: int = 4096,
        collection_name: Optional[str] = None,
        lru_size: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        self.dimension = dimension
        self.collection_name = collection_name or os.getenv(
            "RAG_QUERY_CACHE_COLLECTION", DEFAULT_COLLECTION
        )
        self.lru_size = int(
            lru_size if lru_size is not None
            else os.getenv("RAG_QUERY_CACHE_LRU_SIZE", DEFAULT_LRU_SIZE)
        )
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("RAG_QUERY_CACHE", "true").lower() in (
                "true", "1", "yes", "on",
            )

        # In-memory LRU — OrderedDict for O(1) move_to_end
        self._lru: "OrderedDict[str, Tuple[np.ndarray, float]]" = OrderedDict()

        # Stats
        self._hits_lru = 0
        self._hits_persistent = 0
        self._misses = 0
        self._stores = 0

        self._collection = None
        if self.enabled:
            self._init_collection()
            logger.info(
                f"QueryEmbeddingCache aktif — collection='{self.collection_name}', "
                f"lru_size={self.lru_size}, dimension={dimension}, "
                f"persistent kayıt={self.size()}"
            )
        else:
            logger.info("QueryEmbeddingCache DEVRE DIŞI (RAG_QUERY_CACHE=false)")

    def _init_collection(self):
        """ChromaDB collection'ı get-or-create (answer_cache ile aynı pattern)."""
        from semantic_search.vector_store_chroma import _get_chroma_client

        client = _get_chroma_client()

        try:
            self._collection = client.get_collection(name=self.collection_name)
            logger.info(
                f"Query cache collection mevcut: {self.collection_name} "
                f"({self._collection.count()} kayıt)"
            )
        except Exception:
            self._collection = client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": str(self.dimension),
                    "type": "query_embed_cache",
                },
            )
            logger.info(
                f"Query cache collection oluşturuldu: {self.collection_name}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> Optional[QueryCacheHit]:
        """
        Verilen sorgu için cache'de embedding ara.

        Args:
            query: Kullanıcının orijinal sorusu (normalize edilmiş olabilir, edilmemiş olabilir)

        Returns:
            QueryCacheHit veya None (cache miss / disabled / empty)
        """
        if not self.enabled:
            return None

        if not query or not query.strip():
            return None

        key = cache_key(query)

        # 1. LRU lookup — O(1)
        if key in self._lru:
            emb, cached_at = self._lru[key]
            self._lru.move_to_end(key)  # LRU: recently used
            self._hits_lru += 1
            logger.debug(f"Query cache LRU HIT — key={key}")
            return QueryCacheHit(
                embedding=emb,
                query=query,
                cache_key=key,
                source="lru",
                cached_at=cached_at,
            )

        # 2. Persistent (ChromaDB) lookup — by ID
        if self._collection is not None and self._collection.count() > 0:
            try:
                # include="embeddings" gerekli — ChromaDB default olarak embeddings DÖNDÜRMEZ
                res = self._collection.get(ids=[key], include=["embeddings", "metadatas"])
                if res and res.get("ids"):
                    # Embedding'i çıkar
                    embeddings = res.get("embeddings", [])
                    if embeddings is not None and len(embeddings) > 0:
                        emb = np.array(embeddings[0], dtype=np.float32)
                        # L2 normalize (defensive — ChromaDB storage bazen bozabilir)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm

                        # LRU'ya da yaz
                        self._lru[key] = (emb, time.time())
                        self._evict_lru()

                        self._hits_persistent += 1
                        logger.debug(f"Query cache PERSISTENT HIT — key={key}")

                        # Metadata'dan cached_at al
                        metas = res.get("metadatas", [])
                        cached_at = float(metas[0].get("cached_at", 0)) if metas else 0.0

                        return QueryCacheHit(
                            embedding=emb,
                            query=query,
                            cache_key=key,
                            source="persistent",
                            cached_at=cached_at,
                        )
            except Exception as e:
                logger.warning(f"Query cache persistent lookup hatası: {e}")

        # 3. MISS
        self._misses += 1
        logger.debug(f"Query cache MISS — key={key}, query='{query[:50]}'")
        return None

    def store(self, query: str, embedding: np.ndarray) -> str:
        """
        Yeni (query → embedding) çiftini cache'e yaz (LRU + persistent).

        Args:
            query: Kullanıcının orijinal sorusu
            embedding: NVIDIA query embedding (L2 normalized)

        Returns:
            cache_key — yazılan kayıt ID'si
        """
        if not self.enabled:
            return ""

        if not query or not query.strip():
            return ""

        key = cache_key(query)

        # Embedding'i 1D'ye indir (defensive)
        emb = embedding
        if isinstance(emb, np.ndarray) and emb.ndim == 2:
            emb = emb[0]

        # L2 normalize (defensive — store'da da normalize edelim)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        emb = emb.astype(np.float32)

        # 1. LRU'ya yaz
        self._lru[key] = (emb, time.time())
        self._evict_lru()

        # 2. ChromaDB'ye yaz
        if self._collection is not None:
            try:
                emb_list = emb.tolist()
                meta = {
                    "query": query,
                    "normalized": normalize_query(query),
                    "cached_at": time.time(),
                    "dimension": int(self.dimension),
                    "model": "nvidia/nv-embed-v1",  # bilgi amaçlı
                }
                self._collection.upsert(
                    ids=[key],
                    embeddings=[emb_list],
                    documents=[query],  # orijinal sorguyu sakla
                    metadatas=[meta],
                )
            except Exception as e:
                logger.warning(f"Query cache persistent store hatası: {e}")

        self._stores += 1
        logger.debug(
            f"Query cache STORE — key={key}, query='{query[:50]}', "
            f"lru_size={len(self._lru)}"
        )
        return key

    def _evict_lru(self):
        """LRU boyutunu sınırda tut — en eski kayıtları çıkar."""
        while len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)  # FIFO for LRU

    # ------------------------------------------------------------------
    # Stats / management
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Persistent cache'teki kayıt sayısı."""
        if not self.enabled or self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def lru_size_current(self) -> int:
        """LRU'daki kayıt sayısı."""
        return len(self._lru)

    def clear(self) -> int:
        """Hem LRU hem persistent cache'i temizle. Silinen toplam kayıt sayısı."""
        lru_count = len(self._lru)
        self._lru.clear()

        persistent_count = 0
        if self.enabled and self._collection is not None:
            from semantic_search.vector_store_chroma import (
                _get_chroma_client,
                _chroma_collections,
            )
            persistent_count = self._collection.count()
            client = _get_chroma_client()
            try:
                client.delete_collection(self.collection_name)
                logger.info(
                    f"Query cache temizlendi — LRU {lru_count} + persistent "
                    f"{persistent_count} kayıt silindi"
                )
            except Exception as e:
                logger.warning(f"Query cache temizleme hatası: {e}")
                return lru_count

            _chroma_collections.pop(self.collection_name, None)
            self._init_collection()

        # Stats reset
        self._hits_lru = 0
        self._hits_persistent = 0
        self._misses = 0
        self._stores = 0

        return lru_count + persistent_count

    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistikleri."""
        total_hits = self._hits_lru + self._hits_persistent
        total_requests = total_hits + self._misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        return {
            "enabled": self.enabled,
            "collection": self.collection_name if self.enabled else None,
            "lru_size_current": len(self._lru),
            "lru_size_max": self.lru_size,
            "persistent_size": self.size(),
            "dimension": self.dimension,
            "hits_lru": self._hits_lru,
            "hits_persistent": self._hits_persistent,
            "misses": self._misses,
            "stores": self._stores,
            "hit_rate_pct": round(hit_rate, 1),
            "total_requests": total_requests,
        }
